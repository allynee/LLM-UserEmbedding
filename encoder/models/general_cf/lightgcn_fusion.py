import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_Fusion(BaseModel):
    """
    ID-only LightGCN with Short-term + Long-term Fusion.

    - Two graphs: adj_long (all interactions), adj_short (last-K interactions)
    - Two user ID embeddings: user_embeds_long, user_embeds_short
    - Shared item ID embeddings
    - Fusion of user embeddings (weighted sum or FFN), with:
        * alpha_mode = "global": scalar α
        * alpha_mode = "interaction": per-user α(u) from interaction counts
        * optional L2 norm-before-fusion
    """

    def __init__(self, data_handler):
        super(LightGCN_Fusion, self).__init__(data_handler)

        device = configs["device"]

        # Graphs
        self.adj_long = data_handler.torch_adj_long
        self.adj_short = data_handler.torch_adj_short
        self.keep_rate = configs["model"]["keep_rate"]

        # ID embeddings
        self.user_embeds_long = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.user_embeds_short = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # ----- core hyper-parameters -----
        self.layer_num = self.hyper_config["layer_num"]
        self.reg_weight = self.hyper_config["reg_weight"]

        # ----- fusion hyper-parameters (mirror plus fusion) -----
        self.fusion_type = self.hyper_config.get("fusion_type", "weighted_sum")
        self.alpha_mode = self.hyper_config.get("alpha_mode", "global")  # "global" or "interaction"
        self.alpha_vec = None  # for interaction-aware α
        self.norm_before_fusion = self.hyper_config.get("norm_before_fusion", False)

        if self.fusion_type == "weighted_sum":
            if self.alpha_mode == "global":
                # Fixed scalar α: alpha * u_ST + (1 - alpha) * u_LT
                self.alpha = float(self.hyper_config.get("alpha", 0.5))
                self.fusion_ffn = None
            elif self.alpha_mode == "interaction":
                # Build per-user α(u) from interaction counts (same logic as plus)
                user_inter_num_np = data_handler.user_inter_num  # numpy [num_users]
                user_inter_num = t.from_numpy(user_inter_num_np).to(device).float()  # [U]

                # α(u) = 1 - normalized log(#interactions)
                log_inter = t.log1p(user_inter_num)  # [U]
                min_v = log_inter.min()
                max_v = log_inter.max()
                if max_v > min_v:
                    base = (log_inter - min_v) / (max_v - min_v + 1e-8)  # in [0,1]
                else:
                    base = t.zeros_like(log_inter)

                alpha_vec = 1.0 - base  # [U], fewer interactions -> larger α (more ST weight)
                alpha_vec = t.clamp(alpha_vec, 0.0, 1.0)
                self.alpha_vec = alpha_vec.unsqueeze(-1)  # [U, 1] for broadcasting
                self.alpha = None  # no global scalar
                self.fusion_ffn = None
            else:
                raise ValueError(f"Unknown alpha_mode: {self.alpha_mode}")

        elif self.fusion_type == "ffn":
            self.alpha = None
            self.fusion_ffn = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        self._init_weight()

        print("LightGCN_Fusion hyper-config:", self.hyper_config)
        print(
            "fusion_type:", self.fusion_type,
            "alpha_mode:", self.alpha_mode,
            "norm_before_fusion:", self.norm_before_fusion,
        )

    def _init_weight(self):
        if self.fusion_ffn is not None:
            for m in self.fusion_ffn:
                if isinstance(m, nn.Linear):
                    init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _gcn_forward(self, adj, user_embeds, keep_rate=1.0):
        """
        One LightGCN pass on a given graph, starting from given user embeddings
        and shared item embeddings.
        """
        embeds = t.concat([user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _fuse_users(self, user_long, user_short):
        """
        Fuse long- & short-term user IDs.
        Matches the fusion behavior of LightGCN_plus_Fusion.
        """
        # Optional L2 normalization of each branch to fix norm imbalance
        if self.norm_before_fusion:
            user_long = F.normalize(user_long, p=2, dim=-1)
            user_short = F.normalize(user_short, p=2, dim=-1)

        if self.fusion_type == "weighted_sum":
            if self.alpha_mode == "global":
                alpha = max(0.0, min(1.0, float(self.alpha)))
                return alpha * user_short + (1.0 - alpha) * user_long
            elif self.alpha_mode == "interaction":
                # alpha_vec: [num_users, 1]
                alpha = self.alpha_vec  # already in [0,1]
                return alpha * user_short + (1.0 - alpha) * user_long
            else:
                raise ValueError(f"Unknown alpha_mode in _fuse_users: {self.alpha_mode}")
        else:  # "ffn"
            user_cat = t.cat([user_long, user_short], dim=-1)
            return self.fusion_ffn(user_cat)

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    # -------- forward: used for inference / eval --------
    def forward(self, keep_rate=1.0):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        # GCN on both graphs
        user_long, item_long = self._gcn_forward(
            self.adj_long, self.user_embeds_long, keep_rate
        )
        user_short, item_short = self._gcn_forward(
            self.adj_short, self.user_embeds_short, keep_rate
        )

        # Fuse users; for items, use average of long & short (or switch to item_long if desired)
        user_embeds = self._fuse_users(user_long, user_short)
        item_embeds = (item_long + item_short) / 2.0

        self.final_embeds = t.concat([user_embeds, item_embeds], dim=0)
        return user_embeds, item_embeds

    # -------- training: BPR + reg --------
    def cal_loss(self, batch_data):
        self.is_training = True

        # Re-run with dropout for training
        user_long, item_long = self._gcn_forward(
            self.adj_long, self.user_embeds_long, self.keep_rate
        )
        user_short, item_short = self._gcn_forward(
            self.adj_short, self.user_embeds_short, self.keep_rate
        )

        user_embeds = self._fuse_users(user_long, user_short)
        item_embeds = (item_long + item_short) / 2.0

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            user_embeds, item_embeds, batch_data
        )

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss

        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
        }

        # Log α like in plus
        if self.fusion_type == "weighted_sum":
            if self.alpha_mode == "global" and self.alpha is not None:
                losses["alpha"] = max(0.0, min(1.0, float(self.alpha)))
            elif self.alpha_mode == "interaction" and self.alpha_vec is not None:
                alpha_u = self.alpha_vec.squeeze(-1)
                losses["alpha_mean"] = float(alpha_u.mean().item())

        # Invalidate cache so eval recomputes them
        self.final_embeds = None
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(keep_rate=1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
