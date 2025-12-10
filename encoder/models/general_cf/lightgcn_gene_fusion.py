import torch as t
from torch import nn
import torch.nn.functional as F  # NEW

from config.configurator import configs
from models.aug_utils import NodeMask
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
from models.loss_utils import cal_bpr_loss, reg_params, ssl_con_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_gene_Fusion(BaseModel):
    """
    LightGCN_gene with:
      - two graphs (long-term & short-term),
      - one set of ID embeddings (users/items),
      - GCN propagation on both graphs,
      - interaction-aware fusion of long/short user embeddings,
      - fixed 0.5–0.5 fusion of long/short item embeddings,
      - generative reconstruction loss on fused node embeddings:
          * user -> long-term text
          * user -> short-term text
          * item -> item text.
    """

    def __init__(self, data_handler):
        super(LightGCN_gene_Fusion, self).__init__(data_handler)

        device = configs['device']

        # Two graphs
        self.adj_long = data_handler.torch_adj_long
        self.adj_short = data_handler.torch_adj_short
        self.keep_rate = configs['model']['keep_rate']

        # User & item ID embeddings (shared across both graphs)
        self.user_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size))
        )
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # Hyper-parameters (GCN & generative)
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']

        # ----- fusion hyperparameters (match plus model) -----
        self.fusion_type = self.hyper_config.get("fusion_type", "weighted_sum")
        self.alpha_mode = self.hyper_config.get("alpha_mode", "global")  # "global" or "interaction"
        self.norm_before_fusion = self.hyper_config.get("norm_before_fusion", False)
        self.alpha_vec = None  # for interaction-aware α(u)

        if self.fusion_type == "weighted_sum":
            if self.alpha_mode == "global":
                # Fixed scalar α: alpha * u_ST + (1 - alpha) * u_LT
                self.alpha = float(self.hyper_config.get("alpha", 0.5))
            elif self.alpha_mode == "interaction":
                # Per-user α(u) from interaction counts (same as plus)
                user_inter_num_np = data_handler.user_inter_num  # numpy [num_users]
                user_inter_num = t.from_numpy(user_inter_num_np).to(device).float()  # [U]

                # α(u) = 1 - normalized log(#interactions)
                log_inter = t.log1p(user_inter_num)        # [U]
                min_v = log_inter.min()
                max_v = log_inter.max()
                if max_v > min_v:
                    base = (log_inter - min_v) / (max_v - min_v + 1e-8)  # in [0,1]
                else:
                    base = t.zeros_like(log_inter)

                alpha_vec = 1.0 - base          # [U], more recent / fewer interactions -> larger α
                alpha_vec = t.clamp(alpha_vec, 0.0, 1.0)
                self.alpha_vec = alpha_vec.unsqueeze(-1)  # [U, 1] for broadcasting
                self.alpha = None  # no global scalar
            else:
                raise ValueError(f"Unknown alpha_mode: {self.alpha_mode}")
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # ----- semantic text embeddings (NOW LT + ST for users) -----  # NEW
        self.usrprf_long_embeds = t.tensor(
            configs['usrprf_embeds'], dtype=t.float32, device=device
        )   # [U, text_dim]
        self.usrprf_short_embeds = t.tensor(
            configs['usrprf_short_embeds'], dtype=t.float32, device=device
        )   # [U, text_dim]
        self.itmprf_embeds = t.tensor(
            configs['itmprf_embeds'], dtype=t.float32, device=device
        )   # [I, text_dim]

        # Sanity: require LT and ST text dims to match
        text_dim = self.usrprf_long_embeds.shape[1]
        assert self.usrprf_short_embeds.shape[1] == text_dim, \
            "Long-term and short-term user text dims must match"

        # Masker still operates in ID-embedding space
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)

        # Separate decoders:
        #   - user -> long-term text
        #   - user -> short-term text
        #   - item -> item text
        self.mlp_user_long = nn.Sequential(
            nn.Linear(self.embedding_size, (text_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((text_dim + self.embedding_size) // 2, text_dim),
        )
        self.mlp_user_short = nn.Sequential(
            nn.Linear(self.embedding_size, (text_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((text_dim + self.embedding_size) // 2, text_dim),
        )
        self.mlp_item = nn.Sequential(
            nn.Linear(self.embedding_size, (text_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((text_dim + self.embedding_size) // 2, text_dim),
        )

        self._init_weight()

        print("Gene Fusion hyper-config:", self.hyper_config)
        print(
            "fusion_type:", self.fusion_type,
            "alpha_mode:", self.alpha_mode,
            "norm_before_fusion:", self.norm_before_fusion,
        )

    def _init_weight(self):
        for m in list(self.mlp_user_long) + list(self.mlp_user_short) + list(self.mlp_item):
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _mask(self):
        """
        Mask a subset of nodes (users + items) in the *input* ID embedding space.
        """
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

    def _gcn_forward(self, adj, user_embeds, item_embeds, keep_rate=1.0):
        """
        One LightGCN pass on a given graph, starting from given user/item embeddings.
        """
        embeds = t.concat([user_embeds, item_embeds], dim=0)
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
        Fuse long- and short-term user embeddings, using the same logic as LightGCN_plus_Fusion:
          - optional L2-normalization (norm_before_fusion)
          - global α or interaction-aware α(u)
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
        else:
            raise ValueError(f"Unknown fusion_type in _fuse_users: {self.fusion_type}")

    def _fuse_nodes(self, user_long, item_long, user_short, item_short):
        """
        Fuse users with α(u) as above; fuse items with fixed 0.5–0.5.
        """
        user_fused = self._fuse_users(user_long, user_short)
        item_fused = 0.5 * item_long + 0.5 * item_short
        return user_fused, item_fused

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _reconstruction(self, fused_all_embeds, seeds):
        """
        Generative reconstruction loss:

          - if seed is a user node:
              use fused user embedding to reconstruct BOTH
                * long-term user text
                * short-term user text
          - if seed is an item node:
              reconstruct item text

        fused_all_embeds: [num_users + num_items, d]
        seeds: LongTensor [num_masked_nodes], indices in that concatenation.
        """
        if seeds.numel() == 0:
            # No masked nodes; return zero with correct device & dtype
            return fused_all_embeds.sum() * 0.0

        num_users = self.user_num
        # Boolean masks
        is_user = seeds < num_users
        is_item = ~is_user

        total_loss = 0.0
        parts = 0

        # ----- user reconstruction: LT + ST -----
        if is_user.any():
            user_seeds = seeds[is_user]                      # indices in [0, num_users)
            enc_user = fused_all_embeds[user_seeds]          # [Nu, d]

            # Long-term target
            tgt_long = self.usrprf_long_embeds[user_seeds]   # [Nu, text_dim]
            pred_long = self.mlp_user_long(enc_user)         # [Nu, text_dim]
            loss_long = ssl_con_loss(pred_long, tgt_long, self.re_temperature)

            # Short-term target
            tgt_short = self.usrprf_short_embeds[user_seeds] # [Nu, text_dim]
            pred_short = self.mlp_user_short(enc_user)       # [Nu, text_dim]
            loss_short = ssl_con_loss(pred_short, tgt_short, self.re_temperature)

            total_loss += (loss_long + loss_short)
            parts += 2

        # ----- item reconstruction -----
        if is_item.any():
            item_seeds = seeds[is_item]                      # indices in [num_users, num_users+num_items)
            item_idx = item_seeds - num_users                # [Ni]
            enc_item = fused_all_embeds[item_seeds]          # [Ni, d]
            tgt_item = self.itmprf_embeds[item_idx]          # [Ni, text_dim]
            pred_item = self.mlp_item(enc_item)              # [Ni, text_dim]
            loss_item = ssl_con_loss(pred_item, tgt_item, self.re_temperature)

            total_loss += loss_item
            parts += 1

        # Average over the number of distinct reconstruction “heads” we used
        if parts > 0:
            total_loss = total_loss / parts

        return total_loss

    # -------- forward: inference / eval (no masking) --------
    def forward(self, keep_rate=1.0):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        # Use *unmasked* ID embeddings at inference
        user_long, item_long = self._gcn_forward(
            self.adj_long, self.user_embeds, self.item_embeds, keep_rate
        )
        user_short, item_short = self._gcn_forward(
            self.adj_short, self.user_embeds, self.item_embeds, keep_rate
        )

        user_fused, item_fused = self._fuse_nodes(user_long, item_long, user_short, item_short)

        self.final_embeds = t.concat([user_fused, item_fused], dim=0)
        return user_fused, item_fused

    # -------- training: BPR + reconstruction + reg --------
    def cal_loss(self, batch_data):
        self.is_training = True

        # 1) Mask nodes in input ID embedding space
        masked_user_embeds, masked_item_embeds, seeds = self._mask()

        # 2) GCN on both graphs with *masked* inputs
        user_long, item_long = self._gcn_forward(
            self.adj_long, masked_user_embeds, masked_item_embeds, self.keep_rate
        )
        user_short, item_short = self._gcn_forward(
            self.adj_short, masked_user_embeds, masked_item_embeds, self.keep_rate
        )

        # 3) Fuse long & short outputs for BPR and reconstruction
        user_fused, item_fused = self._fuse_nodes(user_long, item_long, user_short, item_short)

        # 4) BPR loss on fused embeddings
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_fused, item_fused, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        # 5) Reconstruction loss on fused node embeddings (LT user + ST user + items)
        fused_all = t.concat([user_fused, item_fused], dim=0)
        recon_loss = self.recon_weight * self._reconstruction(fused_all, seeds)

        # 6) L2 regularization
        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + recon_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'recon_loss': recon_loss
        }

        # Log α for sanity
        if self.fusion_type == "weighted_sum":
            if self.alpha_mode == "global" and self.alpha is not None:
                losses["alpha"] = max(0.0, min(1.0, float(self.alpha)))
            elif self.alpha_mode == "interaction" and self.alpha_vec is not None:
                alpha_u = self.alpha_vec.squeeze(-1)
                losses["alpha_mean"] = float(alpha_u.mean().item())

        # Invalidate cache so eval recomputes fresh embeddings
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
