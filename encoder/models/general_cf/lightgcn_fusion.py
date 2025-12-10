import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_Fusion(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_Fusion, self).__init__(data_handler)

        device = configs['device']

        # Two graphs (long-term & short-term)
        self.adj_long = data_handler.torch_adj_long
        self.adj_short = data_handler.torch_adj_short

        self.keep_rate = configs['model']['keep_rate']

        # Modes / hyperparams (dataset-specific via hyper_config)
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.fusion_type = self.hyper_config.get('fusion_type', 'weighted_sum')
        self.branch_mode = self.hyper_config.get('branch_mode', 'fused')   # 'long_only' | 'short_only' | 'fused'
        self.share_user_embeds = self.hyper_config.get('share_user_embeds', True)

        # Match plus-model interface
        self.alpha_mode = self.hyper_config.get('alpha_mode', 'global')   # 'global' or 'interaction'
        self.norm_before_fusion = self.hyper_config.get('norm_before_fusion', False)

        # user_inter_num only needed for interaction-aware α
        self.alpha_vec = None   # [num_users, 1] if interaction mode

        # Embeddings
        if self.share_user_embeds:
            self.user_embeds = nn.Parameter(
                init(t.empty(self.user_num, self.embedding_size))
            )
        else:
            self.user_embeds_long = nn.Parameter(
                init(t.empty(self.user_num, self.embedding_size))
            )
            self.user_embeds_short = nn.Parameter(
                init(t.empty(self.user_num, self.embedding_size))
            )

        # Items always shared
        self.item_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size))
        )

        # Fusion params (weighted sum only)
        if self.fusion_type == 'weighted_sum':
            if self.alpha_mode == 'global':
                # Global scalar α (learnable)  -> α * u_ST + (1 - α) * u_LT
                init_alpha = float(self.hyper_config.get('alpha', 0.5))
                self.alpha = nn.Parameter(t.tensor([init_alpha], dtype=t.float32, device=device))
            elif self.alpha_mode == 'interaction':
                # Per-user α(u) from interaction counts
                user_inter_num_np = data_handler.user_inter_num  # numpy [num_users]
                user_inter_num = t.from_numpy(user_inter_num_np).to(device).float()  # [U]

                # α(u) = 1 - normalized log( #interactions )
                log_inter = t.log1p(user_inter_num)      # [U]
                min_v = log_inter.min()
                max_v = log_inter.max()
                if max_v > min_v:
                    base = (log_inter - min_v) / (max_v - min_v + 1e-8)  # in [0,1]
                else:
                    base = t.zeros_like(log_inter)

                alpha_vec = 1.0 - base          # [U]
                alpha_vec = t.clamp(alpha_vec, 0.0, 1.0)
                self.alpha_vec = alpha_vec.unsqueeze(-1)  # [U, 1] for broadcasting
                self.alpha = None
            else:
                raise ValueError(f"Unknown alpha_mode: {self.alpha_mode}")
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        print("Hyper-config:", self.hyper_config)
        print(
            "branch_mode:", self.branch_mode,
            "fusion_type:", self.fusion_type,
            "alpha_mode:", self.alpha_mode,
            "norm_before_fusion:", self.norm_before_fusion,
        )

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _gcn_forward(self, adj, user_embeds, keep_rate=1.0):
        """Run LightGCN over one graph given a user embedding table."""
        embeds = t.concat([user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _compute_branches(self, keep_rate):
        """Compute LT and ST branch embeddings."""
        if self.share_user_embeds:
            u_long_base = self.user_embeds
            u_short_base = self.user_embeds
        else:
            u_long_base = self.user_embeds_long
            u_short_base = self.user_embeds_short

        u_long, i_long = self._gcn_forward(self.adj_long, u_long_base, keep_rate)
        u_short, i_short = self._gcn_forward(self.adj_short, u_short_base, keep_rate)
        return u_long, i_long, u_short, i_short

    def _fuse_users(self, u_long, u_short):
        """Fuse user embeddings according to branch_mode and fusion_type."""
        if self.branch_mode == 'long_only':
            return u_long
        if self.branch_mode == 'short_only':
            return u_short

        # fused mode
        if self.fusion_type != 'weighted_sum':
            raise ValueError(f"Unknown fusion_type in _fuse_users: {self.fusion_type}")

        # Optional L2 normalization of each branch (norm_before_fusion)
        if self.norm_before_fusion:
            u_long = F.normalize(u_long, p=2, dim=-1)
            u_short = F.normalize(u_short, p=2, dim=-1)

        if self.alpha_mode == 'global':
            alpha = t.clamp(self.alpha, 0.0, 1.0)
            return alpha * u_short + (1.0 - alpha) * u_long
        elif self.alpha_mode == 'interaction':
            # alpha_vec: [num_users, 1]
            alpha = self.alpha_vec  # already in [0,1]
            return alpha * u_short + (1.0 - alpha) * u_long
        else:
            raise ValueError(f"Unknown alpha_mode in _fuse_users: {self.alpha_mode}")

    def forward(self, keep_rate=1.0):
        """
        Forward that returns the fused user embedding (used for inference)
        and long-term item embeddings.
        """
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        if self.branch_mode in ['long_only', 'short_only']:
            # Single-branch modes (no fusion)
            if self.share_user_embeds:
                u_base = self.user_embeds
            else:
                u_base = (
                    self.user_embeds_long
                    if self.branch_mode == 'long_only'
                    else self.user_embeds_short
                )

            adj = self.adj_long if self.branch_mode == 'long_only' else self.adj_short
            user_embeds, item_embeds = self._gcn_forward(adj, u_base, keep_rate)

        else:
            # Fused mode: compute both branches and fuse
            u_long, i_long, u_short, i_short = self._compute_branches(keep_rate)
            u_fused = self._fuse_users(u_long, u_short)
            user_embeds = u_fused
            item_embeds = i_long  # keep items from LT graph

        self.final_embeds = t.concat([user_embeds, item_embeds], dim=0)
        return user_embeds, item_embeds

    def cal_loss(self, batch_data):
        """
        Single BPR loss on the user embedding that will be used at inference.
        """
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss

        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}

        # Logging alpha
        if self.fusion_type == 'weighted_sum':
            if self.alpha_mode == 'global' and self.alpha is not None:
                losses['alpha'] = float(t.clamp(self.alpha, 0.0, 1.0).item())
            elif self.alpha_mode == 'interaction' and self.alpha_vec is not None:
                alpha_u = self.alpha_vec.squeeze(-1)
                losses['alpha_mean'] = float(alpha_u.mean().item())

        return loss, losses

    def full_predict(self, batch_data):
        """
        Inference uses the fused (or single-branch) user embedding,
        but diagnostics still look at LT / ST separately (in fused mode).
        """
        self.is_training = False
        user_embeds, item_embeds = self.forward(1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)

        # ---- Diagnostics: LT vs ST ----
        if self.branch_mode == 'fused':
            with t.no_grad():
                u_long, i_long, u_short, i_short = self._compute_branches(keep_rate=1.0)

                norm_long = u_long.norm(dim=-1)
                norm_short = u_short.norm(dim=-1)
                cos = F.cosine_similarity(u_long, u_short, dim=-1)

                def stats(x):
                    return x.mean().item(), x.std().item()

                nL_mean, nL_std = stats(norm_long)
                nS_mean, nS_std = stats(norm_short)
                cos_mean, cos_std = stats(cos)

                q = t.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=cos.device)
                cos_q = cos.quantile(q).cpu().tolist()

                print("\n=== LT vs ST embedding diagnostics (ID-only) ===")
                print(f"LT norm   mean={nL_mean:.4f}, std={nL_std:.4f}")
                print(f"ST norm   mean={nS_mean:.4f}, std={nS_std:.4f}")
                print(f"cos(LT,ST) mean={cos_mean:.4f}, std={cos_std:.4f}")
                print("cos(LT,ST) quantiles (0.1,0.25,0.5,0.75,0.9):")
                print(" ", ", ".join(f"{v:.4f}" for v in cos_q))

                if self.alpha_mode == 'global' and self.alpha is not None:
                    alpha_val = float(t.clamp(self.alpha, 0.0, 1.0).item())
                    print("alpha (global short weight):", alpha_val)
                elif self.alpha_mode == 'interaction' and self.alpha_vec is not None:
                    alpha_u = self.alpha_vec.squeeze(-1)
                    print(
                        "alpha_u (per-user short weight): "
                        f"mean={alpha_u.mean().item():.4f}, "
                        f"min={alpha_u.min().item():.4f}, "
                        f"max={alpha_u.max().item():.4f}"
                    )

                print("======================================\n")

        return full_preds
