import torch as t
from torch import nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_plus_Fusion(BaseModel):
    """
    LightGCN_plus with:
      - two graphs (long-term & short-term),
      - two user ID embeddings (long & short),
      - averaged item embeddings from both graphs,
      - duplicated KD alignment for long & short user branches,
      - item KD as in original plus (but on fused items)
      - fusion of user embeddings at the end (weighted sum or FFN)
    """

    def __init__(self, data_handler):
        super(LightGCN_plus_Fusion, self).__init__(data_handler)

        # Two graphs
        self.adj_long = data_handler.torch_adj_long
        self.adj_short = data_handler.torch_adj_short
        self.keep_rate = configs['model']['keep_rate']

        # User & item parameters
        self.user_embeds_long = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.user_embeds_short = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # Hyper-parameters
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

        device = configs['device']

        # ---- semantic user/item embeddings (long & short for users) ----
        # long-term user profiles (as in original plus)
        self.usrprf_long_embeds = t.tensor(
            configs['usrprf_embeds'], dtype=t.float32, device=device
        )
        # short-term user profiles
        self.usrprf_short_embeds = t.tensor(
            configs['usrprf_short_embeds'], dtype=t.float32, device=device
        )
        # item profiles (same as original plus)
        self.itmprf_embeds = t.tensor(
            configs['itmprf_embeds'], dtype=t.float32, device=device
        )

        text_dim = self.usrprf_long_embeds.shape[1]

        # Shared MLP to project text space -> GCN space (same style as plus)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, (text_dim + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((text_dim + self.embedding_size) // 2, self.embedding_size)
        )

        # New: Fusion between long & short user embeddings (same idea as LightGCN_Fusion)
        self.fusion_type = self.hyper_config.get("fusion_type", "weighted_sum")

        if self.fusion_type == "weighted_sum":
            # Learnable alpha for weighted sum: alpha * short + (1-alpha) * long
            self.alpha = nn.Parameter(t.tensor([0.5], dtype=t.float32, device=device))
        elif self.fusion_type == "ffn":
            # FFN to project concatenated embeddings from 2d -> d
            self.fusion_ffn = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        if self.fusion_type == "ffn":
            for m in self.fusion_ffn:
                if isinstance(m, nn.Linear):
                    init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _gcn_forward(self, adj, user_embeds, keep_rate=1.0):
        #  One LightGCN pass on a given graph
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
        if self.fusion_type == "weighted_sum":
            alpha = t.clamp(self.alpha, 0.0, 1.0)
            return alpha * user_short + (1.0 - alpha) * user_long
        else:
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
        user_long, item_long = self._gcn_forward(self.adj_long, self.user_embeds_long, keep_rate)
        user_short, item_short = self._gcn_forward(self.adj_short, self.user_embeds_short, keep_rate)

        # Fuse users, average items
        user_embeds = self._fuse_users(user_long, user_short)
        item_embeds = (item_long + item_short) / 2.0

        self.final_embeds = t.concat([user_embeds, item_embeds], dim=0)
        return user_embeds, item_embeds

    # -------- training: BPR + duplicated KD + reg --------
    def cal_loss(self, batch_data):
        self.is_training = True

        # 1) GCN outputs from both graphs (with dropout)
        user_long, item_long = self._gcn_forward(self.adj_long, self.user_embeds_long, self.keep_rate)
        user_short, item_short = self._gcn_forward(self.adj_short, self.user_embeds_short, self.keep_rate)

        # 2) Fuse for BPR & item KD
        user_embeds = self._fuse_users(user_long, user_short)
        item_embeds = (item_long + item_short) / 2.0

        # 3) Pick batch anchor/pos/neg from fused embeddings
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        # 4) BPR loss (same as plus)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        # 5) Regularization
        reg_loss = self.reg_weight * reg_params(self)

        # 6) Project text embeddings (same MLP as plus)
        usr_long_txt = self.mlp(self.usrprf_long_embeds)    # [U, d]
        usr_short_txt = self.mlp(self.usrprf_short_embeds)  # [U, d]
        itm_txt = self.mlp(self.itmprf_embeds)              # [I, d]

        ancs, poss, negs = batch_data

        # 6) User KD: long and short branches
        anc_long_id = user_long[ancs]
        anc_short_id = user_short[ancs]
        anc_long_txt = usr_long_txt[ancs]
        anc_short_txt = usr_short_txt[ancs]

        kd_user_long = cal_infonce_loss(anc_long_id, anc_long_txt, usr_long_txt, self.kd_temperature)
        kd_user_short = cal_infonce_loss(anc_short_id, anc_short_txt, usr_short_txt, self.kd_temperature)

        # 7) Item KD: like original plus, but using fused item_embeds & itm_txt
        posprf_embeds = itm_txt[poss]
        negprf_embeds = itm_txt[negs]
        kd_item_pos = cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature)
        kd_item_neg = cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)

        kd_loss = (kd_user_long + kd_user_short + kd_item_pos + kd_item_neg) / anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss}

        if self.fusion_type == "weighted_sum":
            losses["alpha"] = float(t.clamp(self.alpha, 0.0, 1.0).item())

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
