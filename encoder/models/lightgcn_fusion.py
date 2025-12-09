import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN_Fusion(BaseModel):
    """
    LightGCN with Short-term + Long-term Fusion
    Fuses embeddings from two separate graphs:
    - Long-term: All training interactions
    - Short-term: Last 10/20 interactions per user
    """
    def __init__(self, data_handler):
        super(LightGCN_Fusion, self).__init__(data_handler)

        # Two separate adjacency matrices
        self.adj_long = data_handler.torch_adj_long  # All interactions
        self.adj_short = data_handler.torch_adj_short  # Recent interactions

        self.keep_rate = configs['model']['keep_rate']

        # Shared item embeddings (items are the same in both graphs)
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # Separate user embeddings for long-term and short-term
        self.user_embeds_long = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.user_embeds_short = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # Hyper-parameters
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']

        # Fusion parameters
        self.fusion_type = self.hyper_config.get('fusion_type', 'weighted_sum')  # 'weighted_sum', 'ffn', 'average'

        if self.fusion_type == 'weighted_sum':
            # Learnable alpha for weighted sum: alpha * short + (1-alpha) * long
            self.alpha = nn.Parameter(t.tensor([0.5]))
        elif self.fusion_type == 'ffn':
            # FFN to project concatenated embeddings from 2d -> d
            self.fusion_ffn = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.fusion_type == 'average':
            pass
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        self._init_weight()

    def _init_weight(self):
        if self.fusion_type == 'ffn':
            for m in self.fusion_ffn:
                if isinstance(m, nn.Linear):
                    init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _gcn_forward(self, adj, user_embeds, keep_rate=1.0):
        """Single GCN forward pass for one graph"""
        embeds = t.concat([user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def forward(self, keep_rate=1.0):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        # Forward pass through long-term graph
        user_embeds_long, item_embeds_long = self._gcn_forward(
            self.adj_long, self.user_embeds_long, keep_rate
        )

        # Forward pass through short-term graph
        user_embeds_short, item_embeds_short = self._gcn_forward(
            self.adj_short, self.user_embeds_short, keep_rate
        )

        # Fuse user embeddings
        if self.fusion_type == 'weighted_sum':
            # Weighted sum: alpha * short + (1-alpha) * long
            # Clamp alpha to [0, 1]
            alpha = t.clamp(self.alpha, 0, 1)
            user_embeds = alpha * user_embeds_short + (1 - alpha) * user_embeds_long
        elif self.fusion_type == 'ffn':
            # FFN: project from 2d -> d
            user_embeds_concat = t.cat([user_embeds_long, user_embeds_short], dim=-1)
            user_embeds = self.fusion_ffn(user_embeds_concat)
        elif self.fusion_type == 'average':
            user_embeds = (user_embeds_short + user_embeds_long) / 2

        # Average item embeddings from both graphs
        item_embeds = (item_embeds_long + item_embeds_short) / 2
        # ? keep item embeddings the same --> our project will focus only on changing UEs
        # item_embeds = item_embeds_long 

        # Cache final embeddings
        self.final_embeds = t.concat([user_embeds, item_embeds], axis=0)

        return user_embeds, item_embeds

    def cal_loss(self, batch_data):
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

        # Log fusion parameter if using weighted sum
        if self.fusion_type == 'weighted_sum':
            losses['alpha'] = t.clamp(self.alpha, 0, 1).item()

        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
