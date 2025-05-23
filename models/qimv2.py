# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List
import torchvision

from util import box_ops
from util.misc import inverse_sigmoid
from models.structures import Boxes, Instances, pairwise_iou


def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_d, out_d) for in_d, out_d in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = F.relu(layer(x), inplace=False) if i < self.num_layers - 1 else layer(x)

        return x

class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos
        # for memory aggregation
        self.pool_type = args.pool_type
        self.num_obj_queries = args.num_queries
        self.update_threshold = args.update_threshold
        self.hist_length = args.history_memory_length

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout
        self.hidden_dim = hidden_dim // 4

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

        # for memory aggregation
        self.confidence_weight_net = nn.Sequential(
            MLP(hidden_dim // 4, hidden_dim // 4, hidden_dim // 4, 2),
            nn.Sigmoid()
        )
        self.short_memory_aggregation = MLP(hidden_dim // 2, hidden_dim // 2, dim_in, 2)
        aggregation_layer =  nn.TransformerEncoderLayer(
            d_model=dim_in,
            nhead=8,
            dim_feedforward=hidden_dim // 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.long_memory_aggregation = nn.TransformerEncoder(aggregation_layer, 
                                                             num_layers=args.num_agg_layers,
                                                             norm=nn.LayerNorm(dim_in))

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances
    def _update_hist_memory(self, tracks):
        """ 更新记忆 """
        device =  tracks.query_pos.device

        """ 更新历史记忆信息 """
        # embeds
        tracks.hist_memory = tracks.hist_memory.clone()
        tracks.hist_memory = torch.cat((
            tracks.hist_memory[:,1:,:],tracks.output_embedding[:,None,:]),dim=1)
        # padding masks
        tracks.hist_memory_padding_mask = torch.cat((
            tracks.hist_memory_padding_mask[:, 1:], 
            torch.zeros((len(tracks), 1), dtype=torch.bool, device=device)), 
            dim=1)  
        
        return tracks 
    
    def _hist_memory_aggregation(self, tracks):
        x = tracks.hist_memory                # shape: (num_tracks, seq_len, dim)
        memory_mask = tracks.hist_memory_padding_mask  # shape: (num_tracks, seq_len)

        # TransformerEncoder 接收的是 (B, L, D) 的memory buffer; memory_mask: (B, L) -> bool, True 表示要被mask
        agg_out = self.long_memory_aggregation(x, src_key_padding_mask=memory_mask)

        # pooling
        if self.pool_type == 'avg':
            agg_memory = torch.mean(agg_out, dim=1)
        elif self.pool_type == 'max':
            agg_memory, _ = torch.max(agg_out, dim=1)
        elif self.pool_type == 'sum':
            agg_memory = torch.sum(agg_out, dim=1)

        return agg_memory    

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        track_instances[:self.num_obj_queries].last_output = track_instances[:self.num_obj_queries].output_embedding
        track_instances[:self.num_obj_queries].long_memory = track_instances[:self.num_obj_queries].query_pos[:, self.hidden_dim:]
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        # update history memory
        track_instances = self._update_hist_memory(track_instances)

        # prepare 
        dim = track_instances.query_pos.shape[1] // 2

        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim]
        query_feat = track_instances.query_pos[:, dim:]
        scores = track_instances.scores
        is_pos = scores > self.update_threshold
        # memory
        last_output_embedding = track_instances.last_output
        confidence_weight =self.confidence_weight_net(out_embed)
        short_memory = self.short_memory_aggregation(torch.cat((confidence_weight * out_embed, 
                                                           last_output_embedding), dim=-1))
        long_memory = self._hist_memory_aggregation(track_instances)

        # attention
        q = short_memory + query_pos
        k = long_memory + query_pos # pos_embed + current_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # update query pos embed
        if self.update_query_pos: 
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim][is_pos] = query_pos[is_pos]
        # update query feat embed
        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim:] = query_feat
        # update last_out embedding
        # new_is_pos = is_pos.unsqueeze(-1)
        # track_instances.last_output = torch.where(new_is_pos, out_embed, track_instances.last_output)
        track_instances.last_output[is_pos] = out_embed[is_pos]


        # update ref_pts
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes.detach().clone())
        # track_instances.ref_pts[is_pos] = inverse_sigmoid(track_instances[is_pos].pred_boxes.detach().clone())
        track_instances.ref_pts[is_pos] = inverse_sigmoid(track_instances[is_pos].pred_boxes[:, :2].detach().clone())

        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances


def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIMv2': QueryInteractionModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
