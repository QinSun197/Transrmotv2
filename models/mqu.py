# Copyright (c) Ruopeng Gao. All Rights Reserved.
# copy and chaged frome MeMOTR

import os
import math
import torch
import torch.nn as nn
from typing import List, Optional
from .structures.instances import Instances
from util.tool import FFN, MLP
from util.box_ops import box_cxcywh_to_xyxy, box_iou_union
from util.misc import inverse_sigmoid


class MQU(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim // 4
        self.ffn_dim = dim_out
        self.dropout = args.memory_dropout
        self.pos_dropout = args.dropout
        self.tp_drop_ratio = args.tp_drop_ratio
        self.fp_insert_ratio = args.fp_insert_ratio
        self.use_checkpoint = args.use_checkpoint
        self.visualize = args.visualization
        self.update_threshold = args.update_threshold
        self.long_memory_lambda = args.long_memory_lambda

        # Initialize layers
        self.confidence_weight_net = nn.Sequential(
            MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(2 * self.hidden_dim, 2 * self.hidden_dim, self.hidden_dim, 2)
        self.memory_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(self.hidden_dim, self.ffn_dim, self.dropout)

        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(self.hidden_dim, self.ffn_dim, self.dropout)
        self.query_pos_head = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)

        self.linear_pos1 = nn.Linear(self.dim_in, hidden_dim)
        self.linear_pos2 = nn.Linear(hidden_dim, self.dim_in)
        self.norm_pos = nn.LayerNorm(self.dim_in)
        self.activation = nn.ReLU(inplace=True)

        # new query pos update method  to do !!!
        self.dropout_pos1 = nn.Dropout(self.pos_dropout)
        self.dropout_pos2 = nn.Dropout(self.pos_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, previous_tracks, new_tracks, unmatched_dets=None, no_augment=False):
        active_tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment)
        tracks = self.update_tracks_embedding(active_tracks)
        return tracks

    def update_tracks_embedding(self, tracks):
        scores = tracks.scores
        is_pos = scores > self.update_threshold # get mask 

        # Convert bbox to ref points and generate query_pos
        tracks.ref_pts[is_pos] = inverse_sigmoid(tracks[is_pos].pred_boxes.detach().clone())
        # query_pos = self.query_pos_head(pos_to_pos_embed(tracks.ref_pts.sigmoid(), self.hidden_dim // 2))
        query_pos = tracks.query_pos[:, :self.hidden_dim] # to do !!!

        # Get memory and embeddings
        output_embedding = tracks.output_embedding
        last_output_embedding = tracks.last_output
        long_memory = tracks.long_memory.detach()

        # Confidence Weight
        confidence_weight = self.confidence_weight_net(output_embedding)

        # Fuse short memory
        short_memory = self.short_memory_fusion(torch.cat((confidence_weight * output_embedding, 
                                                           last_output_embedding), dim=-1))

        # Apply attention over memory
        q = short_memory + query_pos
        k = long_memory + query_pos
        tgt = output_embedding
        # attention 
        tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
        tgt = self.memory_norm(tgt + self.memory_dropout(tgt2))
        tgt = self.memory_ffn(tgt)

        # Update query features
        query_feat = self.query_feat_norm(long_memory + self.query_feat_dropout(tgt))
        query_feat = self.query_feat_ffn(query_feat)

        # Long-term memory update
        new_long_memory = (1 - self.long_memory_lambda) * long_memory + \
                          self.long_memory_lambda * output_embedding
        new_is_pos = is_pos.unsqueeze(-1)  # shape: (N, 1)
        tracks.long_memory = torch.where(new_is_pos, new_long_memory, tracks.long_memory)
        tracks.last_output = torch.where(new_is_pos, output_embedding, tracks.last_output)

        tracks.query_pos[:, self.hidden_dim:][is_pos] = query_feat[is_pos]

        # Update query pos, new method to do !!!
        query_pos_new = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
        query_pos = query_pos + self.dropout_pos2(query_pos_new)
        query_pos = self.norm_pos(query_pos)
        tracks.query_pos[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

        return tracks

    def select_active_tracks(self, previous_tracks, new_tracks, unmatched_dets, no_augment=False):
        if not self.training:
            return self._select_eval_tracks(previous_tracks, new_tracks)

        new_tracks.last_output = new_tracks.output_embedding
        new_tracks.long_memory = new_tracks.query_pos[:, self.hidden_dim:]
        unmatched_dets.last_output = unmatched_dets.output_embedding
        unmatched_dets.long_memory = unmatched_dets.query_pos[:, self.hidden_dim:]

        if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
            active_tracks = self._basic_merge_without_augment(previous_tracks, new_tracks, unmatched_dets)
        else:
            active_tracks = self._augment_training_data(previous_tracks, new_tracks, unmatched_dets, no_augment)

        if len(active_tracks) == 0:
            return self._create_fake_tracks()
        
        return active_tracks

    def _select_eval_tracks(self, previous_tracks, new_tracks):
        new_tracks.last_output = new_tracks.output_embedding
        new_tracks.long_memory = new_tracks.query_pos[:, self.hidden_dim:]
        active_tracks = Instances.cat([previous_tracks, new_tracks])
        return active_tracks[active_tracks.obj_idxes >= 0]
    
    def _basic_merge_without_augment(self, previous, new, unmatched):
        active = Instances.cat([unmatched, new])
        active = Instances.cat([active, previous])
        scores = active.scores
        keep = (scores > self.update_threshold) | (active.obj_idxes >= 0)
        active = active[keep]
        active.obj_idxes[active.iou < 0.5] = -1
        return active

    def _augment_training_data(self, prev, new, unmatched, no_augment):
        active = Instances.cat([prev, new])
        active = active[(active.iou > 0.5) & (active.obj_idxes >= 0)]

        if self.tp_drop_ratio > 0 and not no_augment:
            tp_keep_mask = torch.rand(len(active)) > self.tp_drop_ratio
            active = active[tp_keep_mask]

        if self.fp_insert_ratio > 0 and not no_augment:
            fp = self._insert_false_positives(active, unmatched)
            active = Instances.cat([active, fp])
        return active

    def _insert_false_positives(self, selected_active_tracks, unmatched_dets):
        if len(unmatched_dets) == 0 or len(selected_active_tracks) == 0:
            return unmatched_dets
        selected_mask = torch.rand(len(selected_active_tracks)) < self.fp_insert_ratio
        selected = selected_active_tracks[selected_mask]
        if len(selected) == 0:
            return unmatched_dets[:1]
        iou, _ = box_iou_union(
            box_cxcywh_to_xyxy(unmatched_dets.pred_boxes),
            box_cxcywh_to_xyxy(selected.pred_boxes)
        )
        fp_idx = torch.max(iou, dim=0).indices.unique()
        return unmatched_dets[fp_idx]

    def _create_fake_tracks(self):
        device = next(self.parameters()).device
        fake_tracks = Instances((1, 1)).to(device)
        fake_tracks.ref_pts = torch.randn((1, 2), device=device)
        fake_tracks.query_pos = torch.randn((1, self.hidden_dim * 2), device=device)
        fake_tracks.output_embedding = torch.randn((1, self.hidden_dim), device=device)
        fake_tracks.obj_idxes = torch.tensor([-2], device=device)
        fake_tracks.scores = torch.zeros((1,), device=device)
        fake_tracks.pred_boxes = torch.randn((1, 4), device=device)
        fake_tracks.pred_logits = torch.randn((1, 1), dtype=torch.float, device=device)
        fake_tracks.pred_refers = torch.randn((1, 1), dtype=torch.float, device=device)
        fake_tracks.labels = torch.zeros((1,), dtype=torch.long, device=device)
        fake_tracks.last_output = torch.randn((1, self.hidden_dim), device=device)
        fake_tracks.track_scores = torch.zeros((1,), device=device)
        fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
        fake_tracks.disappear_time = torch.as_tensor([0], dtype=torch.long, device=device)
        fake_tracks.matched_gt_idxes = torch.as_tensor([-2], dtype=torch.long, device=device)
        fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
        fake_tracks.det_query_embed = torch.randn((1, 512), dtype=torch.float, device=device)   
        fake_tracks.init_queries = torch.randn((1, 256), dtype=torch.float, device=device)
        fake_tracks.mem_bank = torch.zeros((1, 0, 256), dtype=torch.float32, device=device)
        fake_tracks.mem_padding_mask = torch.ones((1, 0), dtype=torch.bool, device=device)
        fake_tracks.save_period = torch.zeros((1, ), dtype=torch.float32, device=device)

        return fake_tracks

def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'MQU': MQU,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
