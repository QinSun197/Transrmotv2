# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.structures import Boxes, matched_boxlist_iou, pairwise_iou
from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn
from einops import rearrange, repeat
from models.vision_language_fusion import BiAttentionBlock
# from transformers import RobertaModel, RobertaTokenizerFast


# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
class DeformableTransformer(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=8,
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1,
                 activation="relu", 
                 return_intermediate_dec=False,
                 num_feature_levels=4, 
                 dec_n_points=4,  
                 enc_n_points=4,
                 two_stage=False, 
                 two_stage_num_proposals=300, 
                 decoder_self_cross=True, 
                 sigmoid_attn=False,
                 extra_track_attn=False,
                 # for fusion module
                 v_dim=256, 
                 l_dim=256, 
                 embed_dim=256,
                 sentence_fusion_mode='add'):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        
        feature_fusion_layer = BiAttentionBlock(v_dim=v_dim,l_dim=l_dim,
                                                embed_dim=embed_dim,
                                                num_heads=nhead,
                                                dropout=dropout,)
        
        # self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 
                                                    num_encoder_layers, 
                                                    feature_fusion_layer)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device), indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, text_dict=None, 
                ref_pts=None, sentence_fusion_mode=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory, memory_text, layer_image_outputs = self.encoder(
                              # for images
                              src=src_flatten, 
                              pos=lvl_pos_embed_flatten, 
                              spatial_shapes=spatial_shapes, 
                              level_start_index=level_start_index, 
                              valid_ratios=valid_ratios, 
                              key_padding_mask=mask_flatten, 
                              # for text
                              memory_text=text_dict['text_sentences_features'],
                              text_attention_mask=None,
                              text_self_attention_masks=None,
                              position_ids=None,
                              pos_text=None,
                              )
        bs, size, c = memory.shape
        sentence_embeds = text_dict['text_sentences_features'].unsqueeze(1)
        # sentence_embeds = 
        # sentence_embeds = memory_text[:,0,:].unsqueeze(1)
        # print('memory_sentence_embeds', sentence_embeds.shape)

        # sentence_embeds = text_dict['text_sentences_features'].unsqueeze(1).expand(-1, size, -1)
        
        # prepare input for decoder
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            if ref_pts is None: # what is rference_points
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points
        # language encoder
        # sentences = [sentence.replace('\'', '') for sentence in sentences]
        # memory_text, mask_text, memory_sentence = self.forward_text(sentences, src.device) # words * expressions * dimension

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index,
                                            valid_ratios, query_embed, mask_flatten, lvl_pos_embed_flatten, sentence_embeds)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, text_dict
        return hs, init_reference_out, inter_references_out, None, None, text_dict


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, 
                 sigmoid_attn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), 
                              reference_points, 
                              src, 
                              spatial_shapes, 
                              level_start_index, 
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        feature_fusion_layer=None,
        enc_layer_share=False,
        d_model=256,
        num_queries=300,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clonesv2(encoder_layer, num_layers, layer_share=enc_layer_share)

            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clonesv2(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        # self.d_model = d_model
        # add

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src, pos, spatial_shapes, level_start_index, 
        valid_ratios, key_padding_mask,
        # for texts
        memory_text, text_attention_mask, pos_text,
        text_self_attention_masks, position_ids,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        layer_image_outputs = []
        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # text sentence feature fusion
            # if (layer_id == 1 or layer_id > 2) and self.fusion_layers:
            # if layer_id > 2 and self.fusion_layers:
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )
            # encode feature    
            output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    padding_mask=key_padding_mask,
                )
            
            layer_image_outputs.append(output)

        return output, memory_text, layer_image_outputs


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, 
                 self_cross=True, sigmoid_attn=False, 
                 extra_track_attn=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # text sentence feature
        self.forword_text_mlp = TextMLP(dropout=dropout)
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

        # v_l fusion
        # self.vl_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.vl_dropout = nn.Dropout(dropout)
        # self.vl_norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # def forward_vl_fusion(self, v_features, v_pos, l_features):
    #     q = self.with_pos_embed(v_features, v_pos)
    #     tgt = self.vl_attention(q.transpose(0, 1), l_features.transpose(0, 1), l_features.transpose(0, 1))[0].transpose(0, 1)
    #     tgt = tgt + self.vl_dropout(v_features)
    #     return self.vl_norm(tgt)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    # def forword_text_mlp(self, sentence_embeds):
    #     """process the sentence_embeds from text encoder"""
    #     sentence_embeds = self.linear2(self.dropout3(self.activation(self.linear1(sentence_embeds))))
    #     sentence_embeds = self.dropout5
    #     return sentence_embeds

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300],self.norm4(tgt[:, 300:]+self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None, sentence_embeds=None):
        # self attention
        # tgt = tgt + sentence_embeds
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)  # query self-attn
        # # fusion sentence_embeds         
        # sentence_embeds = self.forword_text_mlp(sentence_embeds)
        # tgt_, _ = self.cross_attn_text(tgt.permute(1, 0, 2), 
        #                     key=sentence_embeds.permute(1, 0, 2), 
        #                     value=sentence_embeds.permute(1, 0, 2)
        #                     )
        # tgt = tgt_.permute(1, 0, 2) + tgt
        # v_l cross attention
        # src = self.forward_vl_fusion(src, lvl_pos_embed_flatten, memory_text)
        # level_extra_index = torch.as_tensor(src.shape[1], dtype=torch.long, device=level_start_index.device)
        # level_start_index = torch.cat([level_start_index, level_extra_index.view(1, )])
        # src = torch.cat([src, memory_text], dim=1)
        # src_padding_mask = torch.cat([src_padding_mask, mask_text], dim=1)
        # src = src + memory_text
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), # object query + track query
                               reference_points,
                               src, src_spatial_shapes, # image feature from encoder
                               level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, lvl_pos_embed_flatten=None, sentence_embeds=None, attn_mask=None):
        # attn_mask = None
        if self.self_cross:
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask, lvl_pos_embed_flatten, sentence_embeds)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask, lvl_pos_embed_flatten)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        # add query iso mask
        self.num_object_query = 300

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, lvl_pos_embed_flatten=None, sentence_embeds=None):
        # b, n, c = tgt.shape
        # output = repeat(sentence_embeds, 'b c -> b n c', n=n) + tgt # rmot_4a
        # output = repeat(sentence_embeds, 'b c -> b n c', n=n) # rmot_4b
        output = tgt # rmot_3

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # add query iso mask
            if lid < 1:
                query_mask = torch.zeros(len(output[0]), len(output[0]), dtype=torch.bool).to(device=output.device)
                query_mask[:self.num_object_query, self.num_object_query:] = True
                query_mask[self.num_object_query:, :self.num_object_query] = True
            else:
                query_mask = None

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, 
                           src_level_start_index, src_padding_mask, lvl_pos_embed_flatten,
                           sentence_embeds, query_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_clonesv2(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res

class PositionEmbeddingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor, hidden_dim: int):
        """
        @param mask: a tensor of shape [B, H, W]
        @param hidden_dim: int
        @return:
        """
        num_pos_feats = hidden_dim // 2

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2)) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        return tgt

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def  __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # self.fc1 = nn.Linear(output_feat_size, output_feat_size, bias=True)
        # self.layer_norm1 = nn.LayerNorm(output_feat_size, eps=1e-12)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        # output = self.dropout(x)
        # x = x.mean(1).unsqueeze(1)
        # x = self.fc1(x)
        # if self.do_ln:
        #     x = self.layer_norm1(x)
        return x
import torch.nn as nn

class TextMLP(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, dropout=0.1, activation=nn.ReLU()):
        super(TextMLP, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)  # 添加 LayerNorm
        self.residual = True  # 是否使用残差连接

    def forward(self, sentence_embeds):
        residual = sentence_embeds if self.residual else None
        x = self.linear1(sentence_embeds)
        x = self.dropout(x)  # 先 dropout 再激活
        x = self.activation(x)
        x = self.linear2(x)
        x = self.layer_norm(x)  # 归一化
        if residual is not None:
            x = x + residual  # 残差连接
        return x


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        # for fusion module
        v_dim=args.vision_dim, 
        l_dim=args.language_dim, 
        embed_dim=args.hidden_dim,
        sentence_fusion_mode=args.sentence_fusion_mode,
    )


