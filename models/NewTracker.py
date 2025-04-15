import os
import torch
from typing import Dict
from .structures.instances import Instances


class NewTracker:
    def __init__(self, det_score_thresh=0.7, track_score_thresh=0.6, miss_tolerance=5):

        # 目标检测和跟踪的置信度阈值
        self.det_score_thresh = det_score_thresh
        self.track_score_thresh = track_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0  # 目标 ID 计数
    
    def clear(self):
        self.max_obj_id = 0

    def update(self, model_outputs):
        """更新已跟踪目标，并添加新目标"""
        # model_outputs.scores"] = logits_to_scores(model_outputs.pred_logits"])
        n_dets = 300 # num of object queries

        # 获取已存在目标的跟踪结果
        tracks = model_outputs[n_dets:]

        for i in range(len(tracks)):
            # 如果跟踪目标置信度低于阈值，则累加丢失计数
            # if tracks.scores[i][tracks.labels[i]] < self.track_score_thresh:

            if tracks.scores[i] < self.track_score_thresh:
                tracks.disappear_time[i] += 1
            else:
                tracks.disappear_time[i] = 0

            # 如果丢失目标时间超过阈值，则将 ID 置为 -1，表示目标消失
            if tracks.disappear_time[i] >= self.miss_tolerance:
                tracks.obj_idxes[i] = -1

        # 处理新出现的目标
        new_tracks = Instances((1,1))
        new_tracks_idxes = torch.max(model_outputs.scores.unsqueeze(1)[:n_dets], dim=-1).values >= self.det_score_thresh
        new_tracks.pred_logits = model_outputs.pred_logits[:n_dets][new_tracks_idxes]
        new_tracks.pred_boxes = model_outputs.pred_boxes[:n_dets][new_tracks_idxes]
        new_tracks.pred_refers = model_outputs.pred_refers[:n_dets][new_tracks_idxes]
        new_tracks.ref_pts = model_outputs.ref_pts[:n_dets][new_tracks_idxes]
        new_tracks.scores = model_outputs.scores[:n_dets][new_tracks_idxes]
        new_tracks.output_embedding = model_outputs.output_embedding[:n_dets][new_tracks_idxes]
        new_tracks.det_query_embed = model_outputs.det_query_embed[:n_dets][new_tracks_idxes]
        new_tracks.init_queries = model_outputs.init_queries[:n_dets][new_tracks_idxes]
        
        new_tracks.query_pos = torch.cat(
                (model_outputs.det_query_embed[:n_dets][new_tracks_idxes][:, :256],  # hack
                 model_outputs.init_queries[:n_dets][new_tracks_idxes]),
                dim=-1
            )


        # 初始化丢失计数和标签
        new_tracks.disappear_time = torch.zeros((len(new_tracks.pred_logits),), dtype=torch.long)
        new_tracks.labels = torch.max(new_tracks.scores.unsqueeze(1), dim=-1).indices

        # 分配 ID
        ids = []
        for i in range(len(new_tracks)):
            ids.append(self.max_obj_id)
            self.max_obj_id += 1
        new_tracks.obj_idxes = torch.as_tensor(ids, dtype=torch.long)
        new_tracks = new_tracks.to(new_tracks.pred_logits.device)

        return tracks, new_tracks
