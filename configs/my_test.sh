# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1

taskset -c 16-19 python3 inferencev2.py \
--meta_arch rmotv2 \
--dataset_file e2e_rmot \
--epoch 200 \
--with_box_refine \
--lr_drop 100 \
--lr 2e-4 \
--lr_backbone 2e-5 \
--batch_size 1 \
--sample_mode random_interval \
--sample_interval 1 \
--sampler_steps 50 90 150 \
--sampler_lengths 2 3 4 5 \
--update_query_pos \
--merger_dropout 0 \
--dropout 0 \
--random_drop 0.1 \
--fp_ratio 0.3 \
--query_interaction_layer QIM \
--extra_track_attn \
--resume outputs/change_fusion_method_0226/train/checkpoint0089.pth \
--output_dir outputs/change_fusion_method_0226/ \
# --visualization
