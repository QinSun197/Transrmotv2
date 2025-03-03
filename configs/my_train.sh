# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1,2,3

PRETRAIN=weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=outputs/change_fusion_method_0226/train
taskset -c 20-30 python3 -m torch.distributed.launch \
   --nproc_per_node=3 \
   --use_env main_debug.py \
   --meta_arch rmotv2 \
   --use_checkpoint \
   --dataset_file e2e_rmot \
   --epoch 100 \
   --with_box_refine \
   --lr_drop 50 \
   --lr 1e-4 \
   --lr_backbone 1e-5 \
   --output_dir ${EXP_DIR} \
   --batch_size 1 \
   --sample_mode random_interval \
   --sample_interval 1 \
   --sampler_steps 60 80 90 \
   --sampler_lengths 2 2 2 2 \
   --update_query_pos \
   --merger_dropout 0 \
   --dropout 0 \
   --random_drop 0.1 \
   --fp_ratio 0.3 \
   --query_interaction_layer QIM \
   --data_txt_path_train ./datasets/data_path/refer-kitti.train \
   --refer_loss_coef 2 \
   --resume  /home/sq_2023/rmot/transrmot/outputs/change_fusion_method_0226/train/checkpoint0039.pth \

   # --pretrained ${PRETRAIN}\
