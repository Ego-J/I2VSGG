#!/bin/sh
cd ..
gpu_id=1
lr=1
vrd_lr=0.0001
lr_decay_step=1
lr_decay_gamma=0.9
max_epoch=10
train_task=pre_det
resume=true
load_path=adapt/instance_pixel_styleD_bilinear_cr_False_source_VGOR_target_VidOR_session_1_lr_0.0005_epoch_1_bs_4_mscoco_29.9.pth

adaptation=adaptation2

# rm -rf ./frame_feat/VidOR/${adaptation}/*
# rm -rf ./video_feat/VidOR/${adaptation}/*

source_det_ds=VGOR_SGG
target_det_ds=VidOR_SGG
file_name=/VidOR/SGG_vrdlr${vrd_lr}_epoch${max_epoch}_VG_VidOR_${adaptation}_mscoco
mkdir -p log_SGG_emb/$file_name

# # <---------train-------->
# echo training $file_name and load ckpt $load_path
# CUDA_VISIBLE_DEVICES=$gpu_id python trainval_net_SGG_emb.py --cuda --net res101 --dataset $source_det_ds \
#                     --use_tfb --tfb_path ${file_name} \
#                     --lr $lr --lr_decay_step $lr_decay_step --lr_decay_gamma $lr_decay_gamma \
#                     --vrd_task $train_task --vrd_lr ${vrd_lr} --o sgd  \
#                     --epochs $max_epoch \
#                     --r --load_name $load_path \
#                     --num_classes 42 --num_relations 26 \
#                     --adaptation $adaptation \
#                     >&./log_SGG_emb/${file_name}/train.log

#<---------test--------->  
task=rel_det
for epoch in {1..1..1}
do
    ckpt='SGG_emb_p_prior_valsgg_VGOR_SGG_pre_det_session_1_epoch_1_step_9146_un.pth'
    echo clipart testing $epoch load_name $ckpt
     CUDA_VISIBLE_DEVICES=$gpu_id python test_net_SGG_emb_VidOR.py --cuda --net res101 --dataset $target_det_ds\
     --vrd_task $task --load_name './models/res101/VGOR_SGG/'$ckpt \
     --num_classes 42 --num_relations 26 --save_feat_path './frame_feat/VidOR/'$adaptation --save_videofeat_path './video_feat/VidOR/'$adaptation --adaptation $adaptation \
     >&./log_SGG_emb/${file_name}/rebuttal.log
    #python dynamic_reasoning.py --gt_relations './gt_relations.json' --prediction 'frame_predicate_relations.json' >&./log_SGG_emb/${file_name}/test-epoch${epoch}-result.log
done

# #<---------test--------->  
# adaptation=gt

# file_name=gt_box_VidOR
# mkdir -p log_SGG_emb/$file_name

# task=rel_det
# for epoch in {1..1..1}
# do
#     ckpt='adapt/SGG_emb_p_prior_adapt_VGOR_SGG_pre_det_session_1_epoch_1_step_9146_un.pth'
#     echo clipart testing $epoch load_name $ckpt
#      CUDA_VISIBLE_DEVICES=$gpu_id python test_net_SGG_emb_VidOR_gt.py --cuda --net res101 --dataset VidOR_SGG\
#      --vrd_task $task --load_name './models/res101/VGOR_SGG/'$ckpt \
#      --num_classes 42 --num_relations 26 --save_feat_path './frame_feat/VidOR/'$adaptation --save_videofeat_path './video_feat/VidOR/'$adaptation --adaptation $adaptation\
#      #>&./log_SGG_emb/${file_name}/test-epoch${epoch}.log
#     #python dynamic_reasoning.py --gt_relations './gt_relations.json' --prediction 'frame_predicate_relations.json' >&./log_SGG_emb/${file_name}/test-epoch${epoch}-result.log
# done


# # #<---------test--------->
# file_name=gt_box_VidVRD
# mkdir -p log_SGG_emb/$file_name

# task=rel_det
# for epoch in {1..1..1}
# do
#     ckpt='SGG_emb_p_prior_source_MVRD_pre_det_session_1_epoch_1_step_1464_ins_styleD.pth'
#     echo clipart testing $epoch load_name $ckpt
#      CUDA_VISIBLE_DEVICES=$gpu_id python test_net_SGG_emb_VidVRD.py --cuda --net res101 --dataset MVidVRD\
#      --vrd_task $task --load_name './models/res101/VidVRD_SGG/'$ckpt \
#      --num_classes 16 --num_relations 89 --save_feat_path './frame_feat/VidVRD/'$adaptation --save_videofeat_path './video_feat/VidVRD/'$adaptation --adaptation $adaptation \
#      --target_gt_rels_path './data/VidVRD/target_gt_rels.pkl' --source_so_prior_path './data/VidVRD/source_so_prior.pkl' --predicate_file './data/VidVRD/predicates.json' \
#      #>&./log_SGG_emb/${file_name}/test-epoch${epoch}.log
#     #python dynamic_reasoning.py --gt_relations './gt_relations.json' --prediction 'frame_predicate_relations.json' >&./log_SGG_emb/${file_name}/test-epoch${epoch}-result.log
# done


# file_name=gt_box_VRD
# mkdir -p log_SGG_emb/$file_name

# task=rel_det
# for epoch in {1..1..1}
# do
#     ckpt='SGG_emb_p_prior_source_MVRD_pre_det_session_1_epoch_1_step_1464_ins_styleD.pth'
#     echo clipart testing $epoch load_name $ckpt
#      CUDA_VISIBLE_DEVICES=$gpu_id python test_net_SGG_emb_VRD.py --cuda --net res101 --dataset MVRD\
#      --vrd_task $task --load_name './models/res101/VidVRD_SGG/'$ckpt \
#      --num_classes 16 --num_relations 89 --save_feat_path './frame_feat/VRD/'$adaptation --save_videofeat_path './video_feat/VRD/'$adaptation --adaptation $adaptation \
#      --target_gt_rels_path './data/VidVRD/source_gt_rels.pkl' --source_so_prior_path './data/VidVRD/source_so_prior.pkl' --predicate_file './data/VidVRD/predicates.json' \
#      #>&./log_SGG_emb/${file_name}/test-epoch${epoch}.log
#     #python dynamic_reasoning.py --gt_relations './gt_relations.json' --prediction 'frame_predicate_relations.json' >&./log_SGG_emb/${file_name}/test-epoch${epoch}-result.log
# done