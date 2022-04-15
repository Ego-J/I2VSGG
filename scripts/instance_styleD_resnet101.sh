#!/bin/sh
cd ..
gpu_id=0
lr=0.0005
lr_decay_step=10
lr_decay_gamma=0.1
max_epoch=10
eta=0.1
eta_style=0.001
style_lambda=1
bs=1
#fasterRCNN pretrained model on mscoco
load_path=./data/pretrained_model/faster_rcnn_1_10_9771.pth

# <---------file-------->
file_name=VG_VidOR_ins_pix_styD_lr${lr}_eta${eta}_eta_sty${eta_style}_sty${style_lambda}_bs_${bs}_mscoco
mkdir -p ./log/$file_name

# <---------train-------->
echo training $file_name and load ckpt $load_path 
CUDA_VISIBLE_DEVICES=$gpu_id python trainval_net_instance_styleD_bilinear.py --cuda --net res101 --dataset MVRD --dataset_t MVidVRD \
                    --use_tfb --tfb_path ${file_name} \
                    --lr $lr --lr_decay_step $lr_decay_step --lr_decay_gamma $lr_decay_gamma \
                    --eta $eta --eta_style $eta_style --style_lambda $style_lambda --bs $bs \
                    --epochs $max_epoch \
                    --r --load_name $load_path \
                    >&./log/$file_name/train.log


# <---------test--------->
for epoch in {1..10..1}
do  
     ckpt=./models/res101/MVRD/instance_pixel_styleD_bilinear_cr_False_source_MVRD_target_MVidVRD_session_1_lr_${lr}_epoch_${epoch}_bs_${bs}_mscoco.pth
     echo clipart testing $epoch load_name $ckpt
     CUDA_VISIBLE_DEVICES=$gpu_id python test_net_instance_styleD_bilinear.py --cuda --net res101 --dataset MVidVRD \
     --load_name $ckpt \
     #>&./log/${file_name}/test-epoch${epoch}.log
done
