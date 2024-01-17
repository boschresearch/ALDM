#!/bin/bash

# dir of script 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
# parent dir of that dir
PARENT_DIRECTORY="${DIR%/*}"
echo "----> Enter ${PARENT_DIRECTORY}"
cd $PARENT_DIRECTORY
export PYTHONPATH=$PARENT_DIRECTORY:$PYTHONPATH

##### >---------------> Project Parameters <-----------------------< ####
script=train_cldm_seg_pixel_multi_step.py
config=models/cldm_seg_ade20k_multi_step_D.yaml
resume_path=/fs/scratch/rng_cr_bcai_dl/lyu7rng/0_project_large_models/code_repo/0_ControlNet/checkpoint/control_seg_enc_scratch.ckpt
work_dir=./train_log

gpus=2 
lr=0.00002
d_lr=0.000001
seed=777 #7
batch_size=4 # per GPU
max_it=200 # in K
multi_step=9
start_dis_fake=10000 
end_dis_fake=15000 
train_dis_from=5000
lambda_D_loss=0.5
weight_fake_D=0.001 
dataset=ade

num_subset_timesteps=25
logger_freq_epoch=5 # frequency to save checkpoints: per epoch
logger_freq=100
##### >---------------> Project Parameters <-----------------------< ####

python $script --config $config --work_dir $work_dir --resume_path $resume_path --lr $lr --d_lr $d_lr --seed $seed --batch_size $batch_size --gpus $gpus  --max_it $max_it --lambda_D_loss $lambda_D_loss  --start_dis_fake $start_dis_fake --end_dis_fake $end_dis_fake --train_dis_from $train_dis_from --dataset $dataset --num_subset_timesteps $num_subset_timesteps --weight_fake_D $weight_fake_D --multi_step $multi_step  --logger_freq_epoch $logger_freq_epoch --logger_freq $logger_freq 
