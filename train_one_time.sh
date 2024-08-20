#!/bin/bash

# 设置默认值
epochs=${1:-"500"}
eval_start=${2:-"100"}
dataset_file = ${3:-"SHA"}
output_dir = ${4:-"pet_model_ntimes"}

# 输出参数值
echo "total epochs : $epochs"
echo "eval start epoch : $eval_start"
echo "dataset_file : $dataset_file"
echo "output_dir : $output_dir"

CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10003 \
    --use_env main.py \
    --batch_size=8 \
    --lr=0.0001 \
    --backbone="vgg16_bn" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=$epochs \
    --dataset_file=$dataset_file \
    --eval_start=$eval_start \
    --eval_freq=1 \
    --output_dir=$output_dir
    # --resume='outputs/SHA/pet_model/checkpoint.pth'
