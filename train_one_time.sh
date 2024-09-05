#!/bin/bash

# 设置默认值
cfg_path=${1:-""}

# 输出参数值
echo "cfg_path : $cfg_path"

CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10003 \
    --use_env main.py \
    --cfg=$cfg_path \
    --cfg-options \
    output_dir="pet_model_ntimes"
    # --resume='outputs/SHA/pet_model/checkpoint.pth'
