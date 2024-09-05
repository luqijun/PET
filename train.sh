CONFIG=$1
cfg_path=$CONFIG

CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10001 \
    --use_env main.py \
    --cfg=$cfg_path
