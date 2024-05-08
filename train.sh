CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10001 \
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
    --epochs=10000 \
    --dataset_file="SHA" \
    --eval_start=250 \
    --eval_freq=1 \
    --output_dir='pet_model' #\
    #--resume="outputs/SHA/pet_model/checkpoint.pth"
