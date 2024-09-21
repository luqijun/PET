nproc_per_node=1
master_port=10002

num_workers=8
syn_bn=0
world_size=1
seed=42
device="cuda"

lr=1e-4
lr_backbone=1e-5
weight_decay=1e-4
clip_max_norm=0.1

# train
lr=0.0001
epochs=5000
batch_size=16
eval_start=100
eval_freq=1

# model
model="pet"
backbone="vgg16_bn"
position_embedding="sine" # 'sine', 'learned', 'fourier'
resume=None
dec_layers=2
hidden_dim=256
dim_feedforward=512
nheads=8
dropout=0.0
resume="outputs/JHU/pet_model_ntimes/checkpoint.pth"

# criterion
matcher="matcher"
set_cost_class=1
set_cost_point=0.05
ce_loss_coef=1.0
point_loss_coef=5.0
eos_coef=0.5

# dataset
dataset_file="JHU"
data_path="./data/JHU/"
output_dir="pet_model"
seg_level_folder="images_depth"
seg_head_folder="images_head_split_by_depth"
seg_level_split_th = 150
head_size_weight = 1.0
