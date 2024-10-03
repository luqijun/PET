nproc_per_node=1
master_port=10002

num_workers=4
syn_bn=0
world_size=1
seed=42
device="cuda"
sub_save_dir="ignore_pt_loss"

lr=1e-4
lr_backbone=1e-5
weight_decay=1e-4
clip_max_norm=0.1

# train
lr=0.0001
epochs=1500
batch_size=8
eval_start=400
eval_freq=1

# model
model="pet_dialated_full"
backbone="vgg16_bn"
position_embedding="sine" # 'sine', 'learned', 'fourier'
# resume="outputs/SHA_General/pet_dialated_full/ignore_pt_loss/pet_model_ntimes/checkpoint.pth"

hidden_dim=256
dim_feedforward=512
nheads=8
dropout=0.0
use_seg_head=False

# 49.77 80.94
enc_win_size_list = [(8, 4), (8, 4), (8, 4), (8, 4), (8, 4), (8, 4)]  # encoder window size
enc_win_dialation_list = [4, 4, 2, 2, 1, 1] # 长度必须和enc_win_list一致

dec_layers=4
dec_win_size_list_8x = [(8, 4), (8, 4), (8, 4),(8, 4)]
dec_win_dialation_list_8x = [2, 2, 1, 1]

dec_win_size_list_4x = [(4, 2), (4, 2), (4, 2), (4, 2)]
dec_win_dialation_list_4x = [2, 2, 1, 1]

# 53.13 82.14
# enc_win_list = [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2)]  # encoder window size
# enc_win_stride_list = [8, 8, 4, 4, 2, 2, 1, 1] # 长度必须和enc_win_list一致

# 57.31
# enc_win_list = [(8, 8), (8, 8), (8, 8), (8, 8), (4, 4), (4, 4)]  # encoder window size
# enc_win_stride_list = [4, 4, 2, 2, 1, 1] # 长度必须和enc_win_list一致

# criterion
criterion="criterion_ignore_pt_loss"
matcher="matcher_with_points_weight"
set_cost_class=1
set_cost_point=0.05
ce_loss_coef=1.0
point_loss_coef=5.0 #0.5(48.65_80.68) # 5.0
eos_coef=0.5
head_size_range_ratio=0.3

# dataset
dataset_file="SHA_General"
data_path="./data/ShanghaiTech/part_A/"
output_dir="pet_model"
head_sizes_folder = "images_head_size_by_depth_var"
seg_level_folder="images_depth"
seg_head_folder="images_head_split_by_depth_var"
seg_level_split_th = 0.2
head_size_weight = 1.0 # 0.8
