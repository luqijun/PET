nproc_per_node = 1
master_port = 10002

num_workers = 2
syn_bn = 0
world_size = 1
seed = 42
deterministic = False
device = "cuda"
sub_save_dir = "split_dataset"

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 0.1

# train
lr = 0.0001
epochs = 10000
batch_size = 8
eval_start = 500
eval_freq = 1
# eval_by_iters=50
train_print_freq = 20
val_print_freq = 50
save_freq = 2
warmup_ep = 3

eval_func_name = 'evaluate_split'
image_points_num_filename = './data/SHB/processed/min_edge_1024_split_512/test_data/image_points_num.npy'

# model
model = "pet_dialated_full_split_dec"
backbone = "vgg16_bn"
position_embedding = "sine"  # 'sine', 'learned', 'fourier'
# resume = f"outputs/SHB_General_Split/pet_dialated_full_split_dec/{sub_save_dir}/pet_model_ntimes/checkpoint.pth"

hidden_dim = 256
dim_feedforward = 512
nheads = 8
dropout = 0.0
use_seg_head = False
use_seg_head_attention = False
use_seg_level_attention = False

# encoder 结构
enc_blocks = 3  # 为1时应用于所有的window
enc_layers = 2
enc_win_size_list = [(8, 8), (8, 4), (8, 4)]  # encoder window size
enc_win_dialation_list = [4, 2, 1]  # 长度必须和enc_win_list一致

# # # decoder结构 最好结果 48.4615 76.7999
# dec_blocks = 2  # 为1时应用于所有的window
# dec_layers = 2
# dec_win_size_list_8x = [(8, 4), (8, 4)]
# dec_win_dialation_list_8x = [2, 1]
# dec_win_size_list_4x = [(4, 2), (4, 2)]
# dec_win_dialation_list_4x = [2, 1]

# decoder结构
dec_blocks = 1  # 为1时应用于所有的window
dec_layers = 2
dec_win_size_list_8x = [(8, 4)]
dec_win_dialation_list_8x = [1]
dec_win_size_list_4x = [(8, 4)]
dec_win_dialation_list_4x = [1]

# criterion
matcher = "matcher_with_points_weight"
set_cost_class = 1
set_cost_point = 0.05
eos_coef = 0.5
weight_dict = {
    'loss_ce': 1.0,
    'loss_points': 0.5  # 5.0  # 0.5(48.65_80.68) # 5.0
}
losses = ['labels', 'points']

seg_head_loss_weight = 0.1
seg_level_loss_weight = 0.1

# dataset
dataset_file = "SHB_General_Split"
data_path = "./data/SHB/processed/min_edge_1024_split_512"
output_dir = "pet_model"
head_sizes_folder = "images_head_size_by_depth_var"
seg_level_folder = "images_depth"
seg_head_folder = "images_head_split_by_depth_var"
seg_level_split_th = 0.2
head_size_weight = 2.0  # 1.5  # 0.8
patch_size = 256
