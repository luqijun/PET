nproc_per_node = 1
master_port = 10002

num_workers = 4
syn_bn = 0
world_size = 1
seed = 42
deterministic = False
device = "cuda"

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 0.1

# train
lr = 0.0001
epochs = 5000
batch_size = 12
eval_start = 1500
eval_freq = 1
clear_cuda_cache = True

# model
model = "pet_dialated"
backbone = "vgg16_bn"
position_embedding = "sine"  # 'sine', 'learned', 'fourier'
resume = "outputs/JHU_General/pet_dialated/pet_model_ntimes/checkpoint.pth"
dec_layers = 2
hidden_dim = 256
dim_feedforward = 512
nheads = 8
dropout = 0.0
learn_to_scale = False
use_seg_head = False
use_pred_head_sizes = False
num_pts_per_feature = 1

# 48.91 80.83
enc_blocks = 3  # 为1时应用于所有的window
enc_layers = 2
enc_win_size_list = [(8, 4), (8, 4), (8, 4)]  # encoder window size
enc_win_dialation_list = [4, 2, 1]  # 长度必须和enc_win_list一致

# 53.13 82.14
# enc_win_size_list = [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2)]  # encoder window size
# enc_win_dialation_list = [8, 8, 4, 4, 2, 2, 1, 1] # 长度必须和enc_win_list一致

# 57.31
# enc_win_size_list = [(8, 8), (8, 8), (8, 8), (8, 8), (4, 4), (4, 4)]  # encoder window size
# enc_win_dialation_list = [4, 4, 2, 2, 1, 1] # 长度必须和enc_win_list一致

# decoder结构
dec_win_size_list_8x = [(8, 4)]
dec_win_dialation_list_8x = None
dec_win_size_list_4x = [(4, 2)]
dec_win_dialation_list_4x = None

# criterion
# criterion = "criterion_ignore_pt_loss"
matcher = "matcher_with_points_weight"
# matcher = "matcher_head_size"
set_cost_class = 1
set_cost_point = 0.05
eos_coef = 0.5
weight_dict = {
    'loss_ce': 1.0,
    'loss_points': 0.5  # 5.0
}
losses = ['labels', 'points']
# weight_dict = {'loss_ce': 1.0, 'loss_points': 5.0, 'loss_sizes': 5.0}
# losses = ['labels', 'points', 'sizes']

# head_size_range_ratio = 0.1
seg_head_loss_weight = 0.1
seg_level_loss_weight = 0.1

# dataset
dataset_file = "JHU_General"
data_path = "./data/JHU/"
output_dir = "pet_model"
head_sizes_folder = "images_head_size_by_depth_var_512_2048"
seg_level_folder = "images_depth_512_2048"
seg_head_folder = "images_head_split_by_depth_var_512_2048"
seg_level_split_th = 0.2
head_size_weight = 1.5  # 0.8
