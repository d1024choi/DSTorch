import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--app_mode', type=str, default='2DObjDet', help='2DObjDet')

# ------------------------
# Exp Info
# ------------------------
parser.add_argument('--model_name', type=str, default='Yolo', help='Look up config/config.json')
parser.add_argument('--exp_id', type=int, default=300)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--load_pretrained', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--ddp', type=int, default=0)
parser.add_argument('--bool_mixed_precision', type=int, default=0)
parser.add_argument('--num_cores', type=int, default=2)

# ------------------------
# Dataset
# ------------------------
parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--dataset_type', type=str, default='nuscenes')
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--data_split_for_save', type=str, default='train', help='train, val, test')


# ------------------------
# Training Env
# ------------------------
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--valid_step', type=int, default=1)
parser.add_argument('--save_every', type=int, default=3)
parser.add_argument('--max_num_chkpts', type=int, default=5)

parser.add_argument('--optimizer_type', type=str, default='adamw', help='support adan and adamw only')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0, help='default for adam and adamW is 0 and 1e-2')
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--apply_lr_scheduling', type=int, default=0)
parser.add_argument('--lr_schd_type', type=str, default='none', help='StepLR, ExponentialLR, OnecycleLR')
parser.add_argument('--div_factor', type=float, default=10.0)
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--final_div_factor', type=float, default=10.0)

parser.add_argument('--bool_find_unused_params', type=int, default=0)
parser.add_argument('--bool_apply_img_aug', type=int, default=1)


args = parser.parse_args()
