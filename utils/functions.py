# from utils.libraries import *
import json
import os
import glob
import sys
import numpy as np
import shutil
import pickle
from pathlib import Path
import cv2
import time
from tqdm import tqdm
import logging
import traceback
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# ANSI color codes for terminal output
ANSI_COLORS = {
    'CYAN': "\033[96m", 'GREEN': "\033[92m", 'YELLOW': "\033[93m",
    'MAGENTA': "\033[95m", 'RED': "\033[91m", 'BLUE': "\033[94m",
    'BOLD': "\033[1m", 'DIM': "\033[2m", 'RESET': "\033[0m"
}


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_config(path=None):

    if (path is None):
        cfg = read_json(path='./config/config.json')
        cfg.update(read_json(path=f'./config/data.json'))
        cfg.update(read_json(path=f'./config/model.json'))
        cfg.update(read_json(path=f'./config/loss.json'))

    else:
        file_path = os.path.join(path, 'config.json')
        cfg = read_json(path=file_path)

        file_path = os.path.join(path, f'data.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'model.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'loss.json')
        cfg.update(read_json(path=file_path))

    cfg['nuscenes']['dataset_dir'] = check_dataset_path_existence(cfg['nuscenes']['dataset_dir'])

    return cfg

def config_update(cfg, args):
    '''
    Merge all args into cfg. Access via cfg['args']['param_name'] or legacy paths.
    '''
    # Store all args as dictionary
    cfg['args'] = vars(args).copy()

    # Legacy mappings for backward compatibility
    cfg['app_mode'] = args.app_mode
    cfg['model_name'] = args.model_name

    return cfg

def toNP(x):
    return x.detach().to('cpu').numpy()

def toTS(x, dtype):
    return torch.from_numpy(x).to(dtype)

def check_dataset_path_existence(path, candidates=['etri', 'dooseop']):
    """Check if path exists, trying different user directories."""
    cur_id = next((c for c in candidates if c in path), None)
    if cur_id is None:
        sys.exit(f" [Error] {path} doesn't exist!")
    for candi in candidates:
        new_path = path.replace(cur_id, candi)
        if os.path.exists(new_path):
            return new_path
    sys.exit(f" [Error] {path} doesn't exist!")

def get_dtypes(useGPU=True):
    return torch.LongTensor, torch.FloatTensor

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

