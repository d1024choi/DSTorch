import numpy as np
import cv2
import PIL.Image as Image
from pathlib import Path
from pyquaternion import Quaternion

STATIC = ['lane', 'road_segment', 'ped_crossing', 'walkway', 'carpark_area', 'stop_line']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]
CLASSES = STATIC + DIVIDER + DYNAMIC

MAP_NAMES = ['singapore-onenorth',
             'singapore-hollandvillage',
             'singapore-queenstown',
             'boston-seaport']

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

DATA_KEYS = ['cam_idx', 'image', 'intrinsics', 'extrinsics', 'bev', 'view',
             'visibility', 'center', 'pose', 'future_egomotion', 'label', 'instance']

INTERPOLATION = cv2.LINE_8

def get_split(split, dataset_name):
    split_dir = Path(__file__).parent / 'nuscenes/splits'
    split_path = split_dir / f'{split}.txt'

    return split_path.read_text().strip().split('\n')