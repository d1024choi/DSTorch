"""NuScenes Dataset Loader for 2D Object Detection (YOLO Training)."""

import os
import sys
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box as shapely_box
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

# Add project root to Python path for imports (only when running this file directly)
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from utils.functions import read_config, config_update
from dataset.NuscenesDataset.common import CAMERAS


class DatasetLoader(Dataset):
    """NuScenes dataset loader for 2D object detection (YOLO format)."""

    # YOLO class mapping: nuScenes category -> YOLO class ID
    # Common classes for object detection
    CLASS_MAPPING = {
        'car': 0,
        'truck': 1,
        'bus': 2,
        'trailer': 3,
        'construction_vehicle': 4,
        'pedestrian': 5,
        'motorcycle': 6,
        'bicycle': 7,
        'traffic_cone': 8,
        'barrier': 9,
    }
    
    # Visibility levels to include (empty string means all, '1', '2', '3', '4' are visibility bins)
    DEFAULT_VISIBILITIES = ['', '1', '2', '3', '4']

    def __init__(self, args, dtype, world_size=None, rank=None, mode='train'):
        random.seed(1024)

        # Configuration
        self.args, self.rank, self.world_size, self.mode, self.dtype = args, rank, world_size, mode, dtype
        self.cfg = config_update(read_config(), args)

        # Convert tensor type to dtype if necessary (e.g., torch.FloatTensor -> torch.float32)
        try:
            # Try to create an empty tensor to get the dtype
            test_tensor = dtype([])
            self.dtype = test_tensor.dtype
        except (TypeError, ValueError):
            # If it's already a dtype, use it directly
            self.dtype = dtype

        # Get 2DObjDet config if available, otherwise use defaults
        objdet_cfg = self.cfg.get('2DObjDet', {})
        
        # Image preprocessing parameters
        if 'image' in objdet_cfg:
            self.img_h = objdet_cfg['image'].get('h', self.cfg.get('img_h', 640))
            self.img_w = objdet_cfg['image'].get('w', self.cfg.get('img_w', 640))
            self.img_top_crop = objdet_cfg['image'].get('top_crop', self.cfg.get('img_top_crop', 0))
        else:
            self.img_h = self.cfg.get('img_h', 640)
            self.img_w = self.cfg.get('img_w', 640)
            self.img_top_crop = self.cfg.get('img_top_crop', 0)
        
        # Original image dimensions from nuScenes
        if 'original_image' in objdet_cfg:
            self.ori_img_w = objdet_cfg['original_image'].get('w', 1600)
            self.ori_img_h = objdet_cfg['original_image'].get('h', 900)
        else:
            self.ori_img_w = 1600
            self.ori_img_h = 900
        
        # Image normalization, debugged 251208
        # Standard ImageNet normalization for 2D object detection (YOLO)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])
        
        # TODO : apply augmentation if training
        # # Augmentation (can be enabled via config)
        # bool_apply_img_aug = objdet_cfg.get('bool_apply_img_aug', self.cfg.get('bool_apply_img_aug', False))
        # self.apply_augmentation = self.mode == 'train' and bool_apply_img_aug
        
        # Visibility filter
        self.visibilities = objdet_cfg.get('visibilities', self.cfg.get('visibilities', self.DEFAULT_VISIBILITIES))
        
        # Camera selection (use all cameras by default, or specify in config)
        self.cameras = objdet_cfg.get('cameras', self.cfg.get('cameras', CAMERAS))
        
        # Load nuScenes data
        split = 'train' if mode in ['train', 'val', 'valid'] else 'val'
        self._load_nuscenes_data(world_size, split, mode)

        if rank == 0 and mode == 'train':
            print(f">> Dataset loaded from {os.path.basename(__file__)}")

    def _load_nuscenes_data(self, world_size, split, mode):
        """Load data directly from NuScenes."""
        # Initialize NuScenes
        dataset_dir = self.cfg['nuscenes']['dataset_dir']
        version = self.cfg['nuscenes']['version']
        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=False)
        _ = world_size  # May be used for distributed training
        
        if self.rank == 0:
            print(f">> nuScenes loaded: {version}")

        # Get scene splits
        self.target_scenes = self._get_split()
        
        # Get all sample records from target scenes
        self.sample_records = self._get_ordered_sample_records()
        random.shuffle(self.sample_records)
        # Debug ----
        # self.sample_records = self.sample_records[:1024]
        # Debug ----

        # Split into train/val/test
        if mode in ['train', 'val', 'valid']:
            num_val = int(len(self.sample_records) * self.args.val_ratio)
            num_train = len(self.sample_records) - num_val
            if mode == 'train':
                self.sample_records = self.sample_records[:num_train]
            else:
                self.sample_records = self.sample_records[num_train:]
                # Note : we have to repeat 'world_size' times because DDP will split the data into 'world_size' parts
                # val_scenes = []
                # for r in range(world_size):
                #     val_scenes += self.sample_records[num_train:]
                # self.sample_records = val_scenes
                   
        self.num_samples = len(self.sample_records)
        # self.num_samples = 100       
        
        if self.rank == 0:
            print(f">> Loaded {self.num_samples} samples for {mode}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample for 2D object detection.
        
        Returns:
            image: torch.Tensor (C, H, W) - normalized image
            targets: List[Dict] - List of targets for each camera
                Each target dict contains:
                    'boxes': torch.Tensor (N, 4) - YOLO format [x_center, y_center, width, height] (normalized)
                    'labels': torch.Tensor (N,) - class IDs
                    'camera': str - camera name
        """
        sample_record = self.sample_records[idx]
        
        # Get all camera images and annotations
        images, boxes, labels, start_end_idx = [], [], [], [] 
        for camera in self.cameras:
            cam_token = sample_record['data'].get(camera)
            if cam_token is None:
                continue
                
            # Get image and annotations for this camera
            image, target = self._get_camera_data(sample_record, camera, cam_token)
            start_end_idx.append(target['boxes'].size(0)) # count the number of objects for each camera
            images.append(image)
            boxes.append(target['boxes'])
            labels.append(target['labels'])
        
        images = torch.stack(images) # (N_cam, C, H, W)
        boxes = torch.cat(boxes, dim=0) # (N_obj, 4)
        labels = torch.cat(labels) # (N_obj,)
        start_end_idx = torch.cumsum(torch.from_numpy(np.array(start_end_idx)), dim=0) 
       
        return images, boxes, labels, start_end_idx

    def _get_camera_data(self, sample_record, camera, cam_token):
        """Get image and 2D bounding box annotations for a camera."""
        # Get camera record
        cam_record = self.nusc.get('sample_data', cam_token)
        
        # Skip if not a keyframe (for consistency)
        if not cam_record.get('is_key_frame', True):
            return None, None
        
        # Load and preprocess image
        img_path = Path(self.nusc.get_sample_data_path(cam_token))
        if not img_path.exists():
            return None, None
        
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Resize and crop image
        image = self._preprocess_image(image)
        
        # Get 2D bounding boxes
        boxes_2d, labels = self._get_2d_boxes(sample_record, cam_token, cam_record, original_size)
        
        if len(boxes_2d) == 0:
            # Return empty target
            target = {
                'boxes': torch.zeros((0, 4), dtype=self.dtype),
                'labels': torch.zeros((0,), dtype=torch.long),
                'camera': camera
            }
        else:
            # Convert to YOLO format (normalized: x_center, y_center, width, height)
            boxes_yolo = self._convert_to_yolo_format(boxes_2d, original_size)
            
            target = {
                'boxes': torch.tensor(boxes_yolo, dtype=self.dtype),
                'labels': torch.tensor(labels, dtype=torch.long),
                'camera': camera
            }
        
        
        # Verification code -----
        # Uncomment the following lines to visualize samples during data loading
        # if len(boxes_2d) > 0:  # Only visualize if there are boxes
        #     save_dir = Path('./verification_vis')
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     save_path = save_dir / f'camera_{camera}_sample_{sample_record["token"][:8]}.png'
        #     self._visualize_2d_boxes(image, target, save_path=save_path, show=False)
        # Verification code -----
        
        # Convert image to tensor
        image_tensor = self.img_transform(image)
        
        return image_tensor, target

    def _preprocess_image(self, image):
        """Preprocess image: resize and crop."""
        # Resize
        image = image.resize((self.img_w, self.img_h + self.img_top_crop), Image.Resampling.BILINEAR)
        
        # Crop top if needed
        if self.img_top_crop > 0:
            image = image.crop((0, self.img_top_crop, self.img_w, self.img_h + self.img_top_crop))
        
        # TODO : apply augmentation if training
        # # Apply augmentation if training
        # if self.apply_augmentation:
        #     # Simple augmentation: random horizontal flip
        #     if random.random() < 0.5:
        #         image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        return image

    def _visualize_2d_boxes(self, image, target, save_path=None, show=False):
        """Visualize 2D bounding boxes on an image.
        
        Args:
            image: PIL Image - the image to visualize
            target: Dict with 'boxes' (N, 4) in YOLO format [x_center, y_center, width, height] (normalized),
                    'labels' (N,), and 'camera' (str)
            save_path: str or Path - path to save the visualization (optional)
            show: bool - whether to display the image (default: False)
        """
        try:
            # Set non-interactive backend for headless environments
            import matplotlib
            matplotlib.use('Agg')
            
            # Convert PIL image to numpy array
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # Get class names
            class_names = list(self.CLASS_MAPPING.keys())
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_np)
            ax.axis('off')
            
            # Get boxes and labels
            boxes = target['boxes']
            labels = target['labels']
            camera = target.get('camera', 'unknown')
            
            # Convert tensors to numpy if needed
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            # Color palette for different classes
            try:
                colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
            except AttributeError:
                colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(class_names)))
            
            # Draw bounding boxes
            for box, label in zip(boxes, labels):
                # Convert from YOLO format (normalized x_center, y_center, width, height)
                # to (x_min, y_min, x_max, y_max)
                x_center, y_center, box_w, box_h = box
                
                # Denormalize
                x_center = float(x_center) * w
                y_center = float(y_center) * h
                box_w = float(box_w) * w
                box_h = float(box_h) * h
                
                # Convert to corner coordinates
                x_min = x_center - box_w / 2
                y_min = y_center - box_h / 2
                
                # Get class name and color
                class_id = int(label)
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                color = colors[class_id % len(colors)]
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x_min, y_min), box_w, box_h,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x_min, max(0, y_min - 5), class_name,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                        fontsize=10, color='white', weight='bold')
            
            # Add title
            num_objects = len(boxes)
            ax.set_title(f'Camera: {camera} | Objects: {num_objects}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                if self.rank == 0 or self.rank is None:
                    print(f"  Visualization saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"  Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            try:
                plt.close()
            except:
                pass

    def _get_2d_boxes(self, sample_record, cam_token, cam_record, imsize):
        """Project 3D boxes to 2D image coordinates.
        
        Args:
            sample_record: Sample record from nuScenes (unused but kept for API consistency)
            cam_token: Camera token (unused but kept for API consistency)
            cam_record: Camera sample data record
            imsize: Original image size (width, height)
        
        Returns:
            boxes_2d: List of [x_min, y_min, x_max, y_max] in original image coordinates
            labels: List of class IDs
        """
        _ = sample_record, cam_token  # Unused but kept for API consistency
        # Get sample
        sample = self.nusc.get('sample', cam_record['sample_token'])
        
        # Get calibrated sensor and ego pose
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        boxes_2d = []
        labels = []
        
        # Get all annotations for this sample
        for ann_token in sample['anns']:
            ann_rec = self.nusc.get('sample_annotation', ann_token)
            
            # Filter by visibility
            if ann_rec['visibility_token'] not in self.visibilities:
                continue
            
            # Get category and map to YOLO class
            category_name = ann_rec['category_name']
            class_id = self._category_to_class_id(category_name)
            
            if class_id is None:
                continue  # Skip unmapped categories
            
            # Get 3D box
            box_3d = self.nusc.get_box(ann_token)
            
            # Transform box to camera frame
            # Move to ego-pose frame
            box_3d.translate(-np.array(pose_record['translation']))
            box_3d.rotate(Quaternion(pose_record['rotation']).inverse)
            
            # Move to calibrated sensor frame
            box_3d.translate(-np.array(cs_record['translation']))
            box_3d.rotate(Quaternion(cs_record['rotation']).inverse)
            
            # Project to 2D
            box_2d = self._project_box_to_2d(box_3d, camera_intrinsic, imsize)
            
            if box_2d is not None:
                boxes_2d.append(box_2d)
                labels.append(class_id)
        
        return boxes_2d, labels

    def _project_box_to_2d(self, box_3d, camera_intrinsic, imsize):
        """Project a 3D box to 2D image coordinates.
        
        Args:
            box_3d: Box object in camera frame
            camera_intrinsic: Camera intrinsic matrix (3x3)
            imsize: Image size (width, height)
        
        Returns:
            [x_min, y_min, x_max, y_max] or None if box is not visible
        """
        # Get 3D box corners
        corners_3d = box_3d.corners()  # (3, 8)
        
        # Filter corners in front of camera
        in_front = corners_3d[2, :] > 0.1  # At least 0.1m in front
        if not np.any(in_front):
            return None
        
        corners_3d = corners_3d[:, in_front]
        
        # Project to 2D
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True).T[:, :2]
        
        # Get bounding box from projected corners
        x_coords = corners_2d[:, 0]
        y_coords = corners_2d[:, 1]
        
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        # Check intersection with image
        box_2d_polygon = MultiPoint(corners_2d).convex_hull
        img_canvas = shapely_box(0, 0, imsize[0], imsize[1])
        
        if not box_2d_polygon.intersects(img_canvas):
            return None
        
        # Get intersection
        intersection = box_2d_polygon.intersection(img_canvas)
        if intersection.is_empty:
            return None
        
        # Get bounding box of intersection
        intersection_coords = np.array([coord for coord in intersection.exterior.coords])
        x_min = max(0, min(intersection_coords[:, 0]))
        y_min = max(0, min(intersection_coords[:, 1]))
        x_max = min(imsize[0], max(intersection_coords[:, 0]))
        y_max = min(imsize[1], max(intersection_coords[:, 1]))
        
        # Check if box is valid
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # Check minimum size (filter very small boxes)
        min_box_size = 5  # pixels
        if (x_max - x_min) < min_box_size or (y_max - y_min) < min_box_size:
            return None
        
        return [x_min, y_min, x_max, y_max]

    def _convert_to_yolo_format(self, boxes_2d, imsize):
        """Convert boxes from [x_min, y_min, x_max, y_max] to YOLO format.
        
        YOLO format: [x_center, y_center, width, height] (all normalized 0-1)
        
        Args:
            boxes_2d: List of [x_min, y_min, x_max, y_max]
            imsize: Image size (width, height)
        
        Returns:
            List of [x_center, y_center, width, height] (normalized)
        """
        boxes_yolo = []
        img_w, img_h = imsize
        
        for box in boxes_2d:
            x_min, y_min, x_max, y_max = box
            
            # Calculate center and dimensions
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            
            # Normalize
            x_center_norm = x_center / img_w
            y_center_norm = y_center / img_h
            width_norm = width / img_w
            height_norm = height / img_h
            
            boxes_yolo.append([x_center_norm, y_center_norm, width_norm, height_norm])
        
        return boxes_yolo

    def _category_to_class_id(self, category_name):
        """Map nuScenes category to YOLO class ID.
        
        Args:
            category_name: nuScenes category name (e.g., 'vehicle.car', 'human.pedestrian.adult')
        
        Returns:
            Class ID or None if not mapped
        """
        category_lower = category_name.lower()
        
        # Map common categories
        if 'car' in category_lower:
            return self.CLASS_MAPPING['car']
        elif 'truck' in category_lower:
            return self.CLASS_MAPPING['truck']
        elif 'bus' in category_lower:
            return self.CLASS_MAPPING['bus']
        elif 'trailer' in category_lower:
            return self.CLASS_MAPPING['trailer']
        elif 'construction' in category_lower:
            return self.CLASS_MAPPING['construction_vehicle']
        elif 'pedestrian' in category_lower:
            return self.CLASS_MAPPING['pedestrian']
        elif 'motorcycle' in category_lower:
            return self.CLASS_MAPPING['motorcycle']
        elif 'bicycle' in category_lower:
            return self.CLASS_MAPPING['bicycle']
        elif 'traffic_cone' in category_lower or 'cone' in category_lower:
            return self.CLASS_MAPPING['traffic_cone']
        elif 'barrier' in category_lower:
            return self.CLASS_MAPPING['barrier']
        
        return None  # Unmapped category

    def _get_split(self):
        """Get scene names for a split."""
        
        if (self.mode in ['train', 'val', 'valid']):
            split = 'train' # train and valid data from train.txt
        else:
            split = 'val' # test data from val.txt
        
        split_path = Path(__file__).parent / f'splits/{split}.txt'
        if split_path.exists():
            return split_path.read_text().strip().split('\n')
        else:
            sys.exit(f">> Split file not found at {split_path}")

    def _get_ordered_sample_records(self):
        """Get sample records ordered by scene and timestamp."""
        samples = [
            s for s in self.nusc.sample
            if self.nusc.get('scene', s['scene_token'])['name'] in self.target_scenes
        ]
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples


def visualize_2d_boxes(image, target, class_names=None, save_path=None, show=False):
    """Standalone function to visualize 2D bounding boxes on an image.
    
    Args:
        image: PIL Image or numpy array - the image to visualize
        target: Dict with 'boxes' (N, 4) in YOLO format [x_center, y_center, width, height] (normalized),
                'labels' (N,), and optionally 'camera' (str)
        class_names: List[str] - class names (optional, defaults to standard YOLO classes)
        save_path: str or Path - path to save the visualization (optional)
        show: bool - whether to display the image (default: False)
    """
    try:
        # Set non-interactive backend for headless environments
        import matplotlib
        matplotlib.use('Agg')
        
        # Convert image to numpy array
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            # Handle tensor images (C, H, W) or (H, W, C)
            if len(image.shape) == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                else:  # (H, W, C)
                    img_np = image.cpu().numpy()
            else:
                img_np = image.cpu().numpy()
            # Denormalize if needed (assuming ImageNet normalization)
            if img_np.min() < 0:  # Likely normalized
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        h, w = img_np.shape[:2]
        
        # Default class names
        if class_names is None:
            class_names = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                          'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_np)
        ax.axis('off')
        
        # Get boxes and labels
        boxes = target['boxes']
        labels = target['labels']
        camera = target.get('camera', 'unknown')
        
        # Convert tensors to numpy if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Color palette for different classes
        try:
            colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        except AttributeError:
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(class_names)))
        
        # Draw bounding boxes
        for box, label in zip(boxes, labels):
            # Convert from YOLO format (normalized x_center, y_center, width, height)
            # to (x_min, y_min, x_max, y_max)
            x_center, y_center, box_w, box_h = box
            
            # Denormalize
            x_center = float(x_center) * w
            y_center = float(y_center) * h
            box_w = float(box_w) * w
            box_h = float(box_h) * h
            
            # Convert to corner coordinates
            x_min = x_center - box_w / 2
            y_min = y_center - box_h / 2
            
            # Get class name and color
            class_id = int(label)
            class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x_min, max(0, y_min - 5), class_name,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        # Add title
        num_objects = len(boxes)
        ax.set_title(f'Camera: {camera} | Objects: {num_objects}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close()
        except:
            pass

