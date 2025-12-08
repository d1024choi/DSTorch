"""Image augmentation utilities for camera data preprocessing.

This module provides augmentation functionality for camera images, including
random resizing and cropping, with corresponding updates to camera intrinsic parameters.
"""

from typing import Tuple
import numpy as np
from utils.geometry import *

class ImageAugmentation:
    """Image augmentation class for camera data.
    
    Applies random resize and crop transformations to images while maintaining
    consistency with camera intrinsic parameters. This is crucial for maintaining
    geometric correctness when projecting between image and 3D space.
    """

    def __init__(self, data_aug_conf):
        """Initialize augmentation with configuration.
        
        Args:
            data_aug_conf: Dictionary containing augmentation parameters:
                - final_dim: (height, width) of final output image
                - resize_lim: (min, max) range for random resize scale (optional)
                - resize_scale: Fixed resize scale if resize_lim not provided
                - crop_offset: Maximum random offset for crop center
        """
        self.data_aug_conf = data_aug_conf

    def sample_augmentation(self):
        """Sample random augmentation parameters (resize and crop).
        
        Returns:
            resize_dims: (width, height) tuple for resized image dimensions
            crop: (x0, y0, x1, y1) tuple defining crop region
        """
        fH, fW = self.data_aug_conf['final_dim']

        # Sample random resize scale
        if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
            # Random resize within specified limits
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
        else:
            # Use fixed resize scale
            resize = self.data_aug_conf['resize_scale']

        # Calculate resized dimensions
        resize_dims = (int(fW * resize), int(fH * resize))
        newW, newH = resize_dims

        # Calculate centered crop position
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)

        # Add random offset to crop center for data augmentation
        crop_offset = self.data_aug_conf['crop_offset']
        crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
        crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

        # Define crop region as (x0, y0, x1, y1)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        return resize_dims, crop

    def __call__(self, image, intrinsic):
        """Apply augmentation to image and update camera intrinsics.
        
        Args:
            image: PIL.Image format, RGB image (3 x H x W)
            intrinsic: torch.Tensor, 3x3 camera intrinsic matrix
        
        Returns:
            image: Augmented PIL.Image
            intrinsic: Updated 3x3 camera intrinsic matrix corresponding to augmented image
        """
        W, H = image.size
        resize_dims, crop = self.sample_augmentation()

        # Update intrinsic parameters to account for resize
        sx = resize_dims[0] / float(W)  # Scale factor in x direction
        sy = resize_dims[1] / float(H)  # Scale factor in y direction

        # Scale intrinsic matrix by resize factors
        intrinsic = scale_intrinsics(intrinsic.unsqueeze(0), sx, sy).squeeze(0)

        # Update principal point (x0, y0) to account for crop offset
        fx, fy, x0, y0 = split_intrinsics(intrinsic.unsqueeze(0))
        new_x0 = x0 - crop[0]  # Adjust principal point x by crop offset
        new_y0 = y0 - crop[1]  # Adjust principal point y by crop offset

        # Reconstruct intrinsic matrix with updated principal point
        pix_T_cam = merge_intrinsics(fx, fy, new_x0, new_y0)
        intrinsic = pix_T_cam.squeeze(0)

        # Apply resize and crop to image
        image = resize_and_crop_image(image, resize_dims, crop)

        return image, intrinsic[:3, :3]


