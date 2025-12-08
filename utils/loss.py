import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from fvcore.nn import sigmoid_focal_loss
import sys

DYNAMIC = ['pedestrian', 'vehicle']
STATIC_LARGE = ['drivable', 'walkway', 'carpark_area']
STATIC_SMALL = ['stop_line', 'ped_crossing', 'divider']


# --------------------------------
# Common
class Optimizers(nn.Module):
    """Optimizer wrapper supporting multiple optimizer types."""
    
    # Optimizer type mapping
    _OPTIMIZER_MAP = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
    }
    
    def __init__(self, model, optimizer_type, learning_rate, weight_decay, config=None):
        super().__init__()
        
        optimizer_type = optimizer_type.lower()
        if optimizer_type not in self._OPTIMIZER_MAP:
            sys.exit(f">> Optimizer {optimizer_type} is not supported! Available: {list(self._OPTIMIZER_MAP.keys())}")
        
        optimizer_class = self._OPTIMIZER_MAP[optimizer_type]
        self.opt = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

class LRScheduler(nn.Module):
    def __init__(self, optimizer, type='StepLR', config=None):
        super(LRScheduler, self).__init__()
        """ Required config keys
        1) StepLR : step_size, gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

        2) ExponentialLR : gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR

        3) OneCycleLR : max_lr, div_factor, final_div_factor, pct_start, steps_per_epoch, epochs, cycle_momentum
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        """

        if (type == 'StepLR' and config is not None):
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        elif (type == 'ExponentialLR' and config is not None):
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
        elif (type == 'OnecycleLR' and config is not None):
            self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=config['max_lr'],
                                                           div_factor=config['div_factor'], # starts at max_lr / 10
                                                           final_div_factor=config['final_div_factor'], # ends at lr / 10 / 10
                                                           pct_start=config['pct_start'], # reaches max_lr at 30% of total steps
                                                           steps_per_epoch=config['steps_per_epoch'],
                                                           epochs=config['epochs'],
                                                           cycle_momentum=False)
        else:
            sys.exit(f'[Error] LR scheduler named {type} is not implemented!!')

    def __call__(self):
        self.scheduler.step()


# --------------------------------
# 2D Object Detection Loss
class Yolo2DObjDetLoss(nn.Module):
    """
    YOLO Loss function for 2D object detection.
    
    Calculates losses from YOLO predictions and ground-truth labels.
    
    Args:
        cfg: Configuration dictionary. Can contain loss-specific settings under 'Loss' key.
        num_classes: Number of object classes
        lambda_coord: Weight for box coordinate loss (default: 5.0)
        lambda_noobj: Weight for no-object loss (default: 0.5)
        lambda_obj: Weight for object loss (default: 1.0)
        lambda_cls: Weight for classification loss (default: 1.0)
        iou_threshold: IoU threshold for positive/negative anchor assignment (default: 0.5)
    """
    
    def __init__(self, cfg, num_classes=10, lambda_coord=5.0, lambda_noobj=0.5, 
                 lambda_obj=1.0, lambda_cls=1.0, iou_threshold=0.5, anchor_sizes=None, img_size=640):
        super(Yolo2DObjDetLoss, self).__init__()
        
        # Extract configuration from cfg['Loss'] if available
        loss_cfg = cfg.get('2DObjDetLoss', {}) if isinstance(cfg, dict) else {}
        
        self.num_classes = num_classes
        self.lambda_coord = loss_cfg.get('lambda_coord', lambda_coord)
        self.lambda_noobj = loss_cfg.get('lambda_noobj', lambda_noobj)
        self.lambda_obj = loss_cfg.get('lambda_obj', lambda_obj)
        self.lambda_cls = loss_cfg.get('lambda_cls', lambda_cls)
        self.iou_threshold = loss_cfg.get('iou_threshold', iou_threshold)
        self.img_size = img_size
        
        # Anchor sizes - required for proper loss calculation
        if anchor_sizes is None:
            # Default anchor sizes (same as model)
            self.anchor_sizes = [
                [(10, 13), (16, 30), (33, 23)],  # Small objects
                [(30, 61), (62, 45), (59, 119)],  # Medium objects
                [(116, 90), (156, 198), (373, 326)]  # Large objects
            ]
        else:
            self.anchor_sizes = anchor_sizes
        
        self.num_anchors_per_scale = len(self.anchor_sizes[0])
        self.num_scales = len(self.anchor_sizes)
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def __call__(self, predictions, targets):
        """
        Calculate YOLO loss.
        
        Args:
            predictions: dict with keys:
                'boxes': torch.Tensor (batch_size * N_cam, N_anchors, 4)
                         - Predicted boxes in YOLO format [x_center, y_center, width, height] (normalized)
                'labels': torch.Tensor (batch_size * N_cam, N_anchors, num_classes)
                         - Class logits
                'objectness': torch.Tensor (batch_size * N_cam, N_anchors, 1)
                            - Objectness logits
            targets: dict with keys:
                'boxes': torch.Tensor (total_objects, 4) - Ground truth boxes
                'labels': torch.Tensor (total_objects,) - Ground truth class IDs
                'start_end_idx': torch.Tensor (batch_size, N_cam) - Cumulative indices
        
        Returns:
            loss_dict: dict with individual loss components and total loss
        """
        pred_boxes = predictions['boxes']  # (batch * n_cam, anchors, 4)
        pred_labels = predictions['labels']  # (batch * n_cam, anchors, num_classes)
        pred_objectness = predictions['objectness']  # (batch * n_cam, anchors, 1)
        
        # Safety check: detect NaN/Inf in predictions early
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            print("WARNING: NaN/Inf detected in pred_boxes!")
            pred_boxes = torch.where(torch.isfinite(pred_boxes), pred_boxes, torch.zeros_like(pred_boxes))
        
        gt_boxes = targets['boxes'].cuda()  # (total_objects, 4)
        gt_labels = targets['labels'].cuda()  # (total_objects,)
        start_end_idx = targets['start_end_idx'].cuda()  # (batch_size, n_cam)
        
        # Safety check: ensure GT boxes are valid
        if len(gt_boxes) > 0:
            # Ensure GT boxes have valid dimensions (width, height > 0)
            gt_widths = gt_boxes[:, 2]
            gt_heights = gt_boxes[:, 3]
            valid_gt = (gt_widths > 1e-6) & (gt_heights > 1e-6)
            if not valid_gt.all():
                print(f"WARNING: {valid_gt.sum().item()}/{len(gt_boxes)} GT boxes have invalid dimensions!")
        
        batch_size, n_cam = start_end_idx.shape
        device = pred_boxes.device
        
        # Initialize losses
        box_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        obj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        noobj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        cls_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        
        # Track number of positive and negative samples for normalization
        num_positives = 0
        num_negatives = 0
        
        # Calculate cumulative object counts across all samples
        # start_end_idx is cumulative per sample, so we need to offset by previous samples
        cumulative_offset = 0
        
        # Process each sample-camera pair
        for batch_idx in range(batch_size):
            # Get the cumulative counts for this sample
            sample_start_end = start_end_idx[batch_idx]  # (n_cam,)
            
            for cam_idx in range(n_cam):
                # Get prediction index (flattened)
                pred_idx = batch_idx * n_cam + cam_idx
                
                # Get ground truth indices for this sample-camera pair
                # start_end_idx[batch_idx] contains cumulative counts for this sample's cameras
                if cam_idx == 0:
                    start_idx = cumulative_offset
                else:
                    start_idx = cumulative_offset + sample_start_end[cam_idx - 1].item()
                
                end_idx = cumulative_offset + sample_start_end[cam_idx].item()
                
                # Update cumulative offset after processing last camera of this sample
                if cam_idx == n_cam - 1:
                    cumulative_offset = end_idx
                
                # Get predictions for this sample-camera
                pred_boxes_cam = pred_boxes[pred_idx]  # (anchors, 4)
                pred_labels_cam = pred_labels[pred_idx]  # (anchors, num_classes)
                pred_obj_cam = pred_objectness[pred_idx].squeeze(-1)  # (anchors,)
                
                # Get ground truth for this sample-camera
                num_gt = end_idx - start_idx
                
                if num_gt == 0:
                    # No objects in this camera - all anchors are negative
                    noobj_loss += self.lambda_noobj * self.bce_loss(
                        pred_obj_cam, 
                        torch.zeros_like(pred_obj_cam)
                    ).mean()
                    num_negatives += len(pred_obj_cam)
                    continue
                
                gt_boxes_cam = gt_boxes[start_idx:end_idx]  # (num_gt, 4)
                gt_labels_cam = gt_labels[start_idx:end_idx]  # (num_gt,)
                
                # The model now properly predicts anchor-relative offsets and decodes them.
                # pred_boxes_cam contains decoded boxes in normalized [0,1] format:
                # - Decoded from offsets: x = (sigmoid(tx) + grid_x) / grid_w
                # - Decoded from anchors: w = anchor_w * exp(tw) / img_w
                # These decoded boxes are used for IoU matching and loss calculation.
                
                # Match ground truth boxes to anchors using IoU on decoded boxes
                # Shape: (anchors, num_gt)
                ious = self._compute_iou(pred_boxes_cam, gt_boxes_cam)
                
                # For each anchor, find best matching ground truth
                best_iou_per_anchor, best_gt_per_anchor = ious.max(dim=1)  # (anchors,)
                
                # Positive anchors: IoU > threshold
                positive_mask = best_iou_per_anchor > self.iou_threshold  # (anchors,)
                negative_mask = ~positive_mask  # (anchors,)
                
                num_positives += positive_mask.sum().item()
                num_negatives += negative_mask.sum().item()
                
                # Box regression loss (only for positive anchors)
                if positive_mask.any():
                    pos_anchors = positive_mask.nonzero(as_tuple=False).squeeze(-1)
                    matched_gt_indices = best_gt_per_anchor[pos_anchors]
                    
                    pred_boxes_pos = pred_boxes_cam[pos_anchors]  # (num_pos, 4) - decoded boxes
                    gt_boxes_matched = gt_boxes_cam[matched_gt_indices]  # (num_pos, 4)
                    
                    # Calculate box loss (MSE) on decoded boxes
                    # These boxes are already decoded from anchor-relative offsets in the model
                    box_diff = self.mse_loss(pred_boxes_pos, gt_boxes_matched)  # (num_pos, 4)
                    # Sum over all coordinates and average over positive anchors
                    box_loss += self.lambda_coord * box_diff.sum()
                
                # Objectness loss
                # Positive anchors should predict objectness = 1
                if positive_mask.any():
                    obj_targets = torch.zeros_like(pred_obj_cam)
                    obj_targets[positive_mask] = 1.0
                    obj_loss += self.lambda_obj * self.bce_loss(
                        pred_obj_cam[positive_mask],
                        obj_targets[positive_mask]
                    ).mean()
                
                # No-object loss (negative anchors)
                if negative_mask.any():
                    noobj_loss += self.lambda_noobj * self.bce_loss(
                        pred_obj_cam[negative_mask],
                        torch.zeros_like(pred_obj_cam[negative_mask])
                    ).mean()
                
                # Classification loss (only for positive anchors)
                if positive_mask.any():
                    pos_anchors = positive_mask.nonzero(as_tuple=False).squeeze(-1)
                    matched_gt_indices = best_gt_per_anchor[pos_anchors]
                    matched_gt_labels = gt_labels_cam[matched_gt_indices]  # (num_pos,)
                    
                    pred_labels_pos = pred_labels_cam[pos_anchors]  # (num_pos, num_classes)
                    cls_loss += self.lambda_cls * self.ce_loss(
                        pred_labels_pos,
                        matched_gt_labels
                    ).mean()
        
        # Normalize losses (with safety checks to prevent NaN)
        if num_positives > 0:
            box_loss = box_loss / num_positives
            obj_loss = obj_loss / num_positives
            cls_loss = cls_loss / num_positives
        else:
            # If no positives, set losses to zero (not NaN)
            box_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
            obj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
            cls_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        
        if num_negatives > 0:
            noobj_loss = noobj_loss / num_negatives
        else:
            noobj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        
        # Check for NaN/Inf before computing total loss
        if torch.isnan(box_loss) or torch.isinf(box_loss):
            print("WARNING: NaN/Inf in box_loss!")
            box_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        if torch.isnan(obj_loss) or torch.isinf(obj_loss):
            print("WARNING: NaN/Inf in obj_loss!")
            obj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        if torch.isnan(noobj_loss) or torch.isinf(noobj_loss):
            print("WARNING: NaN/Inf in noobj_loss!")
            noobj_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            print("WARNING: NaN/Inf in cls_loss!")
            cls_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        
        # Total loss
        total_loss = box_loss + obj_loss + noobj_loss + cls_loss
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("WARNING: NaN/Inf in total_loss! Returning zero loss.")
            total_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype)
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'cls_loss': cls_loss,
            'num_positives': num_positives,
            'num_negatives': num_negatives
        }
    
    def _decode_boxes_with_anchors(self, pred_boxes, img_size):
        """
        Decode predicted boxes using anchor information.
        
        Since we don't have grid cell positions in flattened predictions,
        we use a simplified approach: treat predictions as if they're already
        in normalized coordinates but scale them appropriately using anchor information.
        
        Args:
            pred_boxes: torch.Tensor (N_anchors, 4) - predicted boxes in normalized [0,1] format
            img_size: int - image size (assumes square images)
        
        Returns:
            decoded_boxes: torch.Tensor (N_anchors, 4) - decoded boxes in normalized format
        """
        # For now, we'll use a simplified approach:
        # Since the model outputs are already normalized [0,1], we treat them as
        # absolute coordinates. However, we should ideally decode using anchors.
        
        # TODO: Proper anchor decoding requires:
        # 1. Grid cell positions for each anchor
        # 2. Scale information for each anchor
        # 3. Proper offset decoding: decoded = anchor + offset
        
        # For now, return predictions as-is but note that proper implementation
        # should decode: x_decoded = (sigmoid(tx) + grid_x) / grid_w
        #                w_decoded = anchor_w * exp(tw) / img_w
        
        # This is a placeholder - the real fix requires changes to the model architecture
        # to output anchor-relative offsets instead of absolute coordinates
        return pred_boxes
    
    def _generate_anchor_boxes(self, grid_h, grid_w, scale_idx, img_size):
        """
        Generate anchor boxes for a given grid and scale.
        
        Args:
            grid_h: int - grid height
            grid_w: int - grid width  
            scale_idx: int - scale index (0=small, 1=medium, 2=large)
            img_size: int - image size
        
        Returns:
            anchor_boxes: torch.Tensor (grid_h, grid_w, num_anchors, 4) - anchor boxes in normalized format
        """
        anchors = self.anchor_sizes[scale_idx]  # List of (w, h) tuples
        num_anchors = len(anchors)
        device = next(self.parameters()).device
        
        # Create grid coordinates
        y_coords = torch.arange(grid_h, device=device).float()
        x_coords = torch.arange(grid_w, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize grid coordinates
        grid_x_norm = (grid_x + 0.5) / grid_w  # Center of grid cell
        grid_y_norm = (grid_y + 0.5) / grid_h
        
        # Create anchor boxes
        anchor_boxes = torch.zeros(grid_h, grid_w, num_anchors, 4, device=device)
        
        for a_idx, (anchor_w, anchor_h) in enumerate(anchors):
            # Normalize anchor sizes
            anchor_w_norm = anchor_w / img_size
            anchor_h_norm = anchor_h / img_size
            
            # Set anchor box: [x_center, y_center, width, height] in normalized coordinates
            anchor_boxes[:, :, a_idx, 0] = grid_x_norm
            anchor_boxes[:, :, a_idx, 1] = grid_y_norm
            anchor_boxes[:, :, a_idx, 2] = anchor_w_norm
            anchor_boxes[:, :, a_idx, 3] = anchor_h_norm
        
        return anchor_boxes
    
    def _compute_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: torch.Tensor (N, 4) - YOLO format [x_center, y_center, width, height]
            boxes2: torch.Tensor (M, 4) - YOLO format [x_center, y_center, width, height]
        
        Returns:
            iou: torch.Tensor (N, M) - IoU matrix
        """
        # Safety check: ensure boxes are valid
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device, dtype=boxes1.dtype)
        
        # Convert YOLO format to corner format
        def yolo_to_corners(boxes):
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # Ensure width and height are positive (safety check)
            width = torch.clamp(width, min=1e-6)
            height = torch.clamp(height, min=1e-6)
            
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            # Ensure valid box coordinates
            x_min = torch.clamp(x_min, min=0.0, max=1.0)
            y_min = torch.clamp(y_min, min=0.0, max=1.0)
            x_max = torch.clamp(x_max, min=0.0, max=1.0)
            y_max = torch.clamp(y_max, min=0.0, max=1.0)
            
            return torch.stack([x_min, y_min, x_max, y_max], dim=1)
        
        corners1 = yolo_to_corners(boxes1)  # (N, 4)
        corners2 = yolo_to_corners(boxes2)  # (M, 4)
        
        # Expand dimensions for broadcasting
        corners1 = corners1.unsqueeze(1)  # (N, 1, 4)
        corners2 = corners2.unsqueeze(0)  # (1, M, 4)
        
        # Calculate intersection
        x_min = torch.max(corners1[:, :, 0], corners2[:, :, 0])  # (N, M)
        y_min = torch.max(corners1[:, :, 1], corners2[:, :, 1])  # (N, M)
        x_max = torch.min(corners1[:, :, 2], corners2[:, :, 2])  # (N, M)
        y_max = torch.min(corners1[:, :, 3], corners2[:, :, 3])  # (N, M)
        
        # Calculate intersection area
        inter_width = torch.clamp(x_max - x_min, min=0)
        inter_height = torch.clamp(y_max - y_min, min=0)
        intersection = inter_width * inter_height  # (N, M)
        
        # Calculate union area
        area1 = (corners1[:, :, 2] - corners1[:, :, 0]) * (corners1[:, :, 3] - corners1[:, :, 1])  # (N, 1)
        area2 = (corners2[:, :, 2] - corners2[:, :, 0]) * (corners2[:, :, 3] - corners2[:, :, 1])  # (1, M)
        union = area1 + area2 - intersection  # (N, M)
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)  # (N, M)
        
        return iou
