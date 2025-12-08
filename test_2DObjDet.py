"""2D Object Detection Testing Script for YOLO Models."""

import os
import sys
import pickle
import logging
import argparse
import traceback
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from tqdm import tqdm
import numpy as np

from utils.functions import get_dtypes, read_all_saved_param_idx, ANSI_COLORS
from helper import load_datasetloader, load_solvers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='2D Object Detection Testing (YOLO)')
    
    # Basic settings
    parser.add_argument('--exp_id', type=int, default=300, help='Experiment ID')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number')
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='Yolo')
    
    # Test settings
    parser.add_argument('--is_test_all', type=int, default=1, help='Test all checkpoints')
    parser.add_argument('--model_num', type=int, default=0, help='Specific model number to test')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS IoU threshold')
    
    return parser.parse_args()


def setup_logging(save_dir):
    """Setup logging to file and console."""
    logging.basicConfig(
        filename=os.path.join(save_dir, 'test.log'),
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


def load_checkpoint(solver, ckp_idx):
    """Load checkpoint for YOLO model."""
    file_name = f'{solver.save_dir}/saved_chk_point_{ckp_idx}.pt'
    if not os.path.exists(file_name):
        raise FileNotFoundError(f'Checkpoint not found: {file_name}')
    
    checkpoint = torch.load(file_name, map_location='cpu')
    
    # Handle DDP vs non-DDP model loading
    if hasattr(solver.model, 'module'):
        # DDP model
        solver.model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Regular model - remove 'module.' prefix if present
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        solver.model.load_state_dict(new_state_dict, strict=False)
    
    solver.log.info(f'>> Loaded parameters from {file_name}')
    if 'prev_mAP' in checkpoint:
        solver.log.info(f">> Previous mAP: {checkpoint['prev_mAP']:.4f}")


def convert_yolo_to_xyxy(boxes):
    """Convert YOLO format [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]."""
    if len(boxes) == 0:
        return boxes
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def post_process_predictions(raw_predictions, batch_size, n_cam, conf_threshold, nms_threshold):
    """Post-process raw YOLO predictions with confidence filtering and NMS."""
    predictions = []
    
    for i in range(batch_size * n_cam):
        boxes = raw_predictions['boxes'][i]  # (total_anchors, 4)
        objectness = raw_predictions['objectness'][i].squeeze(-1)  # (total_anchors,)
        class_logits = raw_predictions['labels'][i]  # (total_anchors, num_classes)
        
        # Get class probabilities and predicted class
        class_probs = F.softmax(class_logits, dim=-1)
        class_scores, class_ids = torch.max(class_probs, dim=-1)
        
        # Combined confidence: objectness * class_score
        confidence = objectness * class_scores
        
        # Filter by confidence threshold
        mask = confidence > conf_threshold
        if not mask.any():
            predictions.append({
                'boxes': torch.empty((0, 4), device=boxes.device),
                'labels': torch.empty((0,), dtype=torch.long, device=boxes.device),
                'scores': torch.empty((0,), device=boxes.device)
            })
            continue
        
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidence = confidence[mask]
        
        # Apply NMS
        if len(boxes) > 0:
            # Convert YOLO format to xyxy for NMS
            boxes_corners = convert_yolo_to_xyxy(boxes)
            keep_indices = nms(boxes_corners, confidence, nms_threshold)
        else:
            keep_indices = torch.tensor([], dtype=torch.long, device=boxes.device)
        
        predictions.append({
            'boxes': boxes[keep_indices],
            'labels': class_ids[keep_indices],
            'scores': confidence[keep_indices]
        })
    
    return predictions


def compute_map(all_predictions, all_targets, num_classes, iou_thresholds=[0.5]):
    """
    Compute mAP (mean Average Precision) for object detection.
    
    Args:
        all_predictions: List of dicts, each with 'boxes', 'labels', 'scores'
        all_targets: List of dicts, each with 'boxes', 'labels'
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds to compute mAP at
    
    Returns:
        dict with mAP values at each IoU threshold
    """
    map_scores = {}
    
    for iou_thresh in iou_thresholds:
        # Collect all predictions and targets per class
        class_predictions = defaultdict(list)  # {class_id: [(score, is_tp), ...]}
        class_target_counts = defaultdict(int)  # {class_id: count}
        
        for pred_dict, target_dict in zip(all_predictions, all_targets):
            pred_boxes = pred_dict['boxes'].cpu()
            pred_labels = pred_dict['labels'].cpu()
            pred_scores = pred_dict['scores'].cpu()
            gt_boxes = target_dict['boxes'].cpu()
            gt_labels = target_dict['labels'].cpu()
            
            # Convert YOLO format to xyxy
            if len(pred_boxes) > 0:
                pred_boxes_xyxy = convert_yolo_to_xyxy(pred_boxes)
            else:
                pred_boxes_xyxy = pred_boxes
            
            if len(gt_boxes) > 0:
                gt_boxes_xyxy = convert_yolo_to_xyxy(gt_boxes)
            else:
                gt_boxes_xyxy = gt_boxes
            
            # Process each class
            for class_id in range(num_classes):
                # Get predictions for this class
                class_mask = (pred_labels == class_id)
                if class_mask.sum() == 0:
                    continue
                
                class_pred_boxes = pred_boxes_xyxy[class_mask]
                class_pred_scores = pred_scores[class_mask]
                
                # Get ground truth for this class
                class_gt_mask = (gt_labels == class_id)
                class_gt_boxes = gt_boxes_xyxy[class_gt_mask]
                class_target_counts[class_id] += len(class_gt_boxes)
                
                if len(class_gt_boxes) == 0:
                    # No ground truth, all predictions are false positives
                    for score in class_pred_scores:
                        class_predictions[class_id].append((score.item(), False))
                    continue
                
                if len(class_pred_boxes) == 0:
                    continue
                
                # Compute IoU between predictions and ground truth
                ious = box_iou(class_pred_boxes, class_gt_boxes)  # (n_pred, n_gt)
                
                # Sort predictions by score (descending)
                sorted_indices = torch.argsort(class_pred_scores, descending=True)
                class_pred_boxes = class_pred_boxes[sorted_indices]
                class_pred_scores = class_pred_scores[sorted_indices]
                ious = ious[sorted_indices]
                
                # Match predictions to ground truth
                matched_gt = set()
                for i, (score, iou_row) in enumerate(zip(class_pred_scores, ious)):
                    # Find best matching ground truth
                    if len(iou_row) > 0:
                        max_iou, best_gt_idx = iou_row.max(dim=0)
                        if max_iou >= iou_thresh and best_gt_idx.item() not in matched_gt:
                            class_predictions[class_id].append((score.item(), True))
                            matched_gt.add(best_gt_idx.item())
                        else:
                            class_predictions[class_id].append((score.item(), False))
                    else:
                        class_predictions[class_id].append((score.item(), False))
        
        # Compute AP for each class
        aps = []
        for class_id in range(num_classes):
            if class_id not in class_predictions or len(class_predictions[class_id]) == 0:
                continue
            
            if class_target_counts[class_id] == 0:
                continue
            
            # Sort by score (descending)
            predictions = sorted(class_predictions[class_id], key=lambda x: x[0], reverse=True)
            
            # Compute precision-recall curve
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            for _, is_tp in predictions:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
                recall = tp_cumsum / (class_target_counts[class_id] + 1e-7)
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP using 11-point interpolation
            if len(recalls) == 0:
                ap = 0.0
            else:
                # 11-point interpolation
                ap = 0.0
                for t in np.arange(0, 1.1, 0.1):
                    if np.sum(np.array(recalls) >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.array(precisions)[np.array(recalls) >= t])
                    ap += p / 11.0
            
            aps.append(ap)
        
        # Compute mAP
        map_scores[iou_thresh] = np.mean(aps) if len(aps) > 0 else 0.0
    
    return map_scores


def test(args, logger):
    """Run 2D object detection testing."""
    C = ANSI_COLORS
    
    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    _, float_dtype = get_dtypes(useGPU=True)

    # Load saved configuration
    save_dir = f'./saved_models/{args.dataset_type}_{args.model_name}_model{args.exp_id}'
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f'Save directory not found: {save_dir}')
    
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    
    # Override settings for testing
    saved_args.batch_size = 1
    saved_args.ddp = 0
    saved_args.save_dir = save_dir
    saved_args.exp_id = args.exp_id


    # Load data and model
    dataset, data_loader, _ = load_datasetloader(
        args=saved_args, dtype=torch.FloatTensor, world_size=1, rank=0, mode='test'
    )
    solver = load_solvers(
        saved_args, dataset.num_samples, 
        world_size=1, rank=0, logger=logger, dtype=float_dtype, isTrain=False
    )

    # Get number of classes from config
    num_classes = solver.cfg.get('2DObjDet', {}).get('num_classes', 10)

    # Determine which checkpoints to test
    ckp_idx_list = read_all_saved_param_idx(solver.save_dir)
    if len(ckp_idx_list) == 0:
        logger.warning(f'No checkpoints found in {solver.save_dir}')
        return
    
    target_models = ckp_idx_list if args.is_test_all else [args.model_num]

    # Test each checkpoint
    for ckp_id in ckp_idx_list:
        if ckp_id not in target_models:
            logger.info(f'{C["DIM"]}[SKIP] Model {ckp_id} not in target list{C["RESET"]}')
            continue

        # Load model
        try:
            load_checkpoint(solver, ckp_id)
        except Exception as e:
            logger.error(f'Failed to load checkpoint {ckp_id}: {e}')
            continue
        
        solver.mode_selection(isTrain=False)

        all_predictions = []
        all_targets = []

        # Run inference
        for b_idx, batch in enumerate(tqdm(data_loader, desc=f'Test (ckp {ckp_id})')):
            images = batch['images'].cuda()
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            start_end_idx = batch['start_end_idx']
            
            with torch.no_grad():
                # Get raw predictions using forward()
                raw_predictions = solver.model(images)
                
                # Post-process predictions
                batch_size, n_cam = images.shape[:2]
                predictions = post_process_predictions(
                    raw_predictions, batch_size, n_cam, 
                    args.conf_threshold, args.nms_threshold
                )
                
                # Process each sample-camera pair
                cumulative_offset = 0
                for sample_idx in range(batch_size):
                    # Get the cumulative counts for this sample
                    sample_start_end = start_end_idx[sample_idx]  # (n_cam,)
                    
                    for cam_idx in range(n_cam):
                        pred_idx = sample_idx * n_cam + cam_idx
                        pred_dict = predictions[pred_idx]
                        
                        # Get ground truth indices for this sample-camera pair
                        if cam_idx == 0:
                            start_idx = cumulative_offset
                        else:
                            start_idx = cumulative_offset + sample_start_end[cam_idx - 1].item()
                        
                        end_idx = cumulative_offset + sample_start_end[cam_idx].item()
                        
                        # Update cumulative offset after processing last camera of this sample
                        if cam_idx == n_cam - 1:
                            cumulative_offset = end_idx
                        
                        # Get ground truth for this sample-camera
                        num_gt = end_idx - start_idx
                        if num_gt > 0:
                            gt_boxes_cam = gt_boxes[start_idx:end_idx]
                            gt_labels_cam = gt_labels[start_idx:end_idx]
                        else:
                            gt_boxes_cam = torch.empty((0, 4))
                            gt_labels_cam = torch.empty((0,), dtype=torch.long)
                        
                        all_predictions.append({
                            'boxes': pred_dict['boxes'].cpu(),
                            'labels': pred_dict['labels'].cpu(),
                            'scores': pred_dict['scores'].cpu()
                        })
                        all_targets.append({
                            'boxes': gt_boxes_cam.cpu(),
                            'labels': gt_labels_cam.cpu()
                        })

        # Compute mAP
        map_scores_50_75 = compute_map(all_predictions, all_targets, num_classes, iou_thresholds=[0.5, 0.75])
        map_50 = map_scores_50_75.get(0.5, 0.0)
        map_75 = map_scores_50_75.get(0.75, 0.0)
        
        # Compute mAP@0.5:0.95
        map_scores_full = compute_map(
            all_predictions, all_targets, num_classes,
            iou_thresholds=np.arange(0.5, 1.0, 0.05).tolist()
        )
        map_50_95 = np.mean(list(map_scores_full.values())) if map_scores_full else 0.0

        # Print results - disable console handler temporarily to avoid duplicates
        console_handlers = [h for h in logger.handlers 
                           if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        for h in console_handlers:
            logger.removeHandler(h)
        
        print(f"\n{C['CYAN']}{'─' * 50}{C['RESET']}")
        print(f"{C['BOLD']}📊 Results for Checkpoint {C['YELLOW']}{ckp_id}{C['RESET']}")
        print(f"{C['CYAN']}{'─' * 50}{C['RESET']}")
        logger.info(f"─── Results for Checkpoint {ckp_id} ───")
        
        # Print mAP results
        color_50 = C['GREEN'] if map_50 > 0.5 else (C['YELLOW'] if map_50 > 0.3 else C['RED'])
        color_75 = C['GREEN'] if map_75 > 0.5 else (C['YELLOW'] if map_75 > 0.3 else C['RED'])
        color_50_95 = C['GREEN'] if map_50_95 > 0.5 else (C['YELLOW'] if map_50_95 > 0.3 else C['RED'])
        
        print(f"  {'mAP@0.5':15s} │ {color_50}{map_50:.4f}{C['RESET']}")
        print(f"  {'mAP@0.75':15s} │ {color_75}{map_75:.4f}{C['RESET']}")
        print(f"  {'mAP@0.5:0.95':15s} │ {color_50_95}{map_50_95:.4f}{C['RESET']}")
        
        logger.info(f"  mAP@0.5: {map_50:.4f}")
        logger.info(f"  mAP@0.75: {map_75:.4f}")
        logger.info(f"  mAP@0.5:0.95: {map_50_95:.4f}")
        
        print(f"{C['CYAN']}{'─' * 50}{C['RESET']}\n")
        
        # Restore console handlers
        for h in console_handlers:
            logger.addHandler(h)


def main():
    args = parse_args()
    save_dir = f'./saved_models/{args.dataset_type}_{args.model_name}_model{args.exp_id}'
    logger = setup_logging(save_dir)
    
    try:
        test(args, logger)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(error_msg)  # Logs to both console and file
        print(f"\n{ANSI_COLORS['RED']}{'─' * 50}")
        print("❌ ERROR OCCURRED")
        print(f"{'─' * 50}{ANSI_COLORS['RESET']}")
        print(error_msg)


if __name__ == '__main__':
    main()
