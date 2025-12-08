import os
import sys
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.ops import box_iou, nms

from utils.functions import (read_config, config_update, save_read_latest_checkpoint_num, 
                             remove_past_checkpoint)

from utils.print import (print_training_info, ANSI_COLORS, progress_bar, format_metrics, format_metrics_inline)

from utils.loss import Yolo2DObjDetLoss, Optimizers, LRScheduler
from utils.metrics import IoUMetric
from models.Yolo.yolo import YOLO

RENEW_KEYS = ['total_loss', 'box_loss', 'obj_loss', 'noobj_loss', 'cls_loss']

# ANSI color codes (for terminal only)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
WIDTH = 70

class Solver:
    '''Yolo model solver for training, evaluation, and checkpoint management.'''

    def __init__(self, args, num_train_scenes, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):

        '''
        Initialize the Yolo model solver.

        Args:
            args: Argument parser containing model configuration.
            num_train_scenes: Number of training scenes.
            world_size: Number of processes for distributed training.
            rank: Rank of the process.
            logger: Logger object for logging.
            dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
            isTrain: Whether to train the model.
        '''


        # save folder path
        self.save_dir = os.path.join('./saved_models/', f'{args.dataset_type}_{args.model_name}_model{args.exp_id}')
        self.args, self.rank, self.world_size = args, rank, world_size
        self.log, self.dtype = logger, dtype

        # Load or save configuration
        self._handle_config(args, isTrain)
        # self._set_target_classes()
        
        # Print training info (only on rank 0)
        if self.rank == 0:
            print_training_info(self.args, logger, return_print_dict())
        
        self.num_batches = num_train_scenes // (args.batch_size * world_size)
        self.monitor = {'iter': 0, 'total_loss': 0, 'box_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'cls_loss': 0, 'prev_mAP': 0}

        # Model setup
        model = YOLO(self.cfg, rank=rank)
        self.model = self._setup_model(model, args, dtype, rank)


        # Optimizer, loss, and scheduler
        self.opt = Optimizers(self.model, args.optimizer_type, args.learning_rate, args.weight_decay).opt
        self.loss = Yolo2DObjDetLoss(self.cfg)
        self.lr_scheduler = LRScheduler(self.opt, type='OnecycleLR', config={
            'max_lr': args.learning_rate, 'div_factor': args.div_factor, 'final_div_factor': args.final_div_factor,
            'pct_start': args.pct_start, 'steps_per_epoch': self.num_batches, 'epochs': args.num_epochs})

        if rank == 0:
            print(f">> Optimizer loaded from {os.path.basename(__file__)}")

    def _handle_config(self, args, isTrain):
        '''Handle configuration loading/saving.'''
        config_path = os.path.join(self.save_dir, 'config.pkl')
        config_dict_path = os.path.join(self.save_dir, 'config_dict.pkl')

        if isTrain:
            if args.load_pretrained:
                if not os.path.exists(self.save_dir):
                    sys.exit(f'>> Path {self.save_dir} does not exist!')
                with open(config_path, 'rb') as f:
                    self.args = pickle.load(f)
                self.args.load_pretrained = 1
            elif self.rank == 0:
                with open(config_path, 'wb') as f:
                    pickle.dump(args, f)

            self.cfg = config_update(read_config(), self.args)
            with open(config_dict_path, 'wb') as f:
                pickle.dump(self.cfg, f)
        else:
            if os.path.exists(config_dict_path):
                with open(config_dict_path, 'rb') as f:
                    self.cfg = pickle.load(f)
            else:
                self.cfg = config_update(read_config(), args)

    def _setup_model(self, model, args, dtype, rank):
        '''Setup model for single or multi-GPU training.'''
        if args.ddp:
            model.type(dtype).to(rank)
            return DDP(model, device_ids=[rank], find_unused_parameters=bool(args.bool_find_unused_params))
        return model.type(dtype).cuda()


    # --- Mode & Loss Management ---
    def mode_selection(self, isTrain=True):
        if isTrain:
            self.model.train()
        else:
            self.model.eval()

    def init_loss_tracker(self):
        for key in RENEW_KEYS:
            self.monitor[key] = 0

    def normalize_loss_tracker(self):
        for key in RENEW_KEYS:
            self.monitor[key] /= self.num_batches

    def learning_rate_step(self, _e=None):
        if self.args.apply_lr_scheduling:
            self.lr_scheduler()

    # --- Checkpoint Management ---
    # def load_pretrained_network_params(self, ckp_idx):
    #     file_name = f'{self.save_dir}/saved_chk_point_{ckp_idx}.pt'
    #     checkpoint = torch.load(file_name, map_location='cpu')

    #     if self.args.ddp:
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #     else:
    #         state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in checkpoint['model_state_dict'].items())
    #         self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})

    #     self.opt.load_state_dict(checkpoint['opt'])
    #     self.lr_scheduler.scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     self.monitor.update({'iter': checkpoint['iter'], 'prev_IoU': checkpoint['prev_IoU']})
    #     self.cfg = checkpoint['cfg']
    #     self.log.info(f'>> Loaded parameters from {file_name}')
    #     self.log.info(f">> Current IoU: {self.monitor['prev_IoU']:.4f}")

    def save_trained_network_params(self, e):
        save_read_latest_checkpoint_num(self.save_dir, e, isSave=True)
        file_name = f'{self.save_dir}/saved_chk_point_{e}.pt'
        torch.save({
            'epoch': e, 'model_state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(), 'opt': self.opt.state_dict(),
            'prev_mAP': self.monitor['prev_mAP'], 'iter': self.monitor['iter'], 'cfg': self.cfg
        }, file_name)
        self.log.info(">> Network saved")
        remove_past_checkpoint(self.save_dir, max_num=self.args.max_num_chkpts)

    # --- Progress Printing ---
    def print_status(self, e, start_epoch, end_epoch):
        if self.rank == 0:
            C = ANSI_COLORS
            hrs_left = (end_epoch - start_epoch) * (self.args.num_epochs - e - 1) / 3600.0
            
            # ========== EASILY ADD/MODIFY METRICS HERE ==========
            metrics = [
                {'label': '⏱ ETA',  'value': hrs_left, 'fmt': '.1f', 'color': 'GREEN'},
                {'label': 'Box Loss', 'value': self.monitor['box_loss'], 'fmt': '.4f', 'color': 'MAGENTA'},
                {'label': 'Obj Loss', 'value': self.monitor['obj_loss'], 'fmt': '.4f', 'color': 'MAGENTA'},
                {'label': 'NoObj Loss', 'value': self.monitor['noobj_loss'], 'fmt': '.4f', 'color': 'MAGENTA'},
                {'label': 'Cls Loss', 'value': self.monitor['cls_loss'], 'fmt': '.4f', 'color': 'MAGENTA'},
                {'label': 'LR',     'value': self.opt.param_groups[0]['lr'], 'fmt': '.2e', 'color': 'CYAN'},
                # Add more metrics here easily:
                # {'label': 'IoU',  'value': self.monitor.get('iou', 0), 'fmt': '.4f', 'color': 'GREEN'},
            ]
            # ====================================================
            
            colored_metrics, plain_metrics = format_metrics(metrics)
            
            # Colored terminal output
            epoch_str = f"{C['BOLD']}Epoch {C['YELLOW']}{e:03d}{C['RESET']}{C['DIM']}/{self.args.num_epochs}{C['RESET']}"
            print(f"{C['CYAN']}━━━{C['RESET']} {epoch_str} {C['DIM']}│{C['RESET']} {colored_metrics} {C['CYAN']}━━━{C['RESET']}")
            
            # Plain text for log file
            self.log.info(f"━━━ Epoch {e:03d}/{self.args.num_epochs} │ {plain_metrics} ━━━")

    def print_training_progress(self, e, b, elapsed):
        if self.rank == 0:
            C = ANSI_COLORS
            if b >= self.num_batches - 2:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
            else:
                pbar = progress_bar(b, self.num_batches)
                
                # ========== EASILY ADD/MODIFY METRICS HERE ==========
                metrics = [
                    {'label': '⏱', 'value': elapsed, 'fmt': '.3f', 'color': 'DIM'},
                    {'label': 'Total Loss', 'value': self.monitor['total_loss'] / self.monitor['iter'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    {'label': 'Box Loss', 'value': self.monitor['box_loss'] / self.monitor['iter'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    {'label': 'Obj Loss', 'value': self.monitor['obj_loss'] / self.monitor['iter'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    {'label': 'NoObj Loss', 'value': self.monitor['noobj_loss'] / self.monitor['iter'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    {'label': 'Cls Loss', 'value': self.monitor['cls_loss'] / self.monitor['iter'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    # Add more metrics here easily:
                    # {'label': 'Grad', 'value': grad_norm, 'fmt': '.2f', 'color': 'YELLOW'},
                ]
                # ====================================================
                
                metric_str = format_metrics_inline(metrics)
                
                line = (
                    f"\r {C['BOLD']}🚀 Train{C['RESET']} "
                    f"{C['DIM']}E{C['RESET']}{C['YELLOW']}{e:03d}{C['RESET']} "
                    f"{pbar} "
                    f"{C['DIM']}({b+1}/{self.num_batches}){C['RESET']} "
                    f"{metric_str}"
                )
                sys.stdout.write(line)
            sys.stdout.flush()

    def print_validation_progress(self, b, num_batches, **extra_metrics):
        """
        Print validation progress. Pass extra metrics as kwargs:
            print_validation_progress(b, num_batches, IoU=0.45, Loss=0.12)
        """
        if self.rank == 0:
            C = ANSI_COLORS
            if b >= num_batches - 2:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
            else:
                pbar = progress_bar(b, num_batches)
                
                # ========== EASILY ADD/MODIFY METRICS HERE ==========
                metrics = [
                    # Add default metrics here if needed
                ]
                # Add any extra metrics passed as kwargs
                for label, value in extra_metrics.items():
                    metrics.append({'label': label, 'value': value, 'fmt': '.4f', 'color': 'GREEN'})
                # ====================================================
                
                metric_str = format_metrics_inline(metrics) if metrics else ""
                extra_str = f" {metric_str}" if metric_str else ""
                
                line = (
                    f"\r {C['BOLD']}🔍 Valid{C['RESET']} "
                    f"{pbar} "
                    f"{C['DIM']}({b+1}/{num_batches}){C['RESET']}"
                    f"{extra_str}"
                )
                sys.stdout.write(line)
            sys.stdout.flush()


    def train(self, batch):
        # self.opt.zero_grad()
        
        pred = self.model(batch['images'].cuda())

        # Loss calculation
        losses = self.loss(pred, batch)
     
        # Backpropagation
        losses['total_loss'].backward()
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.opt.step()

        self.monitor.update({'total_loss': self.monitor['total_loss'] + losses['total_loss'].item(), 
                             'box_loss': self.monitor['box_loss'] + losses['box_loss'].item(),
                             'obj_loss': self.monitor['obj_loss'] + losses['obj_loss'].item(),
                             'noobj_loss': self.monitor['noobj_loss'] + losses['noobj_loss'].item(),
                             'cls_loss': self.monitor['cls_loss'] + losses['cls_loss'].item(),
                             'iter': self.monitor['iter'] + 1})

    def _convert_yolo_to_xyxy(self, boxes, img_size=640):
        """
        Convert YOLO format [x_center, y_center, width, height] (normalized) to [x_min, y_min, x_max, y_max] (pixel coordinates).
        
        Args:
            boxes: Tensor of shape (N, 4) in normalized YOLO format [0-1]
            img_size: Image size (assumes square images, default 640)
        
        Returns:
            Tensor of shape (N, 4) in pixel coordinates [x_min, y_min, x_max, y_max]
        """
        # Denormalize first
        x_center = boxes[:, 0] * img_size
        y_center = boxes[:, 1] * img_size
        width = boxes[:, 2] * img_size
        height = boxes[:, 3] * img_size
        
        # Convert to corner coordinates
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def _compute_map(self, all_predictions, all_targets, iou_thresholds=[0.5]):
        """
        Compute mAP (mean Average Precision) for object detection.
        Optimized version that processes data more efficiently.
        
        Args:
            all_predictions: List of dicts, each with 'boxes', 'labels', 'scores'
            all_targets: List of dicts, each with 'boxes', 'labels'
            iou_thresholds: List of IoU thresholds to compute mAP at
        
        Returns:
            dict with mAP values at each IoU threshold
        """
        num_classes = self.cfg.get('2DObjDet', {}).get('num_classes', 10)
        # Get image size from config (default 640)
        img_size = self.cfg.get('2DObjDet', {}).get('image', {}).get('w', 640)
        map_scores = {}
        
        # Pre-convert all boxes to xyxy format (batch conversion is faster)
        pred_boxes_list = []
        pred_labels_list = []
        pred_scores_list = []
        gt_boxes_list = []
        gt_labels_list = []
        
        for pred_dict, target_dict in zip(all_predictions, all_targets):
            pred_boxes = pred_dict['boxes'].cpu()
            pred_labels = pred_dict['labels'].cpu()
            pred_scores = pred_dict['scores'].cpu()
            gt_boxes = target_dict['boxes'].cpu()
            gt_labels = target_dict['labels'].cpu()
            
            # Convert YOLO format (normalized) to xyxy (pixel coordinates)
            if len(pred_boxes) > 0:
                pred_boxes_xyxy = self._convert_yolo_to_xyxy(pred_boxes, img_size=img_size)
            else:
                pred_boxes_xyxy = pred_boxes
            
            if len(gt_boxes) > 0:
                gt_boxes_xyxy = self._convert_yolo_to_xyxy(gt_boxes, img_size=img_size)
            else:
                gt_boxes_xyxy = gt_boxes
            
            pred_boxes_list.append(pred_boxes_xyxy)
            pred_labels_list.append(pred_labels)
            pred_scores_list.append(pred_scores)
            gt_boxes_list.append(gt_boxes_xyxy)
            gt_labels_list.append(gt_labels)
        
        for iou_thresh in iou_thresholds:
            # Collect all predictions and targets per class
            class_predictions = defaultdict(list)  # {class_id: [(score, is_tp), ...]}
            class_target_counts = defaultdict(int)  # {class_id: count}
            
            # Process each sample
            for pred_boxes_xyxy, pred_labels, pred_scores, gt_boxes_xyxy, gt_labels in zip(
                pred_boxes_list, pred_labels_list, pred_scores_list, gt_boxes_list, gt_labels_list
            ):
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
                        # Use list comprehension for faster appending
                        class_predictions[class_id].extend(
                            [(score.item(), False) for score in class_pred_scores]
                        )
                        continue
                    
                    if len(class_pred_boxes) == 0:
                        continue
                    
                    # Sort predictions by score (descending) before IoU computation
                    sorted_indices = torch.argsort(class_pred_scores, descending=True)
                    class_pred_boxes = class_pred_boxes[sorted_indices]
                    class_pred_scores = class_pred_scores[sorted_indices]
                    
                    # Compute IoU between predictions and ground truth
                    ious = box_iou(class_pred_boxes, class_gt_boxes)  # (n_pred, n_gt)
                    
                    # Match predictions to ground truth (vectorized where possible)
                    matched_gt = set()
                    if ious.numel() > 0:  # Check if IoU matrix is not empty
                        max_ious, best_gt_indices = ious.max(dim=1)  # Get max IoU for each prediction
                    else:
                        # No IoU to compute (shouldn't happen if we have both preds and gts)
                        max_ious = torch.zeros(len(class_pred_scores), device=class_pred_scores.device)
                        best_gt_indices = torch.zeros(len(class_pred_scores), dtype=torch.long, device=class_pred_scores.device)
                    
                    for score, max_iou, best_gt_idx in zip(class_pred_scores, max_ious, best_gt_indices):
                        best_gt_idx_item = best_gt_idx.item()
                        if max_iou >= iou_thresh and best_gt_idx_item not in matched_gt:
                            class_predictions[class_id].append((score.item(), True))
                            matched_gt.add(best_gt_idx_item)
                        else:
                            class_predictions[class_id].append((score.item(), False))
            
            # Compute AP for each class
            aps = []
            classes_with_data = []
            for class_id in range(num_classes):
                if class_id not in class_predictions or len(class_predictions[class_id]) == 0:
                    continue
                
                if class_target_counts[class_id] == 0:
                    continue
                
                classes_with_data.append(class_id)
                
                # Sort by score (descending) - already sorted in most cases, but ensure it
                predictions = sorted(class_predictions[class_id], key=lambda x: x[0], reverse=True)
                
                # Compute precision-recall curve (vectorized)
                is_tp_array = np.array([is_tp for _, is_tp in predictions])
                tp_cumsum = np.cumsum(is_tp_array)
                fp_cumsum = np.cumsum(~is_tp_array)
                
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
                recalls = tp_cumsum / (class_target_counts[class_id] + 1e-7)
                
                # Compute AP using 11-point interpolation
                if len(recalls) == 0:
                    ap = 0.0
                else:
                    # 11-point interpolation (vectorized)
                    ap = 0.0
                    for t in np.arange(0, 1.1, 0.1):
                        recall_mask = recalls >= t
                        if np.any(recall_mask):
                            p = np.max(precisions[recall_mask])
                        else:
                            p = 0
                        ap += p / 11.0
                
                aps.append(ap)
            
            # Compute mAP
            map_scores[iou_thresh] = np.mean(aps) if len(aps) > 0 else 0.0
            
            # Debug logging (only for first threshold to avoid spam)
            if iou_thresh == iou_thresholds[0] and len(classes_with_data) > 0:
                # This will be logged, but we need to pass it back somehow
                # For now, we'll add it as a comment that can be uncommented for debugging
                pass  # Can add debug logging here if needed
        
        return map_scores

    def _compute_simple_metrics(self, all_predictions, all_targets, iou_threshold=0.5):
        """
        Compute simple precision/recall metrics at a single IoU threshold.
        Much faster than full mAP computation.
        
        Returns:
            dict with 'precision', 'recall', 'f1' at the given IoU threshold
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred_dict, target_dict in zip(all_predictions, all_targets):
            pred_boxes = pred_dict['boxes']
            pred_labels = pred_dict['labels']
            gt_boxes = target_dict['boxes']
            gt_labels = target_dict['labels']
            
            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Get image size from config
            img_size = self.cfg.get('2DObjDet', {}).get('image', {}).get('w', 640)
            # Convert to xyxy for IoU computation (denormalize to pixel coordinates)
            pred_boxes_xyxy = self._convert_yolo_to_xyxy(pred_boxes, img_size=img_size)
            gt_boxes_xyxy = self._convert_yolo_to_xyxy(gt_boxes, img_size=img_size)
            
            # Compute IoU
            ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # (n_pred, n_gt)
            
            # Match predictions to ground truth (greedy matching)
            matched_gt = set()
            for pred_idx in range(len(pred_boxes)):
                if len(gt_boxes) == 0:
                    total_fp += 1
                    continue
                
                # Find best matching GT
                max_iou, best_gt_idx = ious[pred_idx].max(dim=0)
                best_gt_idx = best_gt_idx.item()
                
                # Check if labels match and IoU is above threshold
                if (max_iou >= iou_threshold and 
                    best_gt_idx not in matched_gt and
                    pred_labels[pred_idx] == gt_labels[best_gt_idx]):
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
            
            # Count unmatched ground truth as false negatives
            total_fn += len(gt_boxes) - len(matched_gt)
        
        # Compute metrics
        precision = total_tp / (total_tp + total_fp + 1e-7)
        recall = total_tp / (total_tp + total_fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

    def eval(self, dataset, dataloader, _sampler, e):
        
        num_batches = len(dataloader)
        self.mode_selection(isTrain=False)
        
        # Set epoch for validation sampler if using DDP
        if self.args.ddp and _sampler is not None:
            _sampler.set_epoch(e)
        
        all_predictions, all_targets = [], []
        
        # All ranks participate in evaluation
        with torch.no_grad():
            for b, batch in enumerate(dataloader):
                # Only rank 0 prints progress
                if self.rank == 0:
                    self.print_validation_progress(b, num_batches - 1)
                                
                images = batch['images'].cuda()
                gt_boxes = batch['boxes']
                gt_labels = batch['labels']
                start_end_idx = batch['start_end_idx']
                
                # Get raw predictions using forward() (works with DDP)
                raw_predictions = self.model(images)
                
                # Post-process predictions (replicate predict() logic)
                batch_size, n_cam = images.shape[:2]
                predictions = []
                conf_threshold = 0.5  # Confidence threshold - might be too high for early training
                nms_threshold = 0.4
                
                # Get image size for denormalization (boxes from model are normalized [0, 1])
                img_size = self.cfg.get('2DObjDet', {}).get('image', {}).get('w', images.shape[-1])
                
                for i in range(batch_size * n_cam):
                    boxes = raw_predictions['boxes'][i]  # (total_anchors, 4) - normalized [0, 1]
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
                    
                    boxes = boxes[mask]  # Still normalized [0, 1]
                    class_ids = class_ids[mask]
                    confidence = confidence[mask]
                    
                    # Apply NMS
                    # Note: torchvision.ops.nms expects boxes in pixel coordinates, not normalized
                    if len(boxes) > 0:
                        # Denormalize boxes to pixel coordinates for NMS
                        x_center = boxes[:, 0] * img_size  # Denormalize
                        y_center = boxes[:, 1] * img_size
                        width = boxes[:, 2] * img_size
                        height = boxes[:, 3] * img_size
                        
                        # Convert YOLO format to xyxy for NMS (in pixel coordinates)
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2
                        boxes_corners = torch.stack([x_min, y_min, x_max, y_max], dim=1)
                        
                        keep_indices = nms(boxes_corners, confidence, nms_threshold)
                    else:
                        keep_indices = torch.tensor([], dtype=torch.long, device=boxes.device)
                    
                    predictions.append({
                        'boxes': boxes[keep_indices],
                        'labels': class_ids[keep_indices],
                        'scores': confidence[keep_indices]
                    })
                
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
        
        # Gather results from all ranks to rank 0 for mAP computation
        if self.args.ddp:
            # Gather all predictions and targets from all processes
            gathered_predictions = [None] * self.world_size
            gathered_targets = [None] * self.world_size
            
            dist.all_gather_object(gathered_predictions, all_predictions)
            dist.all_gather_object(gathered_targets, all_targets)
            
            # Flatten the gathered results on rank 0
            if self.rank == 0:
                all_predictions = [item for sublist in gathered_predictions for item in sublist]
                all_targets = [item for sublist in gathered_targets for item in sublist]
        else:
            # Single GPU case - use results directly
            pass
        
        # Only rank 0 computes mAP and logs
        if self.rank == 0:
            # Debug: Check if we have predictions and targets
            total_preds = sum(len(p['boxes']) for p in all_predictions)
            total_targets = sum(len(t['boxes']) for t in all_targets)
            self.log.info(f">> Debug - Total predictions: {total_preds}, Total targets: {total_targets}")
            
            if total_preds > 0:
                # Sample a few predictions to check format
                sample_pred = all_predictions[0]
                if len(sample_pred['boxes']) > 0:
                    self.log.info(f">> Debug - Sample pred box (normalized): {sample_pred['boxes'][0].tolist()}")
                    self.log.info(f">> Debug - Sample pred label: {sample_pred['labels'][0].item()}, score: {sample_pred['scores'][0].item():.4f}")
                    # Check converted box
                    img_size = self.cfg.get('2DObjDet', {}).get('image', {}).get('w', 640)
                    converted_box = self._convert_yolo_to_xyxy(sample_pred['boxes'][0:1], img_size=img_size)
                    self.log.info(f">> Debug - Sample pred box (pixel coords): {converted_box[0].tolist()}")
            
            if total_targets > 0:
                # Sample a few targets to check format
                sample_target = all_targets[0]
                if len(sample_target['boxes']) > 0:
                    self.log.info(f">> Debug - Sample target box (normalized): {sample_target['boxes'][0].tolist()}")
                    self.log.info(f">> Debug - Sample target label: {sample_target['labels'][0].item()}")
                    # Check converted box
                    img_size = self.cfg.get('2DObjDet', {}).get('image', {}).get('w', 640)
                    converted_box = self._convert_yolo_to_xyxy(sample_target['boxes'][0:1], img_size=img_size)
                    self.log.info(f">> Debug - Sample target box (pixel coords): {converted_box[0].tolist()}")
            
            # Compute mAP@0.5:0.95
            # TODO : This may take a lot of time so rank0 may lose your connection to other servers.
            #        Need to figure out better way to compute mAP or new metrics to monitor.
            map_scores_full = self._compute_map(all_predictions, all_targets, iou_thresholds=[0.5, 0.95])
            map_50_95 = np.mean(list(map_scores_full.values())) if map_scores_full else 0.0
            
            # Debug: Log individual mAP scores
            if map_scores_full:
                for thresh, score in map_scores_full.items():
                    self.log.info(f">> Debug - mAP@{thresh}: {score:.4f}")

            self.log.info(f">> Evaluation mAP@0.5:0.95: {map_50_95:.4f}")
            if 'prev_mAP' not in self.monitor:
                self.monitor['prev_mAP'] = 0.0
            if self.monitor['prev_mAP'] < map_50_95:
                self.monitor['prev_mAP'] = map_50_95
                self.save_trained_network_params(e)
        
        # Note: Synchronization is handled by barrier at start of next training iteration in train.py (line 103)


def return_print_dict():
        
    return {
        # Items without section (printed first)
        'exp_id': {
            'Label': 'Experiment ID',
            'Color': GREEN,
            'use_color': True
        },
        'gpu_num': {
            'Label': 'GPU Number',
            'Color': GREEN,
            'use_color': True
        },
        'num_epochs': {
            'Label': 'Epochs',
            'Color': GREEN,
            'use_color': True
        },
        'batch_size': {
            'Label': 'Batch Size',
            'Color': GREEN,
            'use_color': True
        },
        
        
        # Second section: "Optimizer Settings" (all items with this section grouped together)
        'optimizer_type': {
            'Label': 'Optimizer',
            'Color': MAGENTA,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
        'learning_rate': {
            'Label': 'Learning Rate',
            'Color': GREEN,
            'use_color': True,
            'format': '.5f',
            'section': 'Optimizer Settings'
        },
        'weight_decay': {
            'Label': 'Weight Decay',
            'Color': GREEN,
            'use_color': True,
            'format': '.8f',
            'section': 'Optimizer Settings'
        },
        'apply_lr_scheduling': {
            'Label': 'LR Scheduling',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'Optimizer Settings'
        },
        'lr_schd_type': {
            'Label': 'LR Schedule Type',
            'Color': YELLOW,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
       

    }