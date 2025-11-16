#!/usr/bin/env python3
"""
Ensemble Inference Pipeline for Document Analysis

Two-stage approach:
1. Main YOLOv11 detection model: Extract all stamps, signatures, QR codes
2. Segmentation model: Refine stamp/signature crops for more accurate boundaries

This combines the broad detection capability of the main model with the 
precise segmentation of the specialized model.
"""

import os
import sys
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

def find_best_model(base_dir: str, model_type: str = "detection", use_latest=False) -> Optional[str]:
    """
    Find the best trained model by highest mAP or most recent timestamp
    
    Args:
        base_dir: Base directory to search (e.g., "runs/train" or "runs/segment")
        model_type: Type of model ("detection" or "segmentation")
        use_latest: If True, select most recent run and load last.pt; else select best mAP and load best.pt
    
    Returns:
        Path to model weights or None if not found
    """
    if not os.path.exists(base_dir):
        return None
    
    weight_name = "last.pt" if use_latest else "best.pt"
    
    if use_latest:
        # Select most recent run by directory modification time
        latest_model_path = None
        latest_time = 0
        latest_run = None
        
        for run_dir in Path(base_dir).iterdir():
            if not run_dir.is_dir():
                continue
            
            # Skip signature_only models for detection
            if model_type == "detection" and run_dir.name.startswith('signature_only'):
                continue
            
            weights_file = run_dir / "weights" / weight_name
            if not weights_file.exists():
                continue
            
            dir_time = run_dir.stat().st_mtime
            if dir_time > latest_time:
                latest_time = dir_time
                latest_model_path = str(weights_file)
                latest_run = run_dir.name
        
        if latest_model_path:
            print(f"  Auto-selected latest {model_type} model: {latest_run}")
        
        return latest_model_path
    
    # Select best model by mAP
    best_model_path = None
    best_map = -1
    best_run = None
    
    # Determine which mAP column to look for
    map_column = 'mask_mAP50-95' if model_type == 'segmentation' else 'mAP50-95'
    
    # Search all run directories
    for run_dir in Path(base_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        # Skip signature_only models for detection
        if model_type == "detection" and run_dir.name.startswith('signature_only'):
            continue
        
        # Check for training_metrics.csv
        metrics_file = run_dir / "training_metrics.csv"
        weights_file = run_dir / "weights" / weight_name
        
        if not weights_file.exists():
            continue
        
        # If metrics file exists, try to parse it for best mAP
        if metrics_file.exists():
            try:
                import csv
                with open(metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Look for the appropriate mAP column
                        if map_column in rows[0]:
                            max_map = max(float(row[map_column]) for row in rows if row[map_column])
                        elif 'mAP50-95' in rows[0]:
                            max_map = max(float(row['mAP50-95']) for row in rows if row['mAP50-95'])
                        else:
                            continue
                        
                        if max_map > best_map:
                            best_map = max_map
                            best_model_path = str(weights_file)
                            best_run = run_dir.name
            except Exception as e:
                # If metrics parsing fails, still consider this model but with low priority
                if best_model_path is None:
                    best_model_path = str(weights_file)
                    best_run = run_dir.name
                    best_map = 0
        else:
            # No metrics file, but model exists - use as fallback
            if best_model_path is None:
                best_model_path = str(weights_file)
                best_run = run_dir.name
                best_map = 0
    
    if best_model_path:
        if best_map > 0:
            print(f"  Auto-selected {model_type} model: {best_run} (mAP: {best_map:.4f})")
        else:
            print(f"  Auto-selected {model_type} model: {best_run} (no metrics available)")
    
    return best_model_path


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference"""
    
    # Models (will be auto-detected if None)
    MAIN_MODEL_PATH: Optional[str] = None
    SEG_MODEL_PATH: Optional[str] = None
    
    # Input/Output
    TEST_IMAGES_DIR: str = "data/datasets/main/images/val"
    VAL_LABELS_DIR: str = "data/datasets/main/labels/val"
    OUTPUT_DIR: str = "outputs/inference/ensemble"
    
    # Stage 1: Main model detection
    MAIN_CONF_THRESHOLD: float = 0.25
    MAIN_IMGSZ: int = 1024  # Must match training image size
    
    # Stage 2: Segmentation refinement
    SEG_CONF_THRESHOLD: float = 0.25
    SEG_IMGSZ: int = 640  # Segmentation model uses 640 (from seg training)
    CROP_MARGIN: float = 0.30  # Increased from 0.15 - seg model needs more context
    
    # Classes
    CLASS_NAMES: Dict[int, str] = None
    
    # Visualization
    COLORS: Dict[str, Tuple[int, int, int]] = None
    
    # Metrics
    SAVE_METRICS_CSV: bool = True
    SAVE_DETECTIONS_JSON: bool = True
    SAVE_VISUALIZATIONS: bool = True
    
    def __post_init__(self):
        if self.CLASS_NAMES is None:
            self.CLASS_NAMES = {
                0: 'qr',
                1: 'signature', 
                2: 'stamp'
            }
        
        if self.COLORS is None:
            self.COLORS = {
                'qr': (255, 0, 0),       # Blue
                'signature': (0, 255, 0), # Green
                'stamp': (0, 0, 255)      # Red
            }


# ============================================================================
# ENSEMBLE INFERENCE ENGINE
# ============================================================================

class EnsembleInference:
    """Two-stage ensemble inference combining detection and segmentation models"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.main_model = None
        self.seg_model = None
        self.group_iou_threshold = 0.3  # IoU threshold for grouping overlapping boxes
        
        # Reverse mapping for class names to IDs
        self.class_name_to_id = {v: k for k, v in config.CLASS_NAMES.items()}
        
    def load_models(self, use_latest=False):
        """Load both detection and segmentation models
        
        Args:
            use_latest: If True, load last.pt instead of best.pt
        """
        print("Loading models...")
        
        model_type = "latest" if use_latest else "best"
        
        # Auto-detect main model if not specified
        if self.config.MAIN_MODEL_PATH is None:
            print(f"  Auto-detecting {model_type} detection model...")
            self.config.MAIN_MODEL_PATH = find_best_model("runs/train", "detection", use_latest=use_latest)
            if self.config.MAIN_MODEL_PATH is None:
                raise FileNotFoundError("No trained detection models found in runs/train")
        
        # Load main detection model
        if not os.path.exists(self.config.MAIN_MODEL_PATH):
            raise FileNotFoundError(f"Main model not found: {self.config.MAIN_MODEL_PATH}")
        
        print(f"  [1/2] Main detection model: {self.config.MAIN_MODEL_PATH}")
        self.main_model = YOLO(self.config.MAIN_MODEL_PATH)
        
        # Auto-detect segmentation model if not specified
        if self.config.SEG_MODEL_PATH is None:
            print(f"  Auto-detecting {model_type} segmentation model...")
            self.config.SEG_MODEL_PATH = find_best_model("runs/segment", "segmentation", use_latest=use_latest)
            if self.config.SEG_MODEL_PATH is None:
                raise FileNotFoundError("No trained segmentation models found in runs/segment")
        
        # Load segmentation model
        if not os.path.exists(self.config.SEG_MODEL_PATH):
            raise FileNotFoundError(f"Segmentation model not found: {self.config.SEG_MODEL_PATH}")
        
        print(f"  [2/2] Segmentation model: {self.config.SEG_MODEL_PATH}")
        self.seg_model = YOLO(self.config.SEG_MODEL_PATH)
        
        print("  ✓ Models loaded successfully")
    
    def stage1_main_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Stage 1: Run main detection model to get all objects.
        
        Args:
            image: Input image
            
        Returns:
            List of detections with keys: box, confidence, class_id, class_name
        """
        results = self.main_model.predict(
            source=image,
            conf=self.config.MAIN_CONF_THRESHOLD,
            imgsz=self.config.MAIN_IMGSZ,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({
                    'box': [float(x) for x in box],  # Convert to native Python float
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.config.CLASS_NAMES.get(cls, 'unknown'),
                    'stage': 'main_detection'
                })
        
        return detections
    
    def crop_with_margin(self, image: np.ndarray, box: List[float]) -> Tuple[np.ndarray, List[int]]:
        """
        Crop image region around box with margin.
        
        Args:
            image: Input image
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped image and actual crop coordinates [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        
        # Calculate box dimensions
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Add margin
        margin = self.config.CROP_MARGIN * (box_w + box_h) / 2
        
        # Calculate crop coordinates with margin
        crop_x1 = max(0, int(x1 - margin))
        crop_y1 = max(0, int(y1 - margin))
        crop_x2 = min(w, int(x2 + margin))
        crop_y2 = min(h, int(y2 + margin))
        
        # Crop
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return crop, [crop_x1, crop_y1, crop_x2, crop_y2]
    
    def stage2_segmentation_refinement(self, image: np.ndarray, 
                                      main_detections: List[Dict]) -> List[Dict]:
        """
        Stage 2: Refine stamp and signature detections using segmentation model.
        
        Args:
            image: Original full image
            main_detections: Detections from stage 1
            
        Returns:
            Refined detections list
        """
        refined_detections = []
        
        for det in main_detections:
            class_name = det['class_name']
            
            # Only refine stamps and signatures
            if class_name not in ['stamp', 'signature']:
                # Keep QR codes as-is from main model
                refined_detections.append(det)
                continue
            
            # Crop region around detection
            crop, crop_coords = self.crop_with_margin(image, det['box'])
            
            if crop.size == 0:
                # Failed to crop, keep original
                refined_detections.append(det)
                continue
            
            # Run segmentation model on crop
            seg_results = self.seg_model.predict(
                source=crop,
                conf=self.config.SEG_CONF_THRESHOLD,
                imgsz=self.config.SEG_IMGSZ,
                task='segment',
                verbose=False
            )
            
            # Process segmentation results
            if len(seg_results) > 0 and seg_results[0].boxes is not None:
                seg_boxes = seg_results[0].boxes.xyxy.cpu().numpy()
                seg_confs = seg_results[0].boxes.conf.cpu().numpy()
                seg_classes = seg_results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Map segmentation classes: 0=stamp, 1=signature
                seg_class_map = {0: 'stamp', 1: 'signature'}
                
                # Find best matching detection in crop
                for seg_box, seg_conf, seg_cls in zip(seg_boxes, seg_confs, seg_classes):
                    seg_class_name = seg_class_map.get(seg_cls, 'unknown')
                    
                    # Only use if class matches
                    if seg_class_name == class_name:
                        # Convert crop coordinates back to full image coordinates
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                        full_x1 = float(crop_x1 + seg_box[0])
                        full_y1 = float(crop_y1 + seg_box[1])
                        full_x2 = float(crop_x1 + seg_box[2])
                        full_y2 = float(crop_y1 + seg_box[3])
                        
                        refined_detections.append({
                            'box': [full_x1, full_y1, full_x2, full_y2],
                            'confidence': float(seg_conf),
                            'class_id': int(det['class_id']),  # Ensure int
                            'class_name': class_name,
                            'stage': 'segmentation_refined',
                            'original_confidence': float(det['confidence'])
                        })
                        break
                else:
                    # No matching segmentation found, keep original
                    det['stage'] = 'main_detection_fallback'
                    refined_detections.append(det)
            else:
                # No segmentation results, keep original
                det['stage'] = 'main_detection_fallback'
                refined_detections.append(det)
        
        return refined_detections
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        if inter_area == 0:
            return 0.0
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def group_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Group overlapping detections of the same class using connected components.
        
        Args:
            detections: List of detections
            
        Returns:
            Grouped detections list
        """
        if len(detections) <= 1:
            return detections
        
        # Separate by class
        by_class = {}
        for det in detections:
            cls_name = det['class_name']
            if cls_name not in by_class:
                by_class[cls_name] = []
            by_class[cls_name].append(det)
        
        grouped_detections = []
        
        # Process each class separately
        for cls_name, dets in by_class.items():
            # Only group signatures (they tend to have multiple overlapping boxes)
            # Keep stamps and QR codes as-is
            if cls_name != 'signature' or len(dets) <= 1:
                grouped_detections.extend(dets)
                continue
            
            # Build connectivity graph using IoU
            n = len(dets)
            graph = {i: set() for i in range(n)}
            
            for i in range(n):
                for j in range(i + 1, n):
                    iou = self.calculate_iou(dets[i]['box'], dets[j]['box'])
                    if iou >= self.group_iou_threshold:
                        graph[i].add(j)
                        graph[j].add(i)
            
            # Find connected components using DFS
            visited = set()
            components = []
            
            def dfs(node: int, component: set):
                if node in visited:
                    return
                visited.add(node)
                component.add(node)
                for neighbor in graph[node]:
                    dfs(neighbor, component)
            
            for i in range(n):
                if i not in visited:
                    component = set()
                    dfs(i, component)
                    components.append(component)
            
            # Merge detections in each component
            for component in components:
                if len(component) == 1:
                    # Single detection, keep as is
                    idx = list(component)[0]
                    grouped_detections.append(dets[idx])
                else:
                    # Multiple detections, merge them
                    indices = list(component)
                    boxes = [dets[i]['box'] for i in indices]
                    confidences = [dets[i]['confidence'] for i in indices]
                    
                    # Calculate bounding box that fits all
                    x1_min = min(box[0] for box in boxes)
                    y1_min = min(box[1] for box in boxes)
                    x2_max = max(box[2] for box in boxes)
                    y2_max = max(box[3] for box in boxes)
                    
                    # Use maximum confidence
                    max_conf = max(confidences)
                    
                    # Determine stage (prefer 'refined' if any are refined)
                    stages = [dets[i].get('stage', '') for i in indices]
                    if any('refined' in s for s in stages):
                        merged_stage = 'segmentation_refined_grouped'
                    else:
                        merged_stage = 'main_detection_grouped'
                    
                    # Create merged detection
                    merged = {
                        'box': [x1_min, y1_min, x2_max, y2_max],
                        'confidence': max_conf,
                        'class_id': dets[indices[0]]['class_id'],
                        'class_name': cls_name,
                        'stage': merged_stage,
                        'grouped': True,
                        'group_count': len(component)
                    }
                    
                    # Keep original confidence if it was refined
                    if 'original_confidence' in dets[indices[0]]:
                        merged['original_confidence'] = dets[indices[0]]['original_confidence']
                    
                    grouped_detections.append(merged)
        
        return grouped_detections
    
    def load_ground_truth(self, image_path: str) -> List[Dict]:
        """
        Load ground truth labels for an image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            List of ground truth detection dictionaries
        """
        # Get corresponding label file
        image_path = Path(image_path)
        label_name = image_path.stem + '.txt'
        label_path = Path(self.config.VAL_LABELS_DIR) / label_name
        
        ground_truth = []
        
        if not label_path.exists():
            return ground_truth
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            return ground_truth
        
        img_height, img_width = image.shape[:2]
        
        # Parse YOLO format labels
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from normalized YOLO format to pixel coordinates
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                
                gt = {
                    'box': [x1, y1, x2, y2],
                    'class_id': cls_id,
                    'class_name': self.config.CLASS_NAMES.get(cls_id, 'unknown')
                }
                ground_truth.append(gt)
        
        return ground_truth
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict], 
                         iou_threshold: float = 0.5) -> Dict:
        """
        Calculate detection metrics (precision, recall, F1, mAP).
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for matching (default: 0.5)
        
        Returns:
            Dictionary with metrics per class and overall
        """
        from collections import defaultdict
        
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # Initialize counters per class
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0})
        
        # Count ground truth per class
        for gt in ground_truth:
            cls_name = gt['class_name']
            class_stats[cls_name]['gt_count'] += 1
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(ground_truth)
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        for pred in sorted_preds:
            pred_cls = pred['class_name']
            pred_box = pred['box']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                if gt['class_name'] != pred_cls:
                    continue
                
                iou = self.calculate_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                class_stats[pred_cls]['tp'] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive
                class_stats[pred_cls]['fp'] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(ground_truth):
            if not gt_matched[gt_idx]:
                cls_name = gt['class_name']
                class_stats[cls_name]['fn'] += 1
        
        # Calculate metrics per class
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        
        for cls_name, stats in class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['per_class'][cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gt_count': stats['gt_count']
            }
        
        # Calculate overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        }
        
        # Calculate mAP50 and mAP50-95
        map50_values = []
        map50_95_values = []
        
        for cls_name in class_stats.keys():
            # mAP@0.5
            ap50 = self.calculate_ap(predictions, ground_truth, cls_name, iou_threshold=0.5)
            map50_values.append(ap50)
            
            # mAP@0.5:0.95 (average over IoU thresholds 0.5 to 0.95)
            ap_values = []
            for iou_t in np.linspace(0.5, 0.95, 10):
                ap = self.calculate_ap(predictions, ground_truth, cls_name, iou_threshold=iou_t)
                ap_values.append(ap)
            map50_95_values.append(np.mean(ap_values))
            
            # Add to per-class metrics
            metrics['per_class'][cls_name]['ap50'] = ap50
            metrics['per_class'][cls_name]['ap50_95'] = np.mean(ap_values)
        
        # Overall mAP
        overall_map50 = np.mean(map50_values) if map50_values else 0.0
        overall_map50_95 = np.mean(map50_95_values) if map50_95_values else 0.0
        
        metrics['overall']['map50'] = overall_map50
        metrics['overall']['map50_95'] = overall_map50_95
        
        return metrics
    
    def calculate_ap(self, predictions: List[Dict], ground_truth: List[Dict], 
                     cls_name: str, iou_threshold: float = 0.5) -> float:
        """
        Calculate Average Precision (AP) for a single class.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            cls_name: Class name to calculate AP for
            iou_threshold: IoU threshold for matching
        
        Returns:
            Average Precision value
        """
        # Filter predictions and GT for this class
        cls_preds = [p for p in predictions if p['class_name'] == cls_name]
        cls_gt = [g for g in ground_truth if g['class_name'] == cls_name]
        
        if len(cls_gt) == 0:
            return 0.0
        
        if len(cls_preds) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        cls_preds = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(cls_gt)
        
        # For each prediction, check if it matches a GT
        tp = []
        fp = []
        
        for pred in cls_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(cls_gt):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self.calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute precision and recall at each threshold
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(cls_gt)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def _save_metrics_csv(self, metrics: Dict, csv_path: str) -> None:
        """
        Save metrics to CSV file.
        
        Args:
            metrics: Metrics dictionary
            csv_path: Path to save CSV
        """
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'AP@50', 'AP@50-95', 'TP', 'FP', 'FN', 'GT Count'])
            
            # Per-class metrics
            per_class = metrics.get('per_class', {})
            for cls_name, cls_metrics in sorted(per_class.items()):
                writer.writerow([
                    cls_name,
                    f"{cls_metrics['precision']:.4f}",
                    f"{cls_metrics['recall']:.4f}",
                    f"{cls_metrics['f1']:.4f}",
                    f"{cls_metrics.get('ap50', 0):.4f}",
                    f"{cls_metrics.get('ap50_95', 0):.4f}",
                    cls_metrics['tp'],
                    cls_metrics['fp'],
                    cls_metrics['fn'],
                    cls_metrics['gt_count']
                ])
            
            # Overall metrics
            overall = metrics.get('overall', {})
            if overall:
                writer.writerow([])  # Empty row
                writer.writerow([
                    'Overall',
                    f"{overall['precision']:.4f}",
                    f"{overall['recall']:.4f}",
                    f"{overall['f1']:.4f}",
                    f"{overall.get('map50', 0):.4f}",
                    f"{overall.get('map50_95', 0):.4f}",
                    overall['tp'],
                    overall['fp'],
                    overall['fn'],
                    '-'
                ])
    
    def predict_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run ensemble prediction on single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (image, detections)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Stage 1: Main detection
        main_detections = self.stage1_main_detection(image)
        
        # Stage 2: Segmentation refinement
        refined_detections = self.stage2_segmentation_refinement(image, main_detections)
        
        # Stage 3: Group overlapping detections
        grouped_detections = self.group_overlapping_detections(refined_detections)
        
        return image, grouped_detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: str):
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections
            save_path: Path to save visualization
        """
        vis_img = image.copy()
        
        for det in detections:
            box = det['box']
            conf = det['confidence']
            class_name = det['class_name']
            stage = det.get('stage', 'unknown')
            is_grouped = det.get('grouped', False)
            group_count = det.get('group_count', 1)
            
            # Get color
            color = self.config.COLORS.get(class_name, (255, 255, 255))
            
            # Draw box
            x1, y1, x2, y2 = map(int, box)
            
            # Thicker line for refined detections
            thickness = 3 if 'refined' in stage else 2
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{class_name} {conf:.2f}"
            if 'refined' in stage:
                label += " [SEG]"
            if is_grouped and group_count > 1:
                label += f" x{group_count}"
            
            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(vis_img, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_img, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend
        legend_height = 120
        legend = np.ones((legend_height, vis_img.shape[1], 3), dtype=np.uint8) * 255
        
        y_offset = 20
        for class_name, color in self.config.COLORS.items():
            cv2.rectangle(legend, (10, y_offset), (30, y_offset + 20), color, -1)
            cv2.putText(legend, class_name.capitalize(), (40, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 30
        
        cv2.putText(legend, "[SEG] = Segmentation Refined", (10, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Combine
        final_vis = np.vstack([vis_img, legend])
        
        # Save
        cv2.imwrite(save_path, final_vis)
    
    def run_inference(self, use_latest=False):
        """Run ensemble inference on all test images
        
        Args:
            use_latest: If True, load last.pt instead of best.pt
        """
        
        print("\n" + "=" * 80)
        print("ENSEMBLE INFERENCE: Main Detection + Segmentation Refinement")
        print("=" * 80)
        
        # Load models
        self.load_models(use_latest=use_latest)
        
        # Find test images
        print(f"\nFinding test images in: {self.config.TEST_IMAGES_DIR}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        
        for ext in image_extensions:
            test_images.extend(Path(self.config.TEST_IMAGES_DIR).glob(f'*{ext}'))
            test_images.extend(Path(self.config.TEST_IMAGES_DIR).glob(f'*{ext.upper()}'))
        
        test_images = sorted(test_images)
        print(f"Found {len(test_images)} test images")
        
        if len(test_images) == 0:
            print("No images found for testing")
            return
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        if self.config.SAVE_VISUALIZATIONS:
            vis_dir = os.path.join(self.config.OUTPUT_DIR, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        # Process images
        print(f"\nProcessing images...")
        print("-" * 80)
        
        all_detections = {}
        all_ground_truth = {}
        stats = {
            'total_images': 0,
            'total_detections': 0,
            'by_class': {},
            'by_stage': {'main_detection': 0, 'segmentation_refined': 0, 
                        'main_detection_fallback': 0}
        }
        
        for img_path in test_images:
            try:
                # Run prediction
                image, detections = self.predict_image(str(img_path))
                
                # Store results
                img_name = img_path.name
                all_detections[img_name] = detections
                
                # Load ground truth for metrics
                ground_truth = self.load_ground_truth(str(img_path))
                all_ground_truth[img_name] = ground_truth
                
                # Update statistics
                stats['total_images'] += 1
                stats['total_detections'] += len(detections)
                
                for det in detections:
                    class_name = det['class_name']
                    stage = det.get('stage', 'unknown')
                    
                    stats['by_class'][class_name] = stats['by_class'].get(class_name, 0) + 1
                    stats['by_stage'][stage] = stats['by_stage'].get(stage, 0) + 1
                
                # Visualize
                if self.config.SAVE_VISUALIZATIONS:
                    vis_path = os.path.join(vis_dir, f"ensemble_{img_name}")
                    self.visualize_detections(image, detections, vis_path)
                
                # Print progress
                refined_count = sum(1 for d in detections if 'refined' in d.get('stage', ''))
                grouped_count = sum(1 for d in detections if d.get('grouped', False))
                
                status = f"{len(detections)} objects"
                if refined_count > 0:
                    status += f" ({refined_count} refined)"
                if grouped_count > 0:
                    status += f" ({grouped_count} grouped)"
                
                print(f"  ✓ {img_name}: {status}")
                
            except Exception as e:
                print(f"  ✗ {img_path.name}: Error - {str(e)}")
        
        print("-" * 80)
        
        # Calculate metrics across all images
        print("\nCalculating metrics...")
        all_preds = []
        all_gt = []
        for img_name in all_detections.keys():
            all_preds.extend(all_detections[img_name])
            all_gt.extend(all_ground_truth.get(img_name, []))
        
        metrics = self.calculate_metrics(all_preds, all_gt, iou_threshold=0.5)
        stats['metrics'] = metrics
        
        # Save detections JSON
        if self.config.SAVE_DETECTIONS_JSON:
            results_data = {
                'detections': all_detections,
                'ground_truth': all_ground_truth,
                'metrics': metrics
            }
            json_path = os.path.join(self.config.OUTPUT_DIR, 'detections.json')
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n✓ Saved detections: {json_path}")
        
        # Save metrics CSV
        if self.config.SAVE_METRICS_CSV:
            metrics_csv_path = os.path.join(self.config.OUTPUT_DIR, 'metrics.csv')
            self._save_metrics_csv(metrics, metrics_csv_path)
            print(f"✓ Saved metrics: {metrics_csv_path}")
        
        # Save statistics CSV
        if self.config.SAVE_METRICS_CSV:
            self.save_statistics_csv(stats)
        
        # Print summary
        self.print_summary(stats)
    
    def save_statistics_csv(self, stats: Dict):
        """Save inference statistics to CSV"""
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'statistics.csv')
        
        with open(csv_path, 'w') as f:
            f.write("Metric,Value\n")
            f.write(f"Total Images,{stats['total_images']}\n")
            f.write(f"Total Detections,{stats['total_detections']}\n")
            f.write(f"Avg Detections per Image,{stats['total_detections']/max(stats['total_images'],1):.2f}\n")
            f.write("\n")
            f.write("Class,Count\n")
            for cls, count in sorted(stats['by_class'].items()):
                f.write(f"{cls},{count}\n")
            f.write("\n")
            f.write("Stage,Count\n")
            for stage, count in sorted(stats['by_stage'].items()):
                f.write(f"{stage},{count}\n")
        
        print(f"✓ Saved statistics: {csv_path}")
    
    def print_summary(self, stats: Dict):
        """Print inference summary"""
        print("\n" + "=" * 80)
        print("ENSEMBLE INFERENCE COMPLETE")
        print("=" * 80)
        
        print(f"\n📊 Statistics:")
        print(f"  Total images processed: {stats['total_images']}")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Average per image: {stats['total_detections']/max(stats['total_images'],1):.2f}")
        
        print(f"\n📦 Detections by Class:")
        for cls, count in sorted(stats['by_class'].items()):
            pct = 100 * count / max(stats['total_detections'], 1)
            print(f"  {cls:12s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n🔄 Processing Stages:")
        refined = stats['by_stage'].get('segmentation_refined', 0)
        refined_grouped = stats['by_stage'].get('segmentation_refined_grouped', 0)
        fallback = stats['by_stage'].get('main_detection_fallback', 0)
        main_only = stats['by_stage'].get('main_detection', 0)
        main_grouped = stats['by_stage'].get('main_detection_grouped', 0)
        
        total_refined_eligible = refined + refined_grouped + fallback
        total_refined = refined + refined_grouped
        if total_refined_eligible > 0:
            refinement_rate = 100 * total_refined / total_refined_eligible
        else:
            refinement_rate = 0
        
        print(f"  Main detection (QR codes): {main_only}")
        print(f"  Segmentation refined: {refined}")
        print(f"  Segmentation refined (grouped): {refined_grouped}")
        print(f"  Fallback to main: {fallback}")
        print(f"  Main detection (grouped): {main_grouped}")
        print(f"  Refinement success rate: {refinement_rate:.1f}%")
        
        # Print metrics if available
        metrics = stats.get('metrics', {})
        if metrics:
            print(f"\n" + "=" * 80)
            print("VALIDATION METRICS (IoU@0.5)")
            print("=" * 80)
            
            # Per-class metrics
            per_class = metrics.get('per_class', {})
            if per_class:
                print(f"\nPer-Class Metrics:")
                print(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AP50':<10} {'AP50-95':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
                print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*5} {'-'*5} {'-'*5}")
                for cls_name, cls_metrics in sorted(per_class.items()):
                    print(f"  {cls_name:<12} "
                          f"{cls_metrics['precision']:<10.4f} "
                          f"{cls_metrics['recall']:<10.4f} "
                          f"{cls_metrics['f1']:<10.4f} "
                          f"{cls_metrics.get('ap50', 0):<10.4f} "
                          f"{cls_metrics.get('ap50_95', 0):<10.4f} "
                          f"{cls_metrics['tp']:<5} "
                          f"{cls_metrics['fp']:<5} "
                          f"{cls_metrics['fn']:<5}")
            
            # Overall metrics
            overall = metrics.get('overall', {})
            if overall:
                print(f"\nOverall Metrics:")
                print(f"  Precision:  {overall['precision']:.4f}")
                print(f"  Recall:     {overall['recall']:.4f}")
                print(f"  F1-Score:   {overall['f1']:.4f}")
                print(f"  mAP@50:     {overall.get('map50', 0):.4f}")
                print(f"  mAP@50-95:  {overall.get('map50_95', 0):.4f}")
                print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")
        
        print(f"\n📁 Output Directory: {os.path.abspath(self.config.OUTPUT_DIR)}")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble inference with detection + segmentation')
    parser.add_argument('--main-model', type=str, help='Path to main detection model')
    parser.add_argument('--seg-model', type=str, help='Path to segmentation model')
    parser.add_argument('--images', type=str, help='Directory with test images')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--main-conf', type=float, default=0.25, help='Main model confidence threshold')
    parser.add_argument('--seg-conf', type=float, default=0.25, help='Segmentation model confidence threshold')
    parser.add_argument('--crop-margin', type=float, default=0.15, help='Crop margin for refinement')
    parser.add_argument('--latest', action='store_true', help='Use last.pt (latest checkpoint) instead of best.pt')
    
    args = parser.parse_args()
    
    # Create config
    config = EnsembleConfig()
    
    # Override with command line arguments
    if args.main_model:
        config.MAIN_MODEL_PATH = args.main_model
    if args.seg_model:
        config.SEG_MODEL_PATH = args.seg_model
    if args.images:
        config.TEST_IMAGES_DIR = args.images
    if args.output:
        config.OUTPUT_DIR = args.output
    elif args.latest:
        config.OUTPUT_DIR = "outputs/inference/ensemble_latest"
    if args.main_conf:
        config.MAIN_CONF_THRESHOLD = args.main_conf
    if args.seg_conf:
        config.SEG_CONF_THRESHOLD = args.seg_conf
    if args.crop_margin:
        config.CROP_MARGIN = args.crop_margin
    
    # Run inference
    ensemble = EnsembleInference(config)
    ensemble.run_inference(use_latest=args.latest)


if __name__ == "__main__":
    main()
