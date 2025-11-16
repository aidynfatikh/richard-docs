#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite

Tests all trained models (detection and segmentation) against annotated test data.
Evaluates:
- Overall mAP@50, mAP@50-95
- Per-class performance (QR, Signature, Stamp)
- Signature detection rate (critical metric)

Selects top 3 models of each type and generates comprehensive reports.

Usage:
    python3 test_all_models.py
"""

import os
import sys
import cv2
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

class TestConfig:
    """Testing configuration"""
    
    # Test data (annotated images with ground truth labels)
    TEST_IMAGES_DIR = "data/raw/test_images"
    
    # Model directories
    DETECTION_RUNS_DIR = "runs/train"
    SEGMENTATION_RUNS_DIR = "runs/segment"
    
    # Output
    TESTING_DIR = "outputs/testing/results"
    RESULTS_FILE = "test_results.csv"
    SUMMARY_FILE = "test_summary.json"
    
    # Top models to test (None = test all)
    TOP_N_MODELS = None  # Test ALL models
    
    # Class mapping
    CLASS_NAMES = {0: 'qr', 1: 'signature', 2: 'stamp'}
    
    # Detection thresholds
    CONF_THRESHOLD = 0.10  # Low confidence threshold to show all predictions (was 0.25)
    IOU_THRESHOLD = 0.5    # IoU threshold for matching predictions to ground truth


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_ground_truth(image_path: Path, label_path: Path) -> List[Dict]:
    """
    Load ground truth annotations from YOLO format label file.
    
    Returns:
        List of dicts with keys: class_id, box [x1, y1, x2, y2]
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []  # Empty label file = no objects
    
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert to absolute coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            annotations.append({
                'class_id': class_id,
                'box': [x1, y1, x2, y2]
            })
    
    return annotations


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(predictions: List[Dict], 
                           ground_truth: List[Dict],
                           iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth annotations.
    
    Returns:
        (true_positives, false_positives, false_negatives)
        Each is a list of dicts with relevant info
    """
    true_positives = []
    false_positives = []
    false_negatives = []
    
    matched_gt = set()
    
    # Sort predictions by confidence (highest first)
    predictions = sorted(predictions, key=lambda x: x.get('confidence', 1.0), reverse=True)
    
    # Match predictions to ground truth
    for pred in predictions:
        pred_class = pred['class_id']
        pred_box = pred['box']
        pred_conf = pred.get('confidence', 1.0)
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            if gt['class_id'] != pred_class:
                continue
            
            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold:
            true_positives.append({
                'class_id': pred_class,
                'confidence': pred_conf,
                'iou': best_iou,
                'pred_box': pred_box,
                'gt_box': ground_truth[best_gt_idx]['box']
            })
            matched_gt.add(best_gt_idx)
        else:
            false_positives.append({
                'class_id': pred_class,
                'confidence': pred_conf,
                'box': pred_box
            })
    
    # Unmatched ground truth = false negatives
    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in matched_gt:
            false_negatives.append({
                'class_id': gt['class_id'],
                'box': gt['box']
            })
    
    return true_positives, false_positives, false_negatives


def calculate_metrics(all_tp: List, all_fp: List, all_fn: List, 
                     num_classes: int = 3) -> Dict:
    """
    Calculate precision, recall, mAP for all classes.
    
    Returns dict with overall and per-class metrics.
    """
    metrics = {
        'overall': {},
        'per_class': {}
    }
    
    # Per-class metrics
    for class_id in range(num_classes):
        tp = [x for x in all_tp if x['class_id'] == class_id]
        fp = [x for x in all_fp if x['class_id'] == class_id]
        fn = [x for x in all_fn if x['class_id'] == class_id]
        
        n_tp = len(tp)
        n_fp = len(fp)
        n_fn = len(fn)
        
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate AP@50 (simple version)
        ap50 = precision * recall if recall > 0 else 0.0
        
        metrics['per_class'][class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap50': ap50,
            'tp': n_tp,
            'fp': n_fp,
            'fn': n_fn
        }
    
    # Overall metrics
    total_tp = len(all_tp)
    total_fp = len(all_fp)
    total_fn = len(all_fn)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # mAP@50 (average of per-class AP@50)
    map50 = np.mean([m['ap50'] for m in metrics['per_class'].values()])
    
    metrics['overall'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    return metrics


def visualize_results(image_path: Path, 
                     ground_truth: List[Dict],
                     predictions: List[Dict],
                     output_path: Path,
                     tp: List, fp: List, fn: List):
    """Create side-by-side visualization: predictions (left) and ground truth (right)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # Create two copies for side-by-side display
    img_pred = img.copy()
    img_gt = img.copy()
    
    # Left side: Predictions (all predictions from model)
    # Color by class: QR=blue, Signature=green, Stamp=red
    class_colors = {
        0: (255, 0, 0),    # QR: Blue
        1: (0, 255, 0),    # Signature: Green
        2: (0, 0, 255)     # Stamp: Red
    }
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['box']
        class_id = pred['class_id']
        conf = pred['confidence']
        color = class_colors.get(class_id, (255, 255, 255))
        
        cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
        label = TestConfig.CLASS_NAMES.get(class_id, str(class_id))
        cv2.putText(img_pred, f"{label} {conf:.2f}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Right side: Ground truth
    for gt in ground_truth:
        x1, y1, x2, y2 = gt['box']
        class_id = gt['class_id']
        color = class_colors.get(class_id, (255, 255, 255))
        
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
        label = TestConfig.CLASS_NAMES.get(class_id, str(class_id))
        cv2.putText(img_gt, f"GT: {label}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add titles
    title_pred = np.zeros((40, w, 3), dtype=np.uint8)
    title_gt = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(title_pred, "Predictions", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(title_gt, "Ground Truth", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine title and image for each side
    img_pred_titled = np.vstack([title_pred, img_pred])
    img_gt_titled = np.vstack([title_gt, img_gt])
    
    # Combine side by side
    combined = np.hstack([img_pred_titled, img_gt_titled])
    
    cv2.imwrite(str(output_path), combined)


# ============================================================================
# MODEL FINDER AND SELECTOR
# ============================================================================

def find_models_by_type(runs_dir: str, model_type: str) -> List[Dict]:
    """
    Find all models of a specific type and extract their training metrics.
    
    Args:
        runs_dir: Directory containing training runs
        model_type: 'detection', 'signature', or 'segmentation'
    
    Returns:
        List of dicts with model info: path, mAP, name, etc.
    """
    if not os.path.exists(runs_dir):
        return []
    
    models = []
    
    for run_dir in Path(runs_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        # Filter by model type
        is_signature = run_dir.name.startswith('signature_only')
        is_segmentation = 'seg' in run_dir.name
        
        if model_type == 'signature' and not is_signature:
            continue
        elif model_type == 'detection' and (is_signature or is_segmentation):
            continue
        elif model_type == 'segmentation' and not is_segmentation:
            continue
        
        # Check for weights
        best_weights = run_dir / "weights" / "best.pt"
        if not best_weights.exists():
            continue
        
        # Try to load training metrics
        metrics_file = run_dir / "training_metrics.csv"
        map_score = 0.0
        
        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                if model_type == 'segmentation' and 'mask_mAP50-95' in df.columns:
                    map_score = df['mask_mAP50-95'].max()
                elif 'mAP50-95' in df.columns:
                    map_score = df['mAP50-95'].max()
                elif len(df.columns) > 6:
                    map_score = df.iloc[:, 6].max()
            except Exception:
                pass
        
        models.append({
            'name': run_dir.name,
            'path': str(best_weights),
            'map': map_score,
            'type': model_type
        })
    
    # Sort by mAP (highest first)
    models = sorted(models, key=lambda x: x['map'], reverse=True)
    
    return models


def select_top_models(config: TestConfig) -> Dict[str, List[Dict]]:
    """Select models to test (detection and ensemble)"""
    
    print("=" * 80)
    print("SCANNING FOR TRAINED MODELS")
    print("=" * 80)
    
    selected = {}
    
    # Detection models (main YOLOv11s) - TEST ALL
    detection_models = find_models_by_type(config.DETECTION_RUNS_DIR, 'detection')
    selected['detection'] = detection_models  # Test ALL detection models
    print(f"\nDetection Models (YOLOv11s):")
    print(f"  Found: {len(detection_models)} models")
    print(f"  Testing ALL {len(selected['detection'])} models:")
    for i, model in enumerate(selected['detection'], 1):
        print(f"    {i}. {model['name']} (mAP: {model['map']:.4f})")
    
    # Ensemble models (detection + segmentation)
    # We'll create ensemble tests by pairing each detection model with TOP 2 segmentation models
    segmentation_models = find_models_by_type(config.SEGMENTATION_RUNS_DIR, 'segmentation')
    top_seg_models = segmentation_models[:2]  # Only top 2 segmentation models
    selected['segmentation'] = top_seg_models
    
    print(f"\nSegmentation Models (for ensemble):")
    print(f"  Found: {len(segmentation_models)} models")
    print(f"  Using TOP 2 for ensemble pairs:")
    for i, model in enumerate(top_seg_models, 1):
        print(f"    {i}. {model['name']} (mAP: {model['map']:.4f})")
    
    # Create ensemble pairs (all detection models × top 2 segmentation models)
    ensemble_pairs = []
    for det_model in detection_models:
        for seg_model in top_seg_models:
            ensemble_pairs.append({
                'name': f"ensemble_{det_model['name']}__{seg_model['name']}",
                'detection_model': det_model,
                'segmentation_model': seg_model,
                'type': 'ensemble'
            })
    
    selected['ensemble'] = ensemble_pairs
    print(f"\nEnsemble Combinations:")
    print(f"  Testing {len(ensemble_pairs)} ensemble pairs ({len(detection_models)} detection × 2 segmentation)")
    
    print("=" * 80)
    
    return selected


# ============================================================================
# MODEL TESTING
# ============================================================================

def test_detection_model(model_path: str, 
                        test_images_dir: Path,
                        output_dir: Path,
                        conf_threshold: float = 0.25) -> Dict:
    """Test a detection model on all test images"""
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(model_path)
    
    # Get all test images
    image_files = sorted(list(test_images_dir.glob("*.png")) + 
                        list(test_images_dir.glob("*.jpg")))
    
    all_tp = []
    all_fp = []
    all_fn = []
    
    # Create visualization directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each image
    for img_path in image_files:
        label_path = img_path.with_suffix('.txt')
        
        # Load ground truth
        gt = load_ground_truth(img_path, label_path)
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Extract predictions
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                predictions.append({
                    'class_id': cls,
                    'confidence': conf,
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Match predictions to ground truth
        tp, fp, fn = match_predictions_to_gt(
            predictions, gt, iou_threshold=TestConfig.IOU_THRESHOLD
        )
        
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)
        
        # Visualize
        vis_path = vis_dir / img_path.name
        visualize_results(img_path, gt, predictions, vis_path, tp, fp, fn)
    
    # Calculate metrics
    metrics = calculate_metrics(all_tp, all_fp, all_fn)
    
    return metrics


def test_ensemble_model(detection_model_path: str,
                       segmentation_model_path: str,
                       test_images_dir: Path,
                       output_dir: Path,
                       conf_threshold: float = 0.25,
                       crop_margin: float = 0.30) -> Dict:
    """Test ensemble model (detection + segmentation refinement)"""
    
    from ultralytics import YOLO
    
    # Load models
    detection_model = YOLO(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
    
    # Get all test images
    image_files = sorted(list(test_images_dir.glob("*.png")) + 
                        list(test_images_dir.glob("*.jpg")))
    
    all_tp = []
    all_fp = []
    all_fn = []
    
    # Create visualization directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each image
    for img_path in image_files:
        label_path = img_path.with_suffix('.txt')
        
        # Load ground truth
        gt = load_ground_truth(img_path, label_path)
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Step 1: Run detection model
        det_results = detection_model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Extract initial detections
        initial_detections = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                initial_detections.append({
                    'class_id': cls,
                    'confidence': conf,
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Step 2: Refine stamps and signatures with segmentation
        refined_predictions = []
        
        for det in initial_detections:
            class_id = det['class_id']
            x1, y1, x2, y2 = det['box']
            
            # Only refine stamps (class 2) and signatures (class 1)
            if class_id in [1, 2]:
                # Crop with margin
                crop_w = x2 - x1
                crop_h = y2 - y1
                margin_x = int(crop_w * crop_margin)
                margin_y = int(crop_h * crop_margin)
                
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(w, x2 + margin_x)
                crop_y2 = min(h, y2 + margin_y)
                
                crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Run segmentation
                seg_results = segmentation_model(crop, conf=conf_threshold, verbose=False)
                
                # Extract refined boxes from segmentation
                if len(seg_results) > 0 and seg_results[0].boxes is not None:
                    seg_boxes = seg_results[0].boxes
                    for j in range(len(seg_boxes)):
                        seg_x1, seg_y1, seg_x2, seg_y2 = seg_boxes.xyxy[j].cpu().numpy()
                        seg_conf = float(seg_boxes.conf[j].cpu().numpy())
                        seg_cls = int(seg_boxes.cls[j].cpu().numpy())
                        
                        # Convert to full image coordinates
                        full_x1 = int(crop_x1 + seg_x1)
                        full_y1 = int(crop_y1 + seg_y1)
                        full_x2 = int(crop_x1 + seg_x2)
                        full_y2 = int(crop_y1 + seg_y2)
                        
                        refined_predictions.append({
                            'class_id': seg_cls,
                            'confidence': seg_conf,
                            'box': [full_x1, full_y1, full_x2, full_y2]
                        })
                else:
                    # No segmentation found, keep original detection
                    refined_predictions.append(det)
            else:
                # QR codes - keep as is
                refined_predictions.append(det)
        
        # Match predictions to ground truth
        tp, fp, fn = match_predictions_to_gt(
            refined_predictions, gt, iou_threshold=TestConfig.IOU_THRESHOLD
        )
        
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)
        
        # Visualize
        vis_path = vis_dir / img_path.name
        visualize_results(img_path, gt, refined_predictions, vis_path, tp, fp, fn)
    
    # Calculate metrics
    metrics = calculate_metrics(all_tp, all_fp, all_fn)
    
    return metrics


# ============================================================================
# MAIN TESTING PIPELINE
# ============================================================================

def main():
    """Main testing pipeline"""
    
    config = TestConfig()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("=" * 80)
    print(f"\nTest Data: {config.TEST_IMAGES_DIR}")
    print(f"Output: {config.TESTING_DIR}")
    print(f"Testing ALL detection models and ALL ensemble combinations")
    
    # Create output directory
    testing_dir = Path(config.TESTING_DIR)
    if testing_dir.exists():
        print(f"\n⚠️  Clearing existing TESTING directory...")
        shutil.rmtree(testing_dir)
    testing_dir.mkdir(parents=True)
    
    # Check test data
    test_images_dir = Path(config.TEST_IMAGES_DIR)
    if not test_images_dir.exists():
        print(f"\n❌ Test images directory not found: {config.TEST_IMAGES_DIR}")
        sys.exit(1)
    
    test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
    print(f"Found {len(test_images)} test images")
    
    # Select models to test
    selected_models = select_top_models(config)
    
    # Storage for all results
    all_results = []
    
    # Test detection models
    print("\n" + "=" * 80)
    print("TESTING DETECTION MODELS")
    print("=" * 80)
    
    for i, model_info in enumerate(selected_models['detection'], 1):
        print(f"\n[{i}/{len(selected_models['detection'])}] Testing: {model_info['name']}")
        
        output_dir = testing_dir / "detection" / model_info['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            metrics = test_detection_model(
                model_info['path'],
                test_images_dir,
                output_dir,
                config.CONF_THRESHOLD
            )
            
            # Add to results
            result = {
                'model_name': model_info['name'],
                'model_type': 'detection',
                'training_map': model_info['map'],
                'test_map50': metrics['overall']['map50'],
                'test_precision': metrics['overall']['precision'],
                'test_recall': metrics['overall']['recall'],
                'test_f1': metrics['overall']['f1'],
                'signature_precision': metrics['per_class'][1]['precision'],
                'signature_recall': metrics['per_class'][1]['recall'],
                'signature_detection_rate': metrics['per_class'][1]['recall'],  # Key metric
                'qr_precision': metrics['per_class'][0]['precision'],
                'qr_recall': metrics['per_class'][0]['recall'],
                'stamp_precision': metrics['per_class'][2]['precision'],
                'stamp_recall': metrics['per_class'][2]['recall'],
                'output_dir': str(output_dir)
            }
            all_results.append(result)
            
            # Print summary
            print(f"  Overall mAP@50: {metrics['overall']['map50']:.4f}")
            print(f"  Signature Detection Rate: {metrics['per_class'][1]['recall']:.4f}")
            print(f"  Precision: {metrics['overall']['precision']:.4f}")
            print(f"  Recall: {metrics['overall']['recall']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error testing model: {e}")
            continue
    
    # Test ensemble models (detection + segmentation)
    print("\n" + "=" * 80)
    print("TESTING ENSEMBLE MODELS")
    print("=" * 80)
    
    for i, ensemble_info in enumerate(selected_models['ensemble'], 1):
        print(f"\n[{i}/{len(selected_models['ensemble'])}] Testing: {ensemble_info['name']}")
        
        output_dir = testing_dir / "ensemble" / ensemble_info['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            metrics = test_ensemble_model(
                ensemble_info['detection_model']['path'],
                ensemble_info['segmentation_model']['path'],
                test_images_dir,
                output_dir,
                config.CONF_THRESHOLD
            )
            
            result = {
                'model_name': ensemble_info['name'],
                'model_type': 'ensemble',
                'detection_model': ensemble_info['detection_model']['name'],
                'segmentation_model': ensemble_info['segmentation_model']['name'],
                'training_map': f"det:{ensemble_info['detection_model']['map']:.3f} seg:{ensemble_info['segmentation_model']['map']:.3f}",
                'test_map50': metrics['overall']['map50'],
                'test_precision': metrics['overall']['precision'],
                'test_recall': metrics['overall']['recall'],
                'test_f1': metrics['overall']['f1'],
                'signature_precision': metrics['per_class'][1]['precision'],
                'signature_recall': metrics['per_class'][1]['recall'],
                'signature_detection_rate': metrics['per_class'][1]['recall'],
                'qr_precision': metrics['per_class'][0]['precision'],
                'qr_recall': metrics['per_class'][0]['recall'],
                'stamp_precision': metrics['per_class'][2]['precision'],
                'stamp_recall': metrics['per_class'][2]['recall'],
                'output_dir': str(output_dir)
            }
            all_results.append(result)
            
            print(f"  Overall mAP@50: {metrics['overall']['map50']:.4f}")
            print(f"  Signature Detection Rate: {metrics['per_class'][1]['recall']:.4f}")
            print(f"  Precision: {metrics['overall']['precision']:.4f}")
            print(f"  Recall: {metrics['overall']['recall']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error testing ensemble: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = testing_dir / config.RESULTS_FILE
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_images_count': len(test_images),
        'models_tested': len(all_results),
        'results': all_results
    }
    json_path = testing_dir / config.SUMMARY_FILE
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {json_path}")
    
    # Print rankings
    print("\n" + "=" * 80)
    print("TOP MODELS BY mAP@50")
    print("=" * 80)
    
    df_sorted = df.sort_values('test_map50', ascending=False)
    print("\n" + df_sorted[['model_name', 'model_type', 'test_map50', 'signature_detection_rate']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP MODELS BY SIGNATURE DETECTION RATE")
    print("=" * 80)
    
    df_sorted = df.sort_values('signature_detection_rate', ascending=False)
    print("\n" + df_sorted[['model_name', 'model_type', 'signature_detection_rate', 'test_map50']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {testing_dir.absolute()}")
    print(f"  - test_results.csv: Detailed metrics for all models")
    print(f"  - test_summary.json: Complete test summary")
    print(f"  - detection/: Visualizations for detection models")
    print(f"  - ensemble/: Visualizations for ensemble models")
    print("=" * 80)


if __name__ == "__main__":
    main()
