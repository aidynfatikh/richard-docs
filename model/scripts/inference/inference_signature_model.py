#!/usr/bin/env python3
"""
Signature Model Inference for Document Analysis
Runs the signature-only YOLOv11 model on validation images and saves visualizations.

Features:
- Uses signature-only detection model
- Processes all images from validation set
- Saves visualizations with bounding boxes and labels
- Generates metrics and detection summary
- Same output format as main model inference
"""

import os
import cv2
import torch
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Set
import json
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

def find_best_signature_model(base_dir="runs/train", use_latest=False):
    """Find the best trained signature model by highest mAP or most recent timestamp
    
    Args:
        base_dir: Directory containing training runs
        use_latest: If True, select most recent run and load last.pt; else select best mAP and load best.pt
    """
    if not os.path.exists(base_dir):
        return None
    
    weight_name = "last.pt" if use_latest else "best.pt"
    
    if use_latest:
        # Select most recent signature_only run by directory modification time
        latest_model_path = None
        latest_time = 0
        latest_run = None
        
        for run_dir in Path(base_dir).iterdir():
            if not run_dir.is_dir():
                continue
            
            if not run_dir.name.startswith('signature_only'):
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
            print(f"  Auto-selected latest signature model: {latest_run}")
        
        return latest_model_path
    
    # Select best model by mAP
    best_model_path = None
    best_map = -1
    best_run = None
    
    for run_dir in Path(base_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        # Only look for signature_only runs
        if not run_dir.name.startswith('signature_only'):
            continue
        
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
                        # Look for mAP50-95 column
                        if 'mAP50-95' in rows[0]:
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
        model_type = "latest" if use_latest else "best"
        if best_map > 0:
            print(f"  Auto-selected {model_type} signature model: {best_run} (mAP: {best_map:.4f})")
        else:
            print(f"  Auto-selected {model_type} signature model: {best_run} (no metrics available)")
    
    return best_model_path


class SignatureInferenceConfig:
    """Configuration for signature model inference"""
    
    # Model path (will be auto-detected if None)
    SIGNATURE_MODEL = None
    
    # Dataset
    VAL_IMAGES = "signature_dataset/images/val"
    VAL_LABELS = "signature_dataset/labels/val"
    
    # Detection parameters
    CONF_THRESHOLD = 0.25  # Confidence threshold
    IOU_THRESHOLD = 0.45   # NMS IoU threshold
    IMGSZ = 1024           # Image size for inference (should match training)
    
    # Grouping parameters
    GROUP_IOU_THRESHOLD = 0.3  # IoU threshold for grouping signatures
    ENABLE_GROUPING = True      # Enable signature grouping
    
    # Output
    OUTPUT_DIR = "outputs/inference/signature"
    SAVE_JSON = True       # Save detection results as JSON
    
    # Visualization
    COLOR = (0, 255, 0)      # Green for signatures
    GT_COLOR = (0, 180, 0)   # Darker green for ground truth
    LINE_THICKNESS = 2
    FONT_SCALE = 0.6
    
    # Device
    DEVICE = 'cpu'  # or 'cuda' for GPU


# ============================================================================
# MAIN INFERENCE CLASS
# ============================================================================

class SignatureModelInference:
    """Signature model inference handler"""
    
    def __init__(self, config: SignatureInferenceConfig, use_latest=False):
        """
        Initialize inference with signature model.
        
        Args:
            config: SignatureInferenceConfig object
            use_latest: If True, use last.pt instead of best.pt
        """
        self.config = config
        
        print("=" * 80)
        print("SIGNATURE MODEL INFERENCE: YOLOv11 Signature Detection")
        print("=" * 80)
        print(f"Device: {config.DEVICE}")
        print(f"Confidence threshold: {config.CONF_THRESHOLD}")
        
        # Auto-detect model if not specified
        if config.SIGNATURE_MODEL is None:
            model_type = "latest" if use_latest else "best"
            print(f"\nAuto-detecting {model_type} signature model...")
            config.SIGNATURE_MODEL = find_best_signature_model(use_latest=use_latest)
            if config.SIGNATURE_MODEL is None:
                raise FileNotFoundError("No trained signature models found in runs/train/signature_only_*")
        
        print(f"Model: {config.SIGNATURE_MODEL}")
        
        # Check if model exists
        if not os.path.exists(config.SIGNATURE_MODEL):
            raise FileNotFoundError(f"Model not found: {config.SIGNATURE_MODEL}")
        
        # Load model
        print("\nLoading model...")
        self.model = YOLO(config.SIGNATURE_MODEL)
        self.model.to(config.DEVICE)
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"Classes: {self.class_names}")
        print("✓ Model loaded successfully")
        
        # Create output directories
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        vis_dir = os.path.join(config.OUTPUT_DIR, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\nOutput directory: {config.OUTPUT_DIR}")
        print(f"Visualizations: {vis_dir}")
    
    def predict(self, image_path: str) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            List of detection dictionaries
        """
        # Run prediction
        results = self.model.predict(
            image_path,
            conf=self.config.CONF_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.config.DEVICE,
            imgsz=self.config.IMGSZ,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = self.class_names[cls]
                
                detection = {
                    'box': xyxy,  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': cls_name
                }
                detections.append(detection)
        
        return detections
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def is_box_inside(self, box1: List[float], box2: List[float], threshold: float = 0.95) -> bool:
        """Check if box1 is completely inside box2."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return False
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        return (inter_area / box1_area) >= threshold if box1_area > 0 else False
    
    def remove_contained_boxes(self, detections: List[Dict]) -> List[Dict]:
        """Remove boxes that are completely inside another box."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Track which boxes to keep
        keep = [True] * len(dets)
        
        # Check each pair
        for i in range(len(dets)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(dets)):
                if not keep[j]:
                    continue
                
                # Check if box j is inside box i
                if self.is_box_inside(dets[j]['box'], dets[i]['box']):
                    keep[j] = False
                # Check if box i is inside box j
                elif self.is_box_inside(dets[i]['box'], dets[j]['box']):
                    keep[i] = False
                    break
        
        # Keep only non-contained boxes
        filtered = [dets[i] for i, should_keep in enumerate(keep) if should_keep]
        return filtered
    
    def group_detections(self, detections: List[Dict]) -> List[Dict]:
        """Group closely stacked signature detections."""
        if not self.config.ENABLE_GROUPING or not detections or len(detections) <= 1:
            return detections
        
        # Build connectivity graph
        n = len(detections)
        graph = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self.calculate_iou(detections[i]['box'], detections[j]['box'])
                if iou >= self.config.GROUP_IOU_THRESHOLD:
                    graph[i].add(j)
                    graph[j].add(i)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node: int, component: Set[int]):
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
        grouped_detections = []
        for component in components:
            if len(component) == 1:
                idx = list(component)[0]
                grouped_detections.append(detections[idx])
            else:
                # Merge multiple detections
                indices = list(component)
                boxes = [detections[i]['box'] for i in indices]
                confidences = [detections[i]['confidence'] for i in indices]
                
                # Calculate bounding box that fits all
                x1_min = min(box[0] for box in boxes)
                y1_min = min(box[1] for box in boxes)
                x2_max = max(box[2] for box in boxes)
                y2_max = max(box[3] for box in boxes)
                
                # Use maximum confidence
                max_conf = max(confidences)
                
                # Create merged detection
                merged = {
                    'box': [x1_min, y1_min, x2_max, y2_max],
                    'confidence': max_conf,
                    'class_id': detections[indices[0]]['class_id'],
                    'class_name': detections[indices[0]]['class_name'],
                    'grouped': True,
                    'group_count': len(component)
                }
                grouped_detections.append(merged)
        
        return grouped_detections
    
    def calculate_ap(self, predictions: List[Dict], ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
        """Calculate Average Precision (AP)."""
        if len(ground_truth) == 0:
            return 0.0
        
        if len(predictions) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(ground_truth)
        
        # For each prediction, check if it matches a GT
        tp = []
        fp = []
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
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
        
        recalls = tp_cumsum / len(ground_truth)
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
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate detection metrics (precision, recall, F1, mAP)."""
        metrics = {}
        
        # Initialize counters
        tp = 0
        fp = 0
        fn = 0
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(ground_truth)
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        for pred in sorted_preds:
            pred_box = pred['box']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self.calculate_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= 0.5 and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
        
        # Count false negatives (unmatched ground truth)
        fn = sum(1 for matched in gt_matched if not matched)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate mAP50 and mAP50-95
        ap50 = self.calculate_ap(predictions, ground_truth, iou_threshold=0.5)
        
        # mAP@0.5:0.95
        ap_values = []
        for iou_t in np.linspace(0.5, 0.95, 10):
            ap = self.calculate_ap(predictions, ground_truth, iou_threshold=iou_t)
            ap_values.append(ap)
        ap50_95 = np.mean(ap_values)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ap50': ap50,
            'ap50_95': ap50_95,
            'gt_count': len(ground_truth)
        }
        
        return metrics
    
    def _save_metrics_csv(self, metrics: Dict, csv_path: str) -> None:
        """Save metrics to CSV file."""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'AP@50', 'AP@50-95', 'TP', 'FP', 'FN', 'GT Count'])
            
            # Signature metrics
            writer.writerow([
                'signature',
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['ap50']:.4f}",
                f"{metrics['ap50_95']:.4f}",
                metrics['tp'],
                metrics['fp'],
                metrics['fn'],
                metrics['gt_count']
            ])
    
    def load_ground_truth(self, image_path: str) -> List[Dict]:
        """Load ground truth labels for an image."""
        # Get corresponding label file
        image_path = Path(image_path)
        label_name = image_path.stem + '.txt'
        label_path = Path(self.config.VAL_LABELS) / label_name
        
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
                    'class_name': self.class_names[cls_id]
                }
                ground_truth.append(gt)
        
        return ground_truth
    
    def draw_boxes(self, image: np.ndarray, detections: List[Dict], 
                   is_ground_truth: bool = False) -> np.ndarray:
        """Draw bounding boxes on image."""
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            
            # Get color
            if is_ground_truth:
                color = self.config.GT_COLOR
                label = "GT: signature"
            else:
                color = self.config.COLOR
                conf = det.get('confidence', 1.0)
                label = f"signature {conf:.2f}"
                
                # Add grouping indicator
                if det.get('grouped', False):
                    group_count = det.get('group_count', 2)
                    label += f" [G:{group_count}]"
            
            # Draw bounding box
            thickness = self.config.LINE_THICKNESS
            if is_ground_truth:
                # Dashed line for ground truth
                self._draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.FONT_SCALE, 2
            )
            
            # Draw label background
            label_y1 = max(text_height + 10, y1)
            cv2.rectangle(
                img, 
                (x1, label_y1 - text_height - 10),
                (x1 + text_width, label_y1),
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img, label, (x1, label_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.FONT_SCALE,
                (255, 255, 255), 
                2
            )
        
        return img
    
    def _draw_dashed_rect(self, img: np.ndarray, pt1: Tuple[int, int], 
                         pt2: Tuple[int, int], color: Tuple[int, int, int], 
                         thickness: int, dash_length: int = 10):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def visualize(self, image_path: str, detections: List[Dict], 
                  output_path: str) -> None:
        """Visualize detections side-by-side with ground truth and save."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Load ground truth
        ground_truth = self.load_ground_truth(image_path)
        
        # Draw predictions on left image
        pred_img = self.draw_boxes(image, detections, is_ground_truth=False)
        
        # Draw ground truth on right image
        gt_img = self.draw_boxes(image, ground_truth, is_ground_truth=True)
        
        # Add titles
        img_height, img_width = image.shape[:2]
        title_height = 50
        
        # Create title bars
        pred_title = np.ones((title_height, img_width, 3), dtype=np.uint8) * 50
        gt_title = np.ones((title_height, img_width, 3), dtype=np.uint8) * 50
        
        # Add text to titles
        pred_text = f"Predictions ({len(detections)} detections)"
        gt_text = f"Ground Truth ({len(ground_truth)} boxes)"
        
        cv2.putText(pred_title, pred_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(gt_title, gt_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Stack titles with images
        pred_with_title = np.vstack([pred_title, pred_img])
        gt_with_title = np.vstack([gt_title, gt_img])
        
        # Create separator
        separator = np.ones((pred_with_title.shape[0], 5, 3), dtype=np.uint8) * 255
        
        # Combine side by side
        combined = np.hstack([pred_with_title, separator, gt_with_title])
        
        # Save visualization
        cv2.imwrite(output_path, combined)
    
    def process_directory(self, image_dir: str) -> Dict:
        """Process all images in a directory."""
        # Find all images
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        image_files = sorted(list(image_dir.glob('*.jpg')) + 
                           list(image_dir.glob('*.jpeg')) + 
                           list(image_dir.glob('*.png')))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nProcessing images...")
        print("-" * 80)
        
        # Statistics
        all_detections = {}
        all_ground_truth = {}
        total_detections = 0
        total_time = 0
        
        vis_dir = os.path.join(self.config.OUTPUT_DIR, 'visualizations')
        
        # Process each image
        for idx, img_path in enumerate(image_files, 1):
            start_time = time.time()
            
            print(f"[{idx}/{len(image_files)}] {img_path.name}...", end=' ')
            
            # Run inference
            detections = self.predict(str(img_path))
            
            # Group detections
            original_count = len(detections)
            detections = self.group_detections(detections)
            after_grouping = len(detections)
            grouped_count = original_count - after_grouping
            
            # Remove contained boxes
            detections = self.remove_contained_boxes(detections)
            removed_count = after_grouping - len(detections)
            
            if grouped_count > 0 or removed_count > 0:
                msg = f"({original_count}"
                if grouped_count > 0:
                    msg += f" → {after_grouping} grouped"
                if removed_count > 0:
                    msg += f" → {len(detections)} removed {removed_count}"
                msg += ") "
                print(msg, end='')
            
            # Load ground truth for metrics
            ground_truth = self.load_ground_truth(str(img_path))
            
            # Save visualizations
            output_path = os.path.join(vis_dir, img_path.name)
            self.visualize(str(img_path), detections, output_path)
            
            # Update statistics
            all_detections[img_path.name] = detections
            all_ground_truth[img_path.name] = ground_truth
            total_detections += len(detections)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"{len(detections)} detections ({elapsed:.2f}s)")
        
        print("-" * 80)
        
        # Calculate metrics across all images
        all_preds = []
        all_gt = []
        for img_name in all_detections.keys():
            all_preds.extend(all_detections[img_name])
            all_gt.extend(all_ground_truth.get(img_name, []))
        
        metrics = self.calculate_metrics(all_preds, all_gt)
        
        # Save detections as JSON if enabled
        if self.config.SAVE_JSON:
            results_data = {
                'detections': all_detections,
                'ground_truth': all_ground_truth,
                'metrics': metrics
            }
            json_path = os.path.join(self.config.OUTPUT_DIR, 'detections.json')
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n✓ Saved detections to {json_path}")
        
        # Compute summary statistics
        summary = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files),
            'total_time': total_time,
            'avg_time_per_image': total_time / len(image_files),
            'metrics': metrics
        }
        
        return summary
    
    def print_summary(self, summary: Dict) -> None:
        """Print inference summary."""
        print(f"\n{'=' * 80}")
        print("SIGNATURE MODEL INFERENCE COMPLETE")
        print(f"{'=' * 80}")
        print(f"\n📊 Statistics:")
        print(f"  Total images processed: {summary['total_images']}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Average per image: {summary['avg_detections_per_image']:.2f}")
        
        # Print metrics
        metrics = summary.get('metrics', {})
        if metrics:
            print(f"\n{'=' * 70}")
            print("Validation Metrics")
            print(f"{'=' * 70}")
            
            print(f"\nSignature Detection Metrics:")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1-Score:   {metrics['f1']:.4f}")
            print(f"  AP@50:      {metrics['ap50']:.4f}")
            print(f"  AP@50-95:   {metrics['ap50_95']:.4f}")
            print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
            print(f"  Ground Truth Count: {metrics['gt_count']}")
            
            # Save metrics to CSV
            csv_path = os.path.join(self.config.OUTPUT_DIR, 'metrics.csv')
            self._save_metrics_csv(metrics, csv_path)
            print(f"\n✓ Metrics saved to: {csv_path}")
        
        print(f"\n⏱️  Performance:")
        print(f"  Total time: {summary['total_time']:.2f}s")
        print(f"  Average time per image: {summary['avg_time_per_image']:.2f}s")
        print(f"  Images per second: {1/summary['avg_time_per_image']:.2f}")
        
        print(f"\n📁 Output Directory: {os.path.abspath(self.config.OUTPUT_DIR)}")
        vis_dir = os.path.join(self.config.OUTPUT_DIR, 'visualizations')
        print(f"📁 Visualizations: {os.path.abspath(vis_dir)}")
        print(f"{'=' * 80}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference with signature model on validation images"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=None,
        help='Path to model weights (default: best trained signature model)'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        default=None,
        help='Path to images directory (default: signature_dataset/images/val)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='outputs/inference/signature',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou', 
        type=float, 
        default=0.45,
        help='NMS IoU threshold (default: 0.45)'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=1024,
        help='Image size for inference (default: 1024)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cpu',
        help='Device to run on (cpu or cuda)'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Do not save detections as JSON'
    )
    parser.add_argument(
        '--no-grouping',
        action='store_true',
        help='Disable signature grouping'
    )
    parser.add_argument(
        '--group-iou',
        type=float,
        default=0.3,
        help='IoU threshold for grouping signatures (default: 0.3)'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use last.pt (latest checkpoint) instead of best.pt'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = SignatureInferenceConfig()
    
    if args.model:
        config.SIGNATURE_MODEL = args.model
    if args.source:
        config.VAL_IMAGES = args.source
        # Also try to find corresponding labels directory
        source_path = Path(args.source)
        if source_path.name == 'val' and source_path.parent.name == 'images':
            labels_path = source_path.parent.parent / 'labels' / 'val'
            if labels_path.exists():
                config.VAL_LABELS = str(labels_path)
    
    config.OUTPUT_DIR = args.output
    config.CONF_THRESHOLD = args.conf
    config.IOU_THRESHOLD = args.iou
    config.IMGSZ = args.imgsz
    config.DEVICE = args.device
    config.SAVE_JSON = not args.no_json
    config.ENABLE_GROUPING = not args.no_grouping
    config.GROUP_IOU_THRESHOLD = args.group_iou
    
    # Handle output directory with _latest suffix
    if config.OUTPUT_DIR == "outputs/inference/signature" and args.latest:
        config.OUTPUT_DIR = "outputs/inference/signature_latest"
    
    # Check if source directory exists
    if not os.path.exists(config.VAL_IMAGES):
        print(f"Error: Source directory not found: {config.VAL_IMAGES}")
        return
    
    try:
        # Initialize inference
        inference = SignatureModelInference(config, use_latest=args.latest)
        
        # Process images
        summary = inference.process_directory(config.VAL_IMAGES)
        
        # Print summary
        inference.print_summary(summary)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
