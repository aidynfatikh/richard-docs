#!/usr/bin/env python3
"""
Production Inference Script

Uses the best performing detection model (yolov11s_lora_20251115_230142) with smart grouping
of stacked annotations (signatures, stamps, QR codes that appear together).

Features:
- Low confidence threshold (0.15) to catch all potential detections
- Smart grouping of nearby/overlapping boxes
- Clean JSON output with grouped annotations
- Visualization with grouped boxes

Usage:
    python3 inference.py --source <image_or_directory> --output <output_dir>
    python3 inference.py --source data/raw/test_images --output outputs/inference/production
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Inference configuration"""
    
    # Best model (from testing results)
    MODEL_PATH = "runs/train/yolov11s_lora_20251115_230142/weights/best.pt"
    
    # Detection settings
    CONF_THRESHOLD = 0.15  # Low threshold to catch all detections
    IOU_THRESHOLD = 0.45   # NMS threshold
    
    # Grouping settings
    GROUP_DISTANCE_THRESHOLD = 50  # Pixels - boxes within this distance are grouped
    GROUP_IOU_THRESHOLD = 0.2      # Boxes with IoU > this are grouped (lowered to catch more overlaps)
    # Containment threshold of 0.6 is hardcoded in boxes_are_close() to catch partial detections
    
    # Class names
    CLASS_NAMES = {0: 'qr', 1: 'signature', 2: 'stamp'}
    CLASS_COLORS = {
        0: (255, 0, 0),    # QR: Blue
        1: (0, 255, 0),    # Signature: Green
        2: (0, 0, 255)     # Stamp: Red
    }


# ============================================================================
# GROUPING LOGIC
# ============================================================================

def calculate_distance(box1: List[int], box2: List[int]) -> float:
    """Calculate center-to-center distance between two boxes"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two boxes"""
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


def calculate_containment(box1: List[int], box2: List[int]) -> float:
    """Calculate how much box1 is contained within box2 (0.0 to 1.0)"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    return intersection / area1 if area1 > 0 else 0.0


def boxes_are_close(box1: Dict, box2: Dict, 
                    distance_threshold: float,
                    iou_threshold: float) -> bool:
    """Check if two boxes should be grouped together"""
    
    # Calculate distance between centers
    distance = calculate_distance(box1['box'], box2['box'])
    
    # Calculate IoU
    iou = calculate_iou(box1['box'], box2['box'])
    
    # Calculate containment (for signatures inside other signatures)
    containment1 = calculate_containment(box1['box'], box2['box'])
    containment2 = calculate_containment(box2['box'], box1['box'])
    max_containment = max(containment1, containment2)
    
    # Group if:
    # 1. Close enough (centers nearby)
    # 2. Overlapping enough (IoU)
    # 3. One box is mostly inside another (>60% containment - catches partial detections)
    return (distance < distance_threshold or 
            iou > iou_threshold or 
            max_containment > 0.6)


def group_detections(detections: List[Dict], 
                    distance_threshold: float = 50,
                    iou_threshold: float = 0.3) -> List[Dict]:
    """
    Group nearby/overlapping detections of the SAME CLASS into unified detections.
    Stacked/overlapping signatures become one signature, etc.
    
    Strategy:
    1. First pass: Remove boxes that are almost completely inside other boxes (duplicates)
    2. Second pass: Merge remaining nearby/overlapping boxes
    
    Returns merged list of detections (ungrouped singles + grouped merged boxes)
    """
    
    if not detections:
        return []
    
    # Group by class
    by_class = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in by_class:
            by_class[class_id] = []
        by_class[class_id].append(det)
    
    # Process each class separately
    merged_detections = []
    
    for class_id, class_detections in by_class.items():
        # Sort by confidence (highest first) then by area (largest first)
        class_detections = sorted(class_detections, 
                                 key=lambda x: (x['confidence'], 
                                              (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1])), 
                                 reverse=True)
        
        # PASS 1: Remove boxes that are almost completely inside other boxes
        # This handles duplicate/redundant detections of same object
        kept_detections = []
        for i, det1 in enumerate(class_detections):
            is_redundant = False
            
            # Check against ALL other boxes (not just kept ones)
            # This catches cases where lower-conf box is added first
            for j, det2 in enumerate(class_detections):
                if i == j:
                    continue
                
                containment = calculate_containment(det1['box'], det2['box'])
                
                # If >75% of this box is inside another box, check which one to keep
                if containment > 0.75:
                    # Keep the one with higher confidence OR larger area
                    if det2['confidence'] > det1['confidence']:
                        is_redundant = True
                        break
                    elif det2['confidence'] == det1['confidence']:
                        # Same confidence - keep larger box
                        area1 = (det1['box'][2] - det1['box'][0]) * (det1['box'][3] - det1['box'][1])
                        area2 = (det2['box'][2] - det2['box'][0]) * (det2['box'][3] - det2['box'][1])
                        if area2 > area1:
                            is_redundant = True
                            break
            
            if not is_redundant:
                kept_detections.append(det1)
        
        # PASS 2: Merge remaining nearby/overlapping boxes
        assigned = set()
        
        for i, det1 in enumerate(kept_detections):
            if i in assigned:
                continue
            
            # Start new group for this detection
            group = [det1]
            assigned.add(i)
            
            # Find all same-class detections that overlap/are close to this one
            for j, det2 in enumerate(kept_detections):
                if j in assigned:
                    continue
                
                # Check if det2 should be grouped with any detection in current group
                should_group = False
                for group_det in group:
                    if boxes_are_close(group_det, det2, distance_threshold, iou_threshold):
                        should_group = True
                        break
                
                if should_group:
                    group.append(det2)
                    assigned.add(j)
            
            # Merge group into single detection
            if len(group) == 1:
                # Single detection, keep as-is
                merged_detections.append(group[0])
            else:
                # Multiple overlapping detections - merge into one
                all_boxes = [d['box'] for d in group]
                x1 = min(box[0] for box in all_boxes)
                y1 = min(box[1] for box in all_boxes)
                x2 = max(box[2] for box in all_boxes)
                y2 = max(box[3] for box in all_boxes)
                
                # Use highest confidence
                max_conf = max(d['confidence'] for d in group)
                
                merged_detections.append({
                    'class_id': class_id,
                    'class_name': det1['class_name'],
                    'confidence': max_conf,
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'merged_count': len(group)  # Track how many boxes were merged
                })
    
    return merged_detections


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(image_path: Path, model, config: InferenceConfig) -> Tuple[List[Dict], List[Dict]]:
    """
    Run inference on a single image.
    
    Returns:
        (raw_detections, merged_detections)
    """
    
    # Run model
    results = model(str(image_path), conf=config.CONF_THRESHOLD, iou=config.IOU_THRESHOLD, verbose=False)
    
    # Extract raw detections
    raw_detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            raw_detections.append({
                'class_id': cls,
                'class_name': config.CLASS_NAMES.get(cls, str(cls)),
                'confidence': conf,
                'box': [int(x1), int(y1), int(x2), int(y2)]
            })
    
    # Merge overlapping detections of same class
    merged_detections = group_detections(
        raw_detections,
        distance_threshold=config.GROUP_DISTANCE_THRESHOLD,
        iou_threshold=config.GROUP_IOU_THRESHOLD
    )
    
    return raw_detections, merged_detections


def visualize_results(image_path: Path, 
                     raw_detections: List[Dict],
                     merged_detections: List[Dict],
                     output_path: Path,
                     config: InferenceConfig):
    """Create side-by-side visualization: raw detections (left) vs merged (right)"""
    
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # Left: Raw detections
    img_raw = img.copy()
    for det in raw_detections:
        x1, y1, x2, y2 = det['box']
        color = config.CLASS_COLORS.get(det['class_id'], (255, 255, 255))
        
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(img_raw, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Right: Merged detections
    img_merged = img.copy()
    for det in merged_detections:
        x1, y1, x2, y2 = det['box']
        color = config.CLASS_COLORS.get(det['class_id'], (255, 255, 255))
        
        cv2.rectangle(img_merged, (x1, y1), (x2, y2), color, 2)
        
        # Show if merged
        if det.get('merged_count', 1) > 1:
            label = f"{det['class_name']} {det['confidence']:.2f} [merged {det['merged_count']}]"
        else:
            label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(img_merged, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add titles
    title_raw = np.zeros((40, w, 3), dtype=np.uint8)
    title_merged = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(title_raw, f"Raw Detections ({len(raw_detections)})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(title_merged, f"Merged Detections ({len(merged_detections)})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine
    img_raw_titled = np.vstack([title_raw, img_raw])
    img_merged_titled = np.vstack([title_merged, img_merged])
    combined = np.hstack([img_raw_titled, img_merged_titled])
    
    cv2.imwrite(str(output_path), combined)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run inference with smart grouping')
    parser.add_argument('--source', required=True, help='Image file or directory')
    parser.add_argument('--output', default='inference_output', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--group-dist', type=float, default=50, help='Grouping distance threshold')
    parser.add_argument('--group-iou', type=float, default=0.3, help='Grouping IoU threshold')
    
    args = parser.parse_args()
    
    config = InferenceConfig()
    config.CONF_THRESHOLD = args.conf
    config.GROUP_DISTANCE_THRESHOLD = args.group_dist
    config.GROUP_IOU_THRESHOLD = args.group_iou
    
    print("=" * 80)
    print("PRODUCTION INFERENCE WITH SMART GROUPING")
    print("=" * 80)
    print(f"\nModel: {config.MODEL_PATH}")
    print(f"Confidence threshold: {config.CONF_THRESHOLD}")
    print(f"Grouping distance: {config.GROUP_DISTANCE_THRESHOLD}px")
    print(f"Grouping IoU: {config.GROUP_IOU_THRESHOLD}")
    
    # Check model exists
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ Model not found: {config.MODEL_PATH}")
        sys.exit(1)
    
    # Load model
    print("\nLoading model...")
    from ultralytics import YOLO
    model = YOLO(config.MODEL_PATH)
    print("✓ Model loaded")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Get input images
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = sorted(list(source_path.glob("*.png")) + 
                           list(source_path.glob("*.jpg")) +
                           list(source_path.glob("*.jpeg")))
    else:
        print(f"\n❌ Source not found: {args.source}")
        sys.exit(1)
    
    print(f"\nProcessing {len(image_files)} images...")
    print("=" * 80)
    
    # Process each image
    all_results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")
        
        try:
            # Run inference
            raw_detections, merged_detections = run_inference(img_path, model, config)
            
            # Save results
            result = {
                'image': img_path.name,
                'raw_detections': raw_detections,
                'merged_detections': merged_detections,
                'summary': {
                    'raw_count': len(raw_detections),
                    'merged_count': len(merged_detections),
                    'qr_count': sum(1 for d in merged_detections if d['class_id'] == 0),
                    'signature_count': sum(1 for d in merged_detections if d['class_id'] == 1),
                    'stamp_count': sum(1 for d in merged_detections if d['class_id'] == 2)
                }
            }
            all_results.append(result)
            
            # Print summary
            print(f"  Raw: {result['summary']['raw_count']} → Merged: {result['summary']['merged_count']} "
                  f"(QR:{result['summary']['qr_count']}, "
                  f"Sig:{result['summary']['signature_count']}, "
                  f"Stamp:{result['summary']['stamp_count']})")
            
            # Visualize
            vis_path = vis_dir / img_path.name
            visualize_results(img_path, raw_detections, merged_detections, vis_path, config)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save JSON results
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': config.MODEL_PATH,
            'config': {
                'conf_threshold': config.CONF_THRESHOLD,
                'group_distance': config.GROUP_DISTANCE_THRESHOLD,
                'group_iou': config.GROUP_IOU_THRESHOLD
            },
            'results': all_results
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE")
    print("=" * 80)
    print(f"\nProcessed: {len(all_results)} images")
    print(f"Raw detections: {sum(r['summary']['raw_count'] for r in all_results)}")
    print(f"Merged detections: {sum(r['summary']['merged_count'] for r in all_results)}")
    print(f"  QR codes: {sum(r['summary']['qr_count'] for r in all_results)}")
    print(f"  Signatures: {sum(r['summary']['signature_count'] for r in all_results)}")
    print(f"  Stamps: {sum(r['summary']['stamp_count'] for r in all_results)}")
    print(f"\nOutputs:")
    print(f"  Results JSON: {json_path}")
    print(f"  Visualizations: {vis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
