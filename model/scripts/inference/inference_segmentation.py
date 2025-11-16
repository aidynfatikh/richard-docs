#!/usr/bin/env python3
"""
Test YOLOv11-seg segmentation model inference
Run predictions on test images and visualize results with masks.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to trained model weights
MODEL_PATH = "runs/segment/yolov11n_seg_20251115_203540/weights/best.pt"  # Update with your run

# Test images directory (use validation set or new images)
TEST_IMAGES_DIR = "seg_dataset/images/val"  # or "seg_dataset/images/train" or any folder

# Output directory for results
OUTPUT_DIR = "outputs/inference/segmentation"

# Ground truth labels directory
GT_LABELS_DIR = "data/datasets/segmentation/masks/val"  # Ground truth masks

# Confidence threshold
CONF_THRESHOLD = 0.25

# Image size (must match training: 640 for segmentation)
IMGSZ = 640

# Save options
SAVE_JSON = True
SAVE_VISUALIZATIONS = True

# Classes
CLASS_NAMES = {
    0: 'stamp',
    1: 'signature'
}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 0, 255),      # Stamp - Red
    1: (0, 255, 0)       # Signature - Green
}


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def find_latest_model(base_dir="runs/segment"):
    """Find the best trained model by searching for highest mAP in training_metrics.csv"""
    if not os.path.exists(base_dir):
        return None
    
    best_model_path = None
    best_map = -1
    best_run = None
    
    # Search all run directories
    for run_dir in Path(base_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check for training_metrics.csv and best.pt
        metrics_file = run_dir / "training_metrics.csv"
        weights_file = run_dir / "weights" / "best.pt"
        
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
                        # Find best mask mAP50-95
                        if 'mask_mAP50-95' in rows[0]:
                            max_map = max(float(row['mask_mAP50-95']) for row in rows if row['mask_mAP50-95'])
                        elif 'mAP50-95' in rows[0]:
                            max_map = max(float(row['mAP50-95']) for row in rows if row['mAP50-95'])
                        else:
                            continue
                        
                        if max_map > best_map:
                            best_map = max_map
                            best_model_path = str(weights_file)
                            best_run = run_dir.name
            except Exception:
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
            print(f"  Auto-selected best model: {best_run} (mask mAP: {best_map:.4f})")
        else:
            print(f"  Auto-selected model: {best_run} (no metrics available)")
    
    best_model = best_model_path
    
    if os.path.exists(best_model):
        return best_model
    
    return None


def load_ground_truth_mask(image_path, dataset_dir="seg_dataset"):
    """
    Load ground truth mask for an image.
    
    Args:
        image_path: Path to image file
        dataset_dir: Base directory of dataset
        
    Returns:
        Ground truth mask as numpy array or None if not found
    """
    # Determine if image is in train or val
    img_name = os.path.basename(image_path)
    
    for split in ['train', 'val']:
        mask_path = os.path.join(dataset_dir, 'masks', split, img_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
    
    return None


def visualize_with_ground_truth(image, results, gt_mask, save_path):
    """
    Create side-by-side visualization: Ground Truth | Prediction
    
    Args:
        image: Original image (numpy array)
        results: YOLO prediction results
        gt_mask: Ground truth mask (grayscale, values 0/1/2)
        save_path: Path to save visualization
    """
    h, w = image.shape[:2]
    
    # Create ground truth visualization
    gt_vis = image.copy()
    gt_overlay = gt_vis.copy()
    
    if gt_mask is not None:
        # Resize GT mask if needed
        if gt_mask.shape != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Color the ground truth
        stamp_mask = (gt_mask == 1)
        signature_mask = (gt_mask == 2)
        
        gt_overlay[stamp_mask] = CLASS_COLORS[0]  # Red for stamps
        gt_overlay[signature_mask] = CLASS_COLORS[1]  # Green for signatures
        
        # Blend
        alpha = 0.4
        gt_vis = cv2.addWeighted(gt_vis, 1 - alpha, gt_overlay, alpha, 0)
        
        # Add title
        cv2.putText(gt_vis, "GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 3)
        cv2.putText(gt_vis, "GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 0), 2)
    else:
        # No ground truth available
        cv2.putText(gt_vis, "NO GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2)
    
    # Create prediction visualization
    pred_vis = image.copy()
    pred_overlay = pred_vis.copy()
    num_detections = 0
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        num_detections = len(masks)
        
        # Process each detection
        for mask, box, cls, conf in zip(masks, boxes, classes, confs):
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Get color
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            
            # Apply colored mask
            pred_overlay[mask_binary == 1] = color
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{CLASS_NAMES.get(cls, 'unknown')} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(pred_vis, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(pred_vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        # Blend
        alpha = 0.4
        pred_vis = cv2.addWeighted(pred_vis, 1 - alpha, pred_overlay, alpha, 0)
    
    # Add title to prediction
    cv2.putText(pred_vis, "PREDICTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (255, 255, 255), 3)
    cv2.putText(pred_vis, "PREDICTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 0, 0), 2)
    
    # Create separator
    separator = np.ones((h, 10, 3), dtype=np.uint8) * 255
    
    # Combine side by side: GT | Separator | Prediction
    combined = np.hstack([gt_vis, separator, pred_vis])
    
    # Add legend at bottom
    legend_height = 80
    legend_width = combined.shape[1]
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    # Draw legend
    cv2.rectangle(legend, (10, 10), (30, 30), CLASS_COLORS[0], -1)
    cv2.putText(legend, "Stamp (Class 0)", (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(legend, (10, 45), (30, 65), CLASS_COLORS[1], -1)
    cv2.putText(legend, "Signature (Class 1)", (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add stats on the right side
    if gt_mask is not None:
        gt_stamps = np.sum(gt_mask == 1)
        gt_sigs = np.sum(gt_mask == 2)
        stats_text = f"GT: {gt_stamps + gt_sigs} pixels | Pred: {num_detections} objects"
        cv2.putText(legend, stats_text, (legend_width - 450, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Stack vertically
    final_vis = np.vstack([combined, legend])
    
    # Save
    cv2.imwrite(save_path, final_vis)
    
    return num_detections


def visualize_segmentation(image, results, save_path):
    """
    Visualize segmentation results with masks and bounding boxes.
    
    Args:
        image: Original image (numpy array)
        results: YOLO prediction results
        save_path: Path to save visualization
    """
    vis_img = image.copy()
    
    # Get results
    if results[0].masks is None:
        print("  No masks detected")
        cv2.imwrite(save_path, vis_img)
        return
    
    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = results[0].boxes.conf.cpu().numpy()  # (N,)
    
    h, w = image.shape[:2]
    
    # Create overlay for masks
    overlay = vis_img.copy()
    
    # Process each detection
    for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confs)):
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Get color
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        # Apply colored mask
        overlay[mask_binary == 1] = color
        
        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{CLASS_NAMES.get(cls, 'unknown')} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original
    alpha = 0.4
    vis_img = cv2.addWeighted(vis_img, 1 - alpha, overlay, alpha, 0)
    
    # Add legend
    legend_height = 80
    legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255
    
    cv2.rectangle(legend, (10, 10), (30, 30), CLASS_COLORS[0], -1)
    cv2.putText(legend, "Stamp", (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(legend, (10, 45), (30, 65), CLASS_COLORS[1], -1)
    cv2.putText(legend, "Signature", (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Combine
    final_vis = np.vstack([vis_img, legend])
    
    # Save
    cv2.imwrite(save_path, final_vis)
    
    return len(masks)


def run_inference(model_path, test_dir, output_dir, conf_threshold=0.25, save_json=True):
    """
    Run segmentation inference on test images.
    
    Args:
        model_path: Path to trained model weights
        test_dir: Directory containing test images
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        save_json: Whether to save detections as JSON
    """
    
    print("\n" + "=" * 80)
    print("SEGMENTATION INFERENCE: YOLOv11-seg Segmentation Model")
    print("=" * 80)
    
    # Check model
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        
        # Try to find latest model
        latest = find_latest_model()
        if latest:
            print(f"   Found latest model: {latest}")
            model_path = latest
        else:
            print("   No trained models found. Please train a model first.")
            return
    
    # Check test directory
    if not os.path.exists(test_dir):
        print(f"\n❌ Test directory not found: {test_dir}")
        return
    
    # Load model
    print(f"\n[1/4] Loading model...")
    print(f"  Model: {model_path}")
    model = YOLO(model_path)
    print("  ✓ Model loaded")
    
    # Get test images
    print(f"\n[2/4] Finding test images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(Path(test_dir).glob(f'*{ext}'))
        test_images.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    
    test_images = sorted(test_images)
    print(f"  Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("  ❌ No images found for testing")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"  Output: {output_dir}")
    
    # Run inference
    print(f"\n[3/4] Processing images...")
    print(f"  Confidence threshold: {conf_threshold}")
    print("-" * 80)
    
    all_detections = {}
    stats = {
        'total_images': 0,
        'total_detections': 0,
        'by_class': {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 0}
    }
    
    for img_path in test_images:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not read: {img_path.name}")
            continue
        
        # Run prediction
        results = model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=IMGSZ,
            task='segment',
            verbose=False
        )
        
        # Extract detections
        detections = []
        if results[0].masks is not None and len(results[0].masks) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                detections.append({
                    'box': [float(x) for x in box],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': CLASS_NAMES.get(cls, 'unknown')
                })
                stats['by_class'][CLASS_NAMES.get(cls, 'unknown')] += 1
        
        # Store detections
        all_detections[img_path.name] = detections
        stats['total_images'] += 1
        stats['total_detections'] += len(detections)
        
        # Load ground truth mask for visualization
        gt_mask = load_ground_truth_mask(str(img_path))
        
        # Visualize and save (with GT if available)
        output_path = os.path.join(vis_dir, f"seg_{img_path.name}")
        
        if gt_mask is not None:
            visualize_with_ground_truth(image, results, gt_mask, output_path)
        else:
            visualize_segmentation(image, results, output_path)
        
        print(f"  ✓ {img_path.name}: {len(detections)} detections")
    
    print("-" * 80)
    
    # Save detections as JSON
    if save_json:
        json_path = os.path.join(output_dir, 'detections.json')
        import json
        with open(json_path, 'w') as f:
            json.dump({
                'detections': all_detections,
                'statistics': stats
            }, f, indent=2)
        print(f"\n✓ Saved detections: {json_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SEGMENTATION INFERENCE COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Statistics:")
    print(f"  Total images processed: {stats['total_images']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average per image: {stats['total_detections']/max(stats['total_images'],1):.2f}")
    
    print(f"\n📦 Detections by Class:")
    for cls_name, count in sorted(stats['by_class'].items()):
        pct = 100 * count / max(stats['total_detections'], 1)
        print(f"  {cls_name:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📁 Output Directory: {os.path.abspath(output_dir)}")
    print("=" * 80)


def test_single_image(model_path, image_path, output_path=None):
    """
    Test segmentation on a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        output_path: Path to save result (optional)
    """
    
    if not os.path.exists(model_path):
        # Try to find latest
        model_path = find_latest_model()
        if not model_path:
            print("❌ No model found")
            return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Run prediction
    print(f"Running inference on: {image_path}")
    results = model.predict(source=image, task='segment', conf=0.25, imgsz=IMGSZ, verbose=False)
    
    # Load ground truth
    gt_mask = load_ground_truth_mask(image_path)
    
    # Visualize
    if output_path is None:
        output_path = f"seg_result_{os.path.basename(image_path)}"
    
    if gt_mask is not None:
        num_detections = visualize_with_ground_truth(image, results, gt_mask, output_path)
    else:
        num_detections = visualize_segmentation(image, results, output_path)
    
    print(f"✓ Detected {num_detections} objects")
    print(f"✓ Result saved to: {output_path}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLOv11-seg segmentation model')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (default: latest)')
    parser.add_argument('--images', type=str, default=TEST_IMAGES_DIR, help='Directory with test images')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--single', type=str, default=None, help='Test single image')
    
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model if args.model else find_latest_model()
    
    if model_path is None:
        print("❌ No model specified and no trained models found")
        print("   Train a model first with: python3 train_yolov11_seg_lora.py")
        return
    
    # Single image or batch
    if args.single:
        test_single_image(model_path, args.single)
    else:
        run_inference(model_path, args.images, args.output, args.conf, save_json=SAVE_JSON)


if __name__ == "__main__":
    main()
