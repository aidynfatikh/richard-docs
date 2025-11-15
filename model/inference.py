#!/usr/bin/env python3
"""
YOLOv11 Inference Script
Optimized inference on images with visual output and JSON results.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Inference configuration parameters"""
    
    # Model
    MODEL_PATH = "runs/train/yolov11s_lora_20251115_144250/weights/best.pt"
    
    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
    IOU_THRESHOLD = 0.45  # NMS IoU threshold
    IMAGE_SIZE = 1024  # Input image size
    
    # Visualization
    BOX_THICKNESS = 3
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    
    # Colors for each class (BGR format)
    CLASS_COLORS = {
        'qr': (255, 0, 0),        # Blue
        'signature': (0, 255, 0),  # Green
        'stamp': (0, 0, 255),      # Red
    }
    
    # Output
    OUTPUT_DIR = "inference_results"
    SAVE_VISUALIZATIONS = True
    SAVE_JSON = True


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_model(model_path, device='auto'):
    """
    Load YOLO model from checkpoint.
    
    Args:
        model_path: Path to model weights (.pt file)
        device: Device to run inference on ('auto', 'cpu', '0', etc.)
    
    Returns:
        Loaded YOLO model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Classes: {model.names}")
    
    return model


def run_inference(model, image_path, config):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model
        image_path: Path to input image
        config: InferenceConfig object
    
    Returns:
        dict: Detection results with boxes, labels, and confidences
    """
    # Run inference
    results = model.predict(
        source=image_path,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        imgsz=config.IMAGE_SIZE,
        verbose=False
    )[0]
    
    # Load original image for visualization
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Parse detections
    detections = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            
            detection = {
                'class_id': int(cls_id),
                'class_name': model.names[cls_id],
                'confidence': float(conf),
                'bbox': {
                    'x_min': float(x1),
                    'y_min': float(y1),
                    'x_max': float(x2),
                    'y_max': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                },
                'bbox_normalized': {
                    'x_center': float((x1 + x2) / 2 / img_width),
                    'y_center': float((y1 + y2) / 2 / img_height),
                    'width': float((x2 - x1) / img_width),
                    'height': float((y2 - y1) / img_height)
                }
            }
            detections.append(detection)
    
    result = {
        'image': os.path.basename(image_path),
        'image_path': image_path,
        'image_size': {
            'width': img_width,
            'height': img_height
        },
        'num_detections': len(detections),
        'detections': detections,
        'timestamp': datetime.now().isoformat()
    }
    
    return result, img


def visualize_detections(img, detections, config):
    """
    Draw bounding boxes and labels on image.
    
    Args:
        img: Input image (numpy array)
        detections: List of detection dictionaries
        config: InferenceConfig object
    
    Returns:
        Image with drawn detections
    """
    img_vis = img.copy()
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        bbox = det['bbox']
        
        # Get coordinates
        x1 = int(bbox['x_min'])
        y1 = int(bbox['y_min'])
        x2 = int(bbox['x_max'])
        y2 = int(bbox['y_max'])
        
        # Get color for this class
        color = config.CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
        )
        
        # Draw label background
        label_y1 = max(text_height + 10, y1)
        cv2.rectangle(
            img_vis,
            (x1, label_y1 - text_height - 10),
            (x1 + text_width + 10, label_y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_vis,
            label,
            (x1 + 5, label_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),
            config.FONT_THICKNESS
        )
    
    return img_vis


def save_results(result, img_vis, output_dir, config):
    """
    Save inference results to disk.
    
    Args:
        result: Detection results dictionary
        img_vis: Visualized image with bounding boxes
        output_dir: Output directory path
        config: InferenceConfig object
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(result['image']).stem
    
    # Save visualization
    if config.SAVE_VISUALIZATIONS:
        vis_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
        cv2.imwrite(vis_path, img_vis)
        print(f"  Visualization saved: {vis_path}")
    
    # Save JSON
    if config.SAVE_JSON:
        json_path = os.path.join(output_dir, f"{base_name}_detections.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  JSON saved: {json_path}")


def process_image(model, image_path, output_dir, config):
    """
    Process a single image: inference + visualization + save results.
    
    Args:
        model: YOLO model
        image_path: Path to input image
        output_dir: Output directory
        config: InferenceConfig object
    """
    print(f"\nProcessing: {image_path}")
    
    # Run inference
    result, img = run_inference(model, image_path, config)
    
    print(f"  Detections: {result['num_detections']}")
    for det in result['detections']:
        print(f"    - {det['class_name']}: {det['confidence']:.3f}")
    
    # Visualize
    img_vis = visualize_detections(img, result['detections'], config)
    
    # Save results
    save_results(result, img_vis, output_dir, config)


def process_directory(model, input_dir, output_dir, config):
    """
    Process all images in a directory.
    
    Args:
        model: YOLO model
        input_dir: Input directory containing images
        output_dir: Output directory for results
        config: InferenceConfig object
    """
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]", end=" ")
        process_image(model, str(image_path), output_dir, config)
    
    # Create summary JSON
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        'total_images': len(image_paths),
        'processed_at': datetime.now().isoformat(),
        'model': config.MODEL_PATH,
        'confidence_threshold': config.CONFIDENCE_THRESHOLD,
        'images': [str(p.name) for p in image_paths]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='YOLOv11 Inference Script - Detect stamps, signatures, and QR codes'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=InferenceConfig.MODEL_PATH,
        help='Path to model weights (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=InferenceConfig.OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=InferenceConfig.CONFIDENCE_THRESHOLD,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=InferenceConfig.IOU_THRESHOLD,
        help='NMS IoU threshold (0-1)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=InferenceConfig.IMAGE_SIZE,
        help='Input image size'
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Disable visualization output'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Disable JSON output'
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config = InferenceConfig()
    config.MODEL_PATH = args.model
    config.OUTPUT_DIR = args.output
    config.CONFIDENCE_THRESHOLD = args.conf
    config.IOU_THRESHOLD = args.iou
    config.IMAGE_SIZE = args.imgsz
    config.SAVE_VISUALIZATIONS = not args.no_vis
    config.SAVE_JSON = not args.no_json
    
    # Print configuration
    print("=" * 70)
    print("YOLOv11 Inference")
    print("=" * 70)
    print(f"Model: {config.MODEL_PATH}")
    print(f"Input: {args.input}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"IOU threshold: {config.IOU_THRESHOLD}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print("=" * 70)
    
    # Load model
    model = load_model(config.MODEL_PATH)
    
    # Process input
    input_path = args.input
    
    if os.path.isfile(input_path):
        # Single image
        process_image(model, input_path, config.OUTPUT_DIR, config)
    elif os.path.isdir(input_path):
        # Directory of images
        process_directory(model, input_path, config.OUTPUT_DIR, config)
    else:
        print(f"Error: Input not found: {input_path}")
        return
    
    print("\n" + "=" * 70)
    print("Inference complete!")
    print(f"Results saved to: {os.path.abspath(config.OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
