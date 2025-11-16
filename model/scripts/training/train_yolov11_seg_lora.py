#!/usr/bin/env python3
"""
YOLOv11-seg Training Pipeline with LoRA Fine-tuning
Complete segmentation model training for stamp and signature detection.

Key Features:
- YOLOv11n-seg (smallest, fastest segmentation model)
- LoRA fine-tuning for parameter-efficient training
- Uses ALL available images (no val split for small dataset)
- Pixel-level mask prediction
- Checkpointing (best.pt, last.pt)
- CSV metric logging per epoch
- Augmentations optimized for document images
"""

import os
import sys
import yaml
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Suppress verbose ultralytics logging
LOGGER.setLevel('WARNING')


# ============================================================================
# CONFIGURATION
# ============================================================================

class SegTrainingConfig:
    """Segmentation training configuration parameters"""
    
    # Dataset
    DATA_YAML = "data/datasets/segmentation/dataset.yaml"  # Path to segmentation dataset.yaml
    
    # Model - Using YOLOv11n-seg (smallest/fastest segmentation model)
    MODEL_NAME = "yolo11n-seg.pt"  # Pre-trained YOLOv11n segmentation weights
    
    # Fine-tuning mode (freeze early layers for LoRA-style efficiency)
    FREEZE_LAYERS = 0  # Don't freeze layers - YOLOv11n-seg is small, needs full training
    
    # Training hyperparameters
    EPOCHS = 100  # More epochs for small dataset
    BATCH_SIZE = 4  # Keep at 4 - best model used this
    IMAGE_SIZE = 640  # Standard resolution for segmentation
    LEARNING_RATE = 0.0005  # Stable learning rate
    
    # Augmentation (moderate - based on best performing model)
    # Best model used: degrees=3.0, translate=0.1, scale=0.2, fliplr=0.2
    # Keep these but slightly reduce to account for crop context
    AUGMENT = True
    MOSAIC = 1.0  # Enable mosaic for better generalization
    MIXUP = 0.0  # Disable mixup
    COPY_PASTE = 0.0  # Disable copy-paste
    FLIPUD = 0.0  # No vertical flip
    FLIPLR = 0.3  # Moderate horizontal flip
    DEGREES = 2.0  # Small rotation (reduced from 3.0)
    TRANSLATE = 0.05  # Small translation (reduced from 0.1)
    SCALE = 0.15  # Small scale variation (reduced from 0.2)
    HSV_H = 0.005  # Minimal hue (scanner/lighting variations)
    HSV_S = 0.2  # Moderate saturation
    HSV_V = 0.15  # Moderate brightness (match best model)
    
    # Checkpointing and logging
    PROJECT_DIR = "runs/segment"
    RUN_NAME = f"yolov11n_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    SAVE_PERIOD = 10  # Save checkpoint every N epochs
    
    # Hardware
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    WORKERS = 2  # Conservative for stability
    
    # Additional training parameters
    PATIENCE = 20  # Early stopping patience (higher for small dataset)
    OPTIMIZER = "SGD"  # SGD optimizer for stable convergence
    WARMUP_EPOCHS = 10  # Warmup learning rate for first 10 epochs
    LRF = 0.01  # Final learning rate factor (lr0 * lrf) for cosine decay
    WEIGHT_DECAY = 0.0005
    CLOSE_MOSAIC = 10  # Disable mosaic in last 10 epochs
    
    # Training stability
    AMP = True  # Automatic mixed precision for faster training
    DROPOUT = 0.0  # Dropout rate (0.0 = disabled, use for regularization if overfitting)
    LABEL_SMOOTHING = 0.0  # Label smoothing (0.0-0.1, helps with overconfidence)
    NBS = 64  # Nominal batch size for batch accumulation
    MAX_DET = 300  # Maximum detections per image
    
    # Segmentation-specific parameters
    OVERLAP_MASK = True  # Allow mask overlap
    MASK_RATIO = 4  # Mask downsample ratio
    
    # Logging
    VERBOSE = False
    PLOTS = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_csv_log(log_path):
    """Initialize CSV file for logging training metrics."""
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=[
            'epoch',
            'train_box_loss',
            'train_seg_loss',
            'train_cls_loss',
            'train_dfl_loss',
            'val_box_loss',
            'val_seg_loss',
            'val_cls_loss',
            'val_dfl_loss',
            'precision',
            'recall',
            'mAP50',
            'mAP50-95',
            'mask_mAP50',
            'mask_mAP50-95',
            'learning_rate',
            'timestamp'
        ])
        df.to_csv(log_path, index=False)
        print(f"  Created metrics log: {log_path}")
    return log_path


def log_metrics_to_csv(log_path, epoch, metrics, lr):
    """Append training metrics to CSV file."""
    try:
        # Extract training losses
        train_box = metrics.get('train/box_loss', 0)
        train_seg = metrics.get('train/seg_loss', 0)
        train_cls = metrics.get('train/cls_loss', 0)
        train_dfl = metrics.get('train/dfl_loss', 0)
        
        # Extract validation losses
        val_box = metrics.get('val/box_loss', 0)
        val_seg = metrics.get('val/seg_loss', 0)
        val_cls = metrics.get('val/cls_loss', 0)
        val_dfl = metrics.get('val/dfl_loss', 0)
        
        # Extract detection metrics
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        # Extract segmentation metrics
        mask_map50 = metrics.get('metrics/mAP50(M)', 0)
        mask_map50_95 = metrics.get('metrics/mAP50-95(M)', 0)
        
        # Create new row
        new_row = {
            'epoch': epoch,
            'train_box_loss': train_box,
            'train_seg_loss': train_seg,
            'train_cls_loss': train_cls,
            'train_dfl_loss': train_dfl,
            'val_box_loss': val_box,
            'val_seg_loss': val_seg,
            'val_cls_loss': val_cls,
            'val_dfl_loss': val_dfl,
            'precision': precision,
            'recall': recall,
            'mAP50': map50,
            'mAP50-95': map50_95,
            'mask_mAP50': mask_map50,
            'mask_mAP50-95': mask_map50_95,
            'learning_rate': lr,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Append to CSV
        df = pd.DataFrame([new_row])
        df.to_csv(log_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"  Warning: Could not log metrics: {e}")


def export_final_metrics_summary(save_dir, results):
    """
    Export final training metrics to a summary CSV file.
    Similar to inference_main_model.py metrics export.
    
    Args:
        save_dir: Training output directory
        results: Training results object
    """
    import csv
    
    if not hasattr(results, 'results_dict'):
        return
    
    metrics = results.results_dict
    csv_path = os.path.join(save_dir, "final_metrics_summary.csv")
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric Type', 'Metric Name', 'Value'])
            
            # Detection (Box) Metrics
            writer.writerow(['Detection', 'Precision', f"{metrics.get('metrics/precision(B)', 0):.4f}"])
            writer.writerow(['Detection', 'Recall', f"{metrics.get('metrics/recall(B)', 0):.4f}"])
            writer.writerow(['Detection', 'mAP@50', f"{metrics.get('metrics/mAP50(B)', 0):.4f}"])
            writer.writerow(['Detection', 'mAP@50-95', f"{metrics.get('metrics/mAP50-95(B)', 0):.4f}"])
            
            # Segmentation (Mask) Metrics
            writer.writerow(['Segmentation', 'mAP@50', f"{metrics.get('metrics/mAP50(M)', 0):.4f}"])
            writer.writerow(['Segmentation', 'mAP@50-95', f"{metrics.get('metrics/mAP50-95(M)', 0):.4f}"])
            
            # Loss Metrics
            writer.writerow(['Loss', 'Box Loss', f"{metrics.get('val/box_loss', 0):.4f}"])
            writer.writerow(['Loss', 'Seg Loss', f"{metrics.get('val/seg_loss', 0):.4f}"])
            writer.writerow(['Loss', 'Cls Loss', f"{metrics.get('val/cls_loss', 0):.4f}"])
            writer.writerow(['Loss', 'DFL Loss', f"{metrics.get('val/dfl_loss', 0):.4f}"])
            
            # Per-Class Metrics (if available)
            writer.writerow([])
            writer.writerow(['Per-Class Metrics', '', ''])
            
            # Try to extract per-class metrics from results
            if hasattr(results, 'names'):
                for cls_id, cls_name in results.names.items():
                    writer.writerow(['Class', cls_name, f'ID: {cls_id}'])
        
        print(f"  Exported final metrics summary: {csv_path}")
    
    except Exception as e:
        print(f"  Warning: Could not export metrics summary: {e}")


def print_training_summary(results, save_dir, config):
    """Print comprehensive training summary."""
    print("\n" + "=" * 90)
    print("                        SEGMENTATION TRAINING COMPLETE                        ")
    print("=" * 90)
    
    # Paths
    best_model = os.path.join(save_dir, "weights", "best.pt")
    last_model = os.path.join(save_dir, "weights", "last.pt")
    
    print(f"\n📁 Output Directory:")
    print(f"   {save_dir}")
    
    print(f"\n🏆 Trained Models:")
    print(f"   Best:  {best_model}")
    print(f"   Last:  {last_model}")
    
    # Final metrics
    if results and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📊 Final Validation Metrics:")
        
        # Detection metrics
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        # Segmentation metrics
        mask_map50 = metrics.get('metrics/mAP50(M)', 0)
        mask_map50_95 = metrics.get('metrics/mAP50-95(M)', 0)
        
        print(f"\n   Detection (Bounding Box):")
        print(f"   ┌─────────────┬──────────┐")
        print(f"   │ Metric      │  Value   │")
        print(f"   ├─────────────┼──────────┤")
        print(f"   │ Precision   │  {precision:.4f}  │")
        print(f"   │ Recall      │  {recall:.4f}  │")
        print(f"   │ mAP@50      │  {map50:.4f}  │")
        print(f"   │ mAP@50-95   │  {map50_95:.4f}  │")
        print(f"   └─────────────┴──────────┘")
        
        print(f"\n   Segmentation (Masks):")
        print(f"   ┌─────────────┬──────────┐")
        print(f"   │ Metric      │  Value   │")
        print(f"   ├─────────────┼──────────┤")
        print(f"   │ mAP@50      │  {mask_map50:.4f}  │")
        print(f"   │ mAP@50-95   │  {mask_map50_95:.4f}  │")
        print(f"   └─────────────┴──────────┘")
    
    # CSV logs
    print(f"\n📈 Training Metrics:")
    
    # Per-epoch metrics
    csv_path = os.path.join(save_dir, "training_metrics.csv")
    if os.path.exists(csv_path):
        print(f"   Per-Epoch CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Total epochs: {len(df)}")
        if len(df) > 0:
            best_epoch = df.loc[df['mask_mAP50-95'].idxmax()]
            print(f"   Best epoch: {int(best_epoch['epoch'])} "
                  f"(Mask mAP@50-95: {best_epoch['mask_mAP50-95']:.4f})")
    
    # Final metrics summary
    summary_csv = os.path.join(save_dir, "final_metrics_summary.csv")
    if os.path.exists(summary_csv):
        print(f"   Final Summary: {summary_csv}")
    
    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Image size: {config.IMAGE_SIZE}")
    print(f"   Frozen layers: {config.FREEZE_LAYERS}")
    
    print("\n" + "=" * 90)


def verify_segmentation_dataset(data_yaml):
    """Verify segmentation dataset structure."""
    if not os.path.exists(data_yaml):
        print(f"❌ Error: Dataset YAML not found at {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    required_fields = ['train', 'val', 'names']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        print(f"❌ Error: Missing required fields: {missing_fields}")
        return False
    
    # Get dataset directory
    dataset_dir = os.path.dirname(data_yaml)
    
    # Check images and masks directories
    train_images = os.path.join(dataset_dir, 'images', 'train')
    train_masks = os.path.join(dataset_dir, 'masks', 'train')
    val_images = os.path.join(dataset_dir, 'images', 'val')
    val_masks = os.path.join(dataset_dir, 'masks', 'val')
    
    # Count files
    n_train_images = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
    n_train_masks = len(os.listdir(train_masks)) if os.path.exists(train_masks) else 0
    n_val_images = len(os.listdir(val_images)) if os.path.exists(val_images) else 0
    n_val_masks = len(os.listdir(val_masks)) if os.path.exists(val_masks) else 0
    
    print(f"  ✓ Dataset verified")
    print(f"    Classes: {list(data['names'].values())}")
    print(f"    Train: {n_train_images} images, {n_train_masks} masks")
    print(f"    Val: {n_val_images} images, {n_val_masks} masks")
    
    if n_train_images != n_train_masks:
        print(f"    ⚠️  WARNING: Train images != masks")
    if n_val_images != n_val_masks:
        print(f"    ⚠️  WARNING: Val images != masks")
    
    return True


def create_training_callbacks(csv_log_path):
    """Create callbacks for logging during training."""
    def on_fit_epoch_end(trainer):
        """Log metrics after each epoch."""
        try:
            epoch = trainer.epoch + 1
            
            # Get learning rate safely
            lr = 0
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                if hasattr(trainer.optimizer, 'param_groups') and len(trainer.optimizer.param_groups) > 0:
                    lr = trainer.optimizer.param_groups[0].get('lr', 0)
            
            # Get training losses from trainer.tloss (tensor with [box, seg, cls, dfl])
            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                tloss = trainer.tloss
                if len(tloss) >= 4:
                    train_box = float(tloss[0])
                    train_seg = float(tloss[1])
                    train_cls = float(tloss[2])
                    train_dfl = float(tloss[3])
                else:
                    train_box = train_seg = train_cls = train_dfl = 0.0
            else:
                train_box = train_seg = train_cls = train_dfl = 0.0
            
            # Get validation metrics
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            
            # Extract validation losses and metrics
            val_box = metrics.get('val/box_loss', 0)
            val_seg = metrics.get('val/seg_loss', 0)
            val_cls = metrics.get('val/cls_loss', 0)
            val_dfl = metrics.get('val/dfl_loss', 0)
            
            precision = metrics.get('metrics/precision(B)', 0)
            recall = metrics.get('metrics/recall(B)', 0)
            mask_map50 = metrics.get('metrics/mAP50(M)', 0)
            mask_map50_95 = metrics.get('metrics/mAP50-95(M)', 0)
            
            # Log to CSV
            if metrics:
                log_metrics_to_csv(csv_log_path, epoch, metrics, lr)
            
            # Print progress with both train and val losses
            train_total = train_box + train_seg
            val_total = val_box + val_seg
            
            print(f"  {epoch:3d} │ Train:{train_total:6.4f} ({train_box:.3f}+{train_seg:.3f}) │ "
                  f"Val:{val_total:6.4f} ({val_box:.3f}+{val_seg:.3f}) │ "
                  f"P:{precision:.3f} R:{recall:.3f} │ "
                  f"M50:{mask_map50:.4f} M:{mask_map50_95:.4f} │ LR:{lr:.6f}")
        except Exception as e:
            if epoch % 10 == 0:
                print(f"  Warning: Callback error at epoch {epoch}: {str(e)[:50]}")
    
    return {'on_fit_epoch_end': on_fit_epoch_end}


def convert_masks_to_yolo_format(dataset_dir):
    """
    Convert grayscale masks to YOLO segmentation format (polygon annotations).
    YOLO expects .txt files with normalized polygon coordinates.
    
    Note: This is a simplified conversion. For production, consider using
    proper contour detection and polygon simplification.
    """
    print("\n  Converting masks to YOLO format...")
    
    import cv2
    
    for split in ['train', 'val']:
        images_dir = os.path.join(dataset_dir, 'images', split)
        masks_dir = os.path.join(dataset_dir, 'masks', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        
        os.makedirs(labels_dir, exist_ok=True)
        
        if not os.path.exists(masks_dir):
            continue
        
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        
        for mask_file in mask_files:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue
            
            h, w = mask.shape
            
            # Create label file
            label_file = mask_file.replace('.png', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'w') as f:
                # Process each class (1=stamp, 2=signature)
                for class_id in [1, 2]:
                    # Get mask for this class
                    class_mask = (mask == class_id).astype(np.uint8)
                    
                    if class_mask.sum() == 0:
                        continue
                    
                    # Find contours
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Skip very small contours
                        if len(contour) < 3:
                            continue
                        
                        # Simplify contour
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(approx) < 3:
                            continue
                        
                        # Normalize coordinates and flatten
                        normalized = approx.reshape(-1, 2).astype(float)
                        normalized[:, 0] /= w  # Normalize x
                        normalized[:, 1] /= h  # Normalize y
                        
                        # Clip to [0, 1]
                        normalized = np.clip(normalized, 0, 1)
                        
                        # Write to file: class_id x1 y1 x2 y2 ... xn yn
                        # YOLO uses class_id - 1 (0-indexed), but our masks use 1,2
                        # So stamp(1) -> 0, signature(2) -> 1
                        yolo_class_id = class_id - 1
                        coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized])
                        f.write(f"{yolo_class_id} {coords}\n")
    
    print(f"  ✓ Mask conversion complete")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_yolov11_seg(config):
    """Main training function for YOLOv11-seg."""
    
    print("=" * 80)
    print("YOLOv11-seg Training for Stamp & Signature Segmentation")
    print("=" * 80)
    
    # Verify dataset
    print(f"\n[1/6] Verifying dataset...")
    if not verify_segmentation_dataset(config.DATA_YAML):
        return
    
    # Convert masks to YOLO format
    print(f"\n[2/6] Converting masks to YOLO segmentation format...")
    dataset_dir = os.path.dirname(config.DATA_YAML)
    convert_masks_to_yolo_format(dataset_dir)
    
    # Initialize model
    print(f"\n[3/6] Loading YOLOv11n-seg model...")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Freeze layers: {config.FREEZE_LAYERS}")
    
    import warnings
    warnings.filterwarnings('ignore')
    
    model = YOLO(config.MODEL_NAME)
    print("  ✓ Model loaded")
    
    # Setup output directory
    save_dir = os.path.join(config.PROJECT_DIR, config.RUN_NAME)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize CSV logging
    csv_log_path = os.path.join(save_dir, "training_metrics.csv")
    initialize_csv_log(csv_log_path)
    
    # Setup callbacks
    callbacks = create_training_callbacks(csv_log_path)
    model.add_callback('on_fit_epoch_end', callbacks['on_fit_epoch_end'])
    
    # Prepare training arguments
    print(f"\n[4/6] Preparing training configuration...")
    train_args = {
        # Data
        'data': config.DATA_YAML,
        'task': 'segment',  # Explicitly set segmentation task
        
        # Training
        'epochs': config.EPOCHS,
        'batch': config.BATCH_SIZE,
        'imgsz': config.IMAGE_SIZE,
        'lr0': config.LEARNING_RATE,
        'device': config.DEVICE,
        'workers': config.WORKERS,
        
        # Augmentation
        'augment': config.AUGMENT,
        'mosaic': config.MOSAIC,
        'mixup': config.MIXUP,
        'copy_paste': config.COPY_PASTE,
        'flipud': config.FLIPUD,
        'fliplr': config.FLIPLR,
        'degrees': config.DEGREES,
        'translate': config.TRANSLATE,
        'scale': config.SCALE,
        'hsv_h': config.HSV_H,
        'hsv_s': config.HSV_S,
        'hsv_v': config.HSV_V,
        'close_mosaic': config.CLOSE_MOSAIC,
        
        # Optimization
        'optimizer': config.OPTIMIZER,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'lrf': config.LRF,  # Final OneCycleLR learning rate (lr0 * lrf)
        'weight_decay': config.WEIGHT_DECAY,
        'patience': config.PATIENCE,
        'amp': config.AMP,
        'dropout': config.DROPOUT,  # Dropout for regularization
        'label_smoothing': config.LABEL_SMOOTHING,  # Label smoothing
        'nbs': config.NBS,  # Nominal batch size for gradient accumulation
        'max_det': config.MAX_DET,
        
        # Fine-tuning
        'freeze': config.FREEZE_LAYERS if config.FREEZE_LAYERS > 0 else None,
        
        # Segmentation-specific
        'overlap_mask': config.OVERLAP_MASK,
        'mask_ratio': config.MASK_RATIO,
        
        # Checkpointing
        'project': config.PROJECT_DIR,
        'name': config.RUN_NAME,
        'save': True,
        'save_period': config.SAVE_PERIOD,
        
        # Logging
        'verbose': config.VERBOSE,
        'plots': config.PLOTS,
        'exist_ok': True,
    }
    
    print(f"  Configuration:")
    print(f"    Task: Segmentation")
    print(f"    Epochs: {config.EPOCHS}")
    print(f"    Batch size: {config.BATCH_SIZE}")
    print(f"    Image size: {config.IMAGE_SIZE}")
    print(f"    Learning rate: {config.LEARNING_RATE} (warmup {config.WARMUP_EPOCHS} epochs)")
    print(f"    LR scheduler: Cosine decay to {config.LEARNING_RATE * config.LRF:.6f}")
    print(f"    Optimizer: {config.OPTIMIZER}")
    print(f"    Frozen layers: {config.FREEZE_LAYERS}")
    print(f"    Weight decay: {config.WEIGHT_DECAY}")
    print(f"    Mixed precision: {config.AMP}")
    print(f"    Batch accumulation: {config.NBS // config.BATCH_SIZE} steps")
    if config.DROPOUT > 0:
        print(f"    Dropout: {config.DROPOUT}")
    if config.LABEL_SMOOTHING > 0:
        print(f"    Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"    Early stopping patience: {config.PATIENCE}")
    print(f"    Save period: {config.SAVE_PERIOD} epochs")
    
    # Start training
    print(f"\n[5/6] Starting training...")
    print(f"  Output: {save_dir}")
    print("=" * 150)
    print(f"  Ep  │ Train Loss (total|box+seg) │ Val Loss (total|box+seg)   │ Metrics (P|R) │ Mask mAP (50|50-95) │ Learning Rate")
    print("-" * 150)
    
    try:
        # Validate configuration
        print("\n  Validating configuration...")
        print("  Starting training loop...\n")
        
        results = model.train(**train_args)
        
        # Training completed
        print(f"\n[6/6] Processing results...")
        
        # Log final metrics to training_metrics.csv
        if hasattr(results, 'results_dict'):
            log_metrics_to_csv(csv_log_path, config.EPOCHS, 
                             results.results_dict, config.LEARNING_RATE)
        
        # Export final metrics summary (similar to inference metrics.csv)
        export_final_metrics_summary(save_dir, results)
        
        # Print summary
        print_training_summary(results, save_dir, config)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    
    # Create configuration
    config = SegTrainingConfig()
    
    # Check CUDA
    print("")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  CPU mode (training will be slower)")
    
    # Run training
    train_yolov11_seg(config)


if __name__ == "__main__":
    main()
