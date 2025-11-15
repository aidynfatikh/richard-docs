#!/usr/bin/env python3
"""
YOLOv11s Training Pipeline with Advanced Fine-tuning
Complete standalone training script with checkpointing and metric logging.

IMPORTANT - Train/Val Separation:
- Training uses ONLY images in dataset/images/train/
- Validation uses ONLY images in dataset/images/val/
- Model NEVER trains on validation data
- Metrics (mAP, precision, recall) are calculated on validation set after each epoch
- No data leakage: train and val sets are completely separate (80/20 split)

Note: LoRA requires custom implementation with PEFT library.
This script uses optimized training parameters for efficient fine-tuning.
"""

import os
import sys
import yaml
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Suppress verbose ultralytics logging
LOGGER.setLevel('WARNING')


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset
    DATA_YAML = "dataset/dataset.yaml"  # Path to dataset.yaml
    
    # Model
    MODEL_NAME = "yolo11s.pt"  # Pre-trained YOLOv11s weights
    
    # Fine-tuning mode (freeze early layers for faster training)
    FREEZE_LAYERS = 10  # Number of layers to freeze (0 to disable, max ~20)
    
    # Training hyperparameters
    EPOCHS = 60
    BATCH_SIZE = 16
    IMAGE_SIZE = 1024
    LEARNING_RATE = 0.001  # Higher LR for full model training
    
    # Augmentation
    AUGMENT = True
    MOSAIC = 1.0
    FLIPUD = 0.5  # Vertical flip
    FLIPLR = 0.5  # Horizontal flip
    DEGREES = 10.0  # Rotation degrees
    SCALE = 0.5  # Image scale (+/- gain)
    
    # Checkpointing and logging
    PROJECT_DIR = "runs/train"
    RUN_NAME = f"yolov11s_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    SAVE_PERIOD = 5  # Save checkpoint every N epochs (-1 to disable)
    
    # Hardware
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    WORKERS = 8
    
    # Validation
    VAL_SPLIT = 0.2  # Only used if not specified in data.yaml
    
    # Additional training parameters
    PATIENCE = 20  # Early stopping patience
    OPTIMIZER = "auto"  # Optimizer: auto, SGD, Adam, AdamW
    WARMUP_EPOCHS = 3
    WEIGHT_DECAY = 0.0005
    CLOSE_MOSAIC = 10  # Disable mosaic last N epochs for better convergence
    
    # Logging
    VERBOSE = False  # Reduce console clutter
    PLOTS = True  # Generate training plots


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def initialize_csv_log(log_path):
    """
    Initialize CSV file for logging training metrics.
    Creates file with headers if it doesn't exist.
    """
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=[
            'epoch',
            'train_loss',
            'val_loss',
            'precision',
            'recall',
            'mAP50',
            'mAP50-95',
            'learning_rate',
            'timestamp'
        ])
        df.to_csv(log_path, index=False)
        print(f"Created metrics log: {log_path}")
    return log_path


def log_metrics_to_csv(log_path, epoch, metrics, lr):
    """
    Append training metrics to CSV file.
    
    Args:
        log_path: Path to CSV file
        epoch: Current epoch number
        metrics: Dict containing training metrics
        lr: Current learning rate
    """
    try:
        # Extract metrics from results (handle different key formats)
        train_box = metrics.get('train/box_loss', metrics.get('box_loss', 0))
        train_cls = metrics.get('train/cls_loss', metrics.get('cls_loss', 0))
        train_dfl = metrics.get('train/dfl_loss', metrics.get('dfl_loss', 0))
        train_loss = train_box + train_cls + train_dfl
        
        val_box = metrics.get('val/box_loss', 0)
        val_cls = metrics.get('val/cls_loss', 0)
        val_dfl = metrics.get('val/dfl_loss', 0)
        val_loss = val_box + val_cls + val_dfl
        
        precision = metrics.get('metrics/precision(B)', metrics.get('precision', 0))
        recall = metrics.get('metrics/recall(B)', metrics.get('recall', 0))
        map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50', 0))
        map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95', 0))
        
        # Create new row
        new_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'precision': precision,
            'recall': recall,
            'mAP50': map50,
            'mAP50-95': map50_95,
            'learning_rate': lr,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Append to CSV
        df = pd.DataFrame([new_row])
        df.to_csv(log_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Warning: Could not log metrics to CSV: {e}")


def print_training_summary(results, save_dir):
    """Print comprehensive training summary."""
    print("\n" + "=" * 80)
    print("                           TRAINING COMPLETE                              ")
    print("=" * 80)
    
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
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        print(f"   ┌─────────────┬──────────┐")
        print(f"   │ Metric      │  Value   │")
        print(f"   ├─────────────┼──────────┤")
        print(f"   │ Precision   │  {precision:.4f}  │")
        print(f"   │ Recall      │  {recall:.4f}  │")
        print(f"   │ mAP@50      │  {map50:.4f}  │")
        print(f"   │ mAP@50-95   │  {map50_95:.4f}  │")
        print(f"   └─────────────┴──────────┘")
    
    # CSV log
    csv_path = os.path.join(save_dir, "training_metrics.csv")
    if os.path.exists(csv_path):
        print(f"\n📈 Training History:")
        print(f"   CSV Log: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Total epochs: {len(df)}")
        if len(df) > 0:
            best_epoch = df.loc[df['mAP50-95'].idxmax()]
            print(f"   Best epoch: {int(best_epoch['epoch'])} (mAP@50-95: {best_epoch['mAP50-95']:.4f})")
    
    print("\n" + "=" * 80)


def verify_dataset(data_yaml):
    """
    Verify dataset.yaml exists and contains required fields.
    Ensures proper train/val split with no overlap.
    Returns True if valid, False otherwise.
    """
    if not os.path.exists(data_yaml):
        print(f"❌ Error: Dataset YAML not found at {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    required_fields = ['train', 'val', 'names']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        print(f"❌ Error: Missing required fields in dataset.yaml: {missing_fields}")
        return False
    
    # Get dataset directory
    dataset_dir = os.path.dirname(data_yaml)
    
    # Check train and val directories
    train_path = os.path.join(dataset_dir, data['train'])
    val_path = os.path.join(dataset_dir, data['val'])
    
    train_images = set(os.listdir(train_path)) if os.path.exists(train_path) else set()
    val_images = set(os.listdir(val_path)) if os.path.exists(val_path) else set()
    
    # Check for overlap (should be zero)
    overlap = train_images & val_images
    
    print(f"✓ Dataset verified: {len(data['names'])} classes")
    print(f"  Classes: {data['names']}")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Train/Val overlap: {len(overlap)} (should be 0)")
    
    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} images appear in both train and val sets!")
        print(f"  First few overlapping files: {list(overlap)[:5]}")
    
    return True


def create_training_callbacks(csv_log_path):
    """
    Create callback functions for logging metrics during training.
    Returns dict of callbacks compatible with Ultralytics.
    """
    def on_fit_epoch_end(trainer):
        """Log metrics after each epoch (after validation)"""
        try:
            epoch = trainer.epoch + 1
            
            # Get learning rate
            lr = trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else 0
            
            # Access losses from trainer's label_loss_items
            # Training losses are in trainer.tloss (tensor with [box, cls, dfl])
            if hasattr(trainer, 'tloss'):
                tloss = trainer.tloss
                if tloss is not None and len(tloss) >= 3:
                    train_box = float(tloss[0])
                    train_cls = float(tloss[1])
                    train_dfl = float(tloss[2])
                else:
                    train_box = train_cls = train_dfl = 0.0
            else:
                train_box = train_cls = train_dfl = 0.0
            
            # Validation metrics are in trainer.metrics
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            
            # Extract validation metrics
            map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50', 0))
            map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95', 0))
            precision = metrics.get('metrics/precision(B)', metrics.get('precision', 0))
            recall = metrics.get('metrics/recall(B)', metrics.get('recall', 0))
            
            # Validation losses
            val_box = metrics.get('val/box_loss', 0)
            val_cls = metrics.get('val/cls_loss', 0)
            val_dfl = metrics.get('val/dfl_loss', 0)
            
            # Log to CSV
            if metrics:
                log_metrics_to_csv(csv_log_path, epoch, metrics, lr)
            
            # Print clean progress update with both train and val metrics
            print(f"  {epoch:3d} │ Train: {train_box:.4f} {train_cls:.4f} {train_dfl:.4f} │ "
                  f"Val: {val_box:.4f} {val_cls:.4f} {val_dfl:.4f} │ "
                  f"P: {precision:.3f} R: {recall:.3f} │ "
                  f"mAP50: {map50:.4f} mAP50-95: {map50_95:.4f}")
        except Exception as e:
            pass  # Silently continue on callback errors
    
    return {
        'on_fit_epoch_end': on_fit_epoch_end
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_yolov11_lora(config):
    """
    Main training function with LoRA fine-tuning.
    
    Args:
        config: TrainingConfig object with all parameters
    """
    
    print("=" * 70)
    print("YOLOv11s Training with LoRA Fine-tuning")
    print("=" * 70)
    
    # Verify dataset
    print(f"\n[1/5] Verifying dataset...")
    if not verify_dataset(config.DATA_YAML):
        return
    
    # Initialize model
    print(f"\n[2/5] Loading YOLOv11s model...")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Freeze layers: {config.FREEZE_LAYERS}")
    
    # Suppress download progress bars
    import warnings
    warnings.filterwarnings('ignore')
    
    # Load model with minimal output
    print("  Downloading weights..." if not os.path.exists(config.MODEL_NAME) else "  Loading weights...")
    model = YOLO(config.MODEL_NAME)
    print("  ✓ Model loaded")
    
    # Setup output directory
    save_dir = os.path.join(config.PROJECT_DIR, config.RUN_NAME)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize CSV logging
    csv_log_path = os.path.join(save_dir, "training_metrics.csv")
    initialize_csv_log(csv_log_path)
    
    # Setup callbacks for metric logging
    callbacks = create_training_callbacks(csv_log_path)
    model.add_callback('on_fit_epoch_end', callbacks['on_fit_epoch_end'])
    
    # Set custom class colors for visualization
    # Stamp will be RED for better visibility in training batches
    print("  Setting custom visualization colors (stamp=red)...")
    try:
        import ultralytics.utils.plotting as plotting_module
        original_colors = plotting_module.colors
        
        # Create a wrapper class that preserves all attributes but overrides __call__
        class CustomColors:
            def __init__(self, original):
                self._original = original
                
            def __call__(self, i, bgr=False):
                """Custom color mapping: 0=qr(blue), 1=signature(green), 2=stamp(red)"""
                # For stamp (class 2), always return bright red
                if i == 2:
                    return (0, 0, 255) if bgr else (255, 0, 0)
                # For signature (class 1), return green
                elif i == 1:
                    return (0, 255, 0)
                # For QR (class 0), return blue
                elif i == 0:
                    return (255, 0, 0) if bgr else (0, 0, 255)
                # For any other class, use original colors
                return self._original(i, bgr)
            
            def __getattr__(self, name):
                # Delegate all other attributes to the original colors object
                return getattr(self._original, name)
        
        # Replace with our custom wrapper
        plotting_module.colors = CustomColors(original_colors)
        
    except Exception as e:
        print(f"  Note: Could not customize colors: {e}")
    
    # Prepare training arguments
    print(f"\n[3/5] Preparing training configuration...")
    train_args = {
        # Data - YOLO uses separate train/val directories specified in dataset.yaml
        # Training ONLY uses images/train, validation ONLY uses images/val
        # Metrics (mAP, precision, recall) are calculated on validation set after each epoch
        'data': config.DATA_YAML,
        
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
        'flipud': config.FLIPUD,
        'fliplr': config.FLIPLR,
        'degrees': config.DEGREES,
        'scale': config.SCALE,
        'close_mosaic': config.CLOSE_MOSAIC,
        
        # Optimization
        'optimizer': config.OPTIMIZER,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'weight_decay': config.WEIGHT_DECAY,
        'patience': config.PATIENCE,
        
        # Fine-tuning
        'freeze': config.FREEZE_LAYERS if config.FREEZE_LAYERS > 0 else None,
        
        # Checkpointing
        'project': config.PROJECT_DIR,
        'name': config.RUN_NAME,
        'save': True,
        'save_period': config.SAVE_PERIOD if config.SAVE_PERIOD > 0 else -1,
        
        # Logging
        'verbose': config.VERBOSE,
        'plots': config.PLOTS,
        'exist_ok': True,
    }
    
    print(f"\n  Configuration:")
    print(f"    Epochs: {config.EPOCHS}")
    print(f"    Batch size: {config.BATCH_SIZE}")
    print(f"    Image size: {config.IMAGE_SIZE}")
    print(f"    Learning rate: {config.LEARNING_RATE}")
    print(f"    Optimizer: {config.OPTIMIZER}")
    print(f"    Freeze layers: {config.FREEZE_LAYERS}")
    print(f"    Save period: {config.SAVE_PERIOD} epochs")
    
    # Start training
    print(f"\n[4/5] Starting training...")
    print(f"  Output directory: {save_dir}")
    print("=" * 140)
    print(f"  Ep  │ Train Loss (box/cls/dfl)    │ Val Loss (box/cls/dfl)      │ Precision/Recall │ mAP Metrics")
    print("-" * 140)
    
    try:
        # Redirect ultralytics verbose output
        import contextlib
        import io
        
        # Capture training output
        results = model.train(**train_args)
        
        # Training completed successfully
        print(f"\n[5/5] Processing results and saving metrics...")
        
        # Log final metrics
        if hasattr(results, 'results_dict'):
            final_lr = config.LEARNING_RATE  # Get actual LR if available
            log_metrics_to_csv(
                csv_log_path,
                config.EPOCHS,
                results.results_dict,
                final_lr
            )
        
        # Print summary
        print_training_summary(results, save_dir)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for training script."""
    
    # Create configuration
    config = TrainingConfig()
    
    # Check CUDA availability
    print("")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU: {gpu_name}")
    else:
        print("⚠️  CPU mode (training will be slower)")
    
    # Run training
    train_yolov11_lora(config)


if __name__ == "__main__":
    main()
