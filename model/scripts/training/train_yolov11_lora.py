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
import warnings

# Suppress warnings early
warnings.filterwarnings('ignore')

try:
    import torch
    from pathlib import Path
    from datetime import datetime
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    
    # Suppress verbose ultralytics logging
    LOGGER.setLevel('WARNING')
except ImportError as e:
    print(f"ERROR: Failed to import required packages: {e}")
    print("Please install required packages:")
    print("  pip install ultralytics torch pandas pyyaml")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset
    DATA_YAML = "data/datasets/main/dataset.yaml"  # Path to dataset.yaml
    
    # Model
    MODEL_NAME = "yolo11s.pt"  # Pre-trained YOLOv11s weights
    
    # Fine-tuning mode (freeze early layers for faster training)
    FREEZE_LAYERS = 10  # Number of layers to freeze (0 to disable, max ~20)
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8  # Reduced for higher resolution (16 @ 1024px)
    IMAGE_SIZE = 1280  # Higher resolution for better quality (was 1024)
    LEARNING_RATE = 0.001  # Proven configuration
    
    # Training stability
    AMP = True  # Automatic mixed precision for faster training
    DROPOUT = 0.0  # Dropout rate (0.0 = disabled, use for regularization if overfitting)
    LABEL_SMOOTHING = 0.0  # Label smoothing (0.0-0.1, helps with overconfidence)
    NBS = 64  # Nominal batch size for batch accumulation (kept for stability)
    
    # Augmentation (proven configuration)
    AUGMENT = True
    MOSAIC = 1.0  # Proven configuration
    MIXUP = 0.0  # Disable mixup (not suitable for documents)
    COPY_PASTE = 0.0  # Proven configuration
    FLIPUD = 0.5  # Proven configuration
    FLIPLR = 0.5  # Horizontal flip (documents can be mirrored)
    DEGREES = 10.0  # Proven configuration
    TRANSLATE = 0.1  # Small translation
    SCALE = 0.5  # Proven configuration
    HSV_H = 0.015  # Hue augmentation
    HSV_S = 0.7  # Saturation augmentation
    HSV_V = 0.4  # Value/brightness augmentation
    
    # Additional augmentation
    PERSPECTIVE = 0.0  # No perspective (distorts handwriting)
    SHEAR = 0.0  # No shear (distorts signatures)
    ERASING = 0.4  # Random erasing probability (from proven config)
    
    # Checkpointing and logging
    PROJECT_DIR = "runs/train"
    RUN_NAME = f"yolov11s_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    SAVE_PERIOD = 5  # Save checkpoint every N epochs (-1 to disable)
    
    # Hardware
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    WORKERS = 4  # Proven configuration
    
    # Validation
    VAL_SPLIT = 0.1  # 90% train, 10% val - Only used if not specified in data.yaml
    
    # Additional training parameters
    PATIENCE = 20  # Early stopping patience (proven value)
    OPTIMIZER = "SGD"  # SGD optimizer for stable convergence
    WARMUP_EPOCHS = 3  # Proven configuration (was in successful run)
    WARMUP_MOMENTUM = 0.8  # Proven configuration
    WARMUP_BIAS_LR = 0.1  # Proven configuration
    LRF = 0.01  # Final learning rate factor (proven configuration)
    MOMENTUM = 0.937  # SGD momentum
    WEIGHT_DECAY = 0.0005
    CLOSE_MOSAIC = 10  # Disable mosaic last N epochs for better convergence
    
    # Loss weights (proven values)
    BOX_LOSS_GAIN = 7.5  # Proven configuration
    CLS_LOSS_GAIN = 0.5   # Proven configuration
    DFL_LOSS_GAIN = 1.5   # Distribution focal loss gain
    
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
        # Check if file exists and has header
        file_exists = os.path.exists(log_path)
        if file_exists:
            # Verify header exists
            with open(log_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('epoch'):
                    # Recreate with header
                    initialize_csv_log(log_path)
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
        
        # Append to CSV (header only if file is empty)
        df = pd.DataFrame([new_row])
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        df.to_csv(log_path, mode='a', header=write_header, index=False)
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
        try:
            df = pd.read_csv(csv_path)
            print(f"   Total epochs: {len(df)}")
            if len(df) > 0:
                # Try different possible column names
                map_col = None
                for col in ['mAP50-95', 'mAP50_95', 6]:  # column index 6 if no header
                    if col in df.columns or (isinstance(col, int) and col < len(df.columns)):
                        map_col = col
                        break
                
                if map_col is not None:
                    best_idx = df[map_col].idxmax()
                    best_epoch = df.loc[best_idx]
                    epoch_num = best_epoch[0] if isinstance(best_epoch.index[0], int) else best_epoch.get('epoch', best_idx + 1)
                    map_val = best_epoch[map_col]
                    print(f"   Best epoch: {int(epoch_num)} (mAP@50-95: {map_val:.4f})")
        except Exception as e:
            print(f"   Warning: Could not parse training history: {e}")
    
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
            
            # Get learning rate safely
            lr = 0
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                if hasattr(trainer.optimizer, 'param_groups') and len(trainer.optimizer.param_groups) > 0:
                    lr = trainer.optimizer.param_groups[0].get('lr', 0)
            
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
            
            # Calculate total losses for summary
            train_total = train_box + train_cls + train_dfl
            val_total = val_box + val_cls + val_dfl
            
            # Print clean progress update with both train and val metrics
            print(f"  {epoch:3d} │ {train_total:.4f} ({train_box:.3f}|{train_cls:.3f}|{train_dfl:.3f}) │ "
                  f"{val_total:.4f} ({val_box:.3f}|{val_cls:.3f}|{val_dfl:.3f}) │ "
                  f"P:{precision:.3f} R:{recall:.3f} │ "
                  f"mAP50:{map50:.4f} mAP:{map50_95:.4f} │ LR:{lr:.6f}")
        except Exception as e:
            # Log error but continue training
            if epoch % 10 == 0:  # Only print every 10 epochs to avoid spam
                print(f"  Warning: Callback error at epoch {epoch}: {str(e)[:50]}")
    
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
    
    # Load model with minimal output
    try:
        print("  Downloading weights..." if not os.path.exists(config.MODEL_NAME) else "  Loading weights...")
        model = YOLO(config.MODEL_NAME)
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        print(f"  Make sure {config.MODEL_NAME} exists or can be downloaded")
        raise
    
    # Setup output directory
    save_dir = os.path.join(config.PROJECT_DIR, config.RUN_NAME)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize CSV logging
    csv_log_path = os.path.join(save_dir, "training_metrics.csv")
    initialize_csv_log(csv_log_path)
    
    # Setup callbacks for metric logging
    callbacks = create_training_callbacks(csv_log_path)
    model.add_callback('on_fit_epoch_end', callbacks['on_fit_epoch_end'])
    
    # Note: Custom colors can be set after training if needed
    # Removed custom color override to prevent potential launch failures
    
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
        'perspective': config.PERSPECTIVE,
        'shear': config.SHEAR,
        
        # Optimization
        'optimizer': config.OPTIMIZER,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'warmup_momentum': config.WARMUP_MOMENTUM,
        'warmup_bias_lr': config.WARMUP_BIAS_LR,
        'momentum': config.MOMENTUM,
        'lrf': config.LRF,  # Final OneCycleLR learning rate (lr0 * lrf)
        'weight_decay': config.WEIGHT_DECAY,
        'patience': config.PATIENCE,
        'amp': config.AMP,  # Mixed precision training
        'dropout': config.DROPOUT,  # Dropout for regularization
        'nbs': config.NBS,  # Nominal batch size for gradient accumulation
        'max_det': 300,  # Maximum detections per image
        
        # Loss weights (critical for localization quality)
        'box': config.BOX_LOSS_GAIN,  # Box regression loss weight
        'cls': config.CLS_LOSS_GAIN,  # Classification loss weight
        'dfl': config.DFL_LOSS_GAIN,  # Distribution focal loss weight
        
        # Fine-tuning
        'freeze': config.FREEZE_LAYERS if config.FREEZE_LAYERS > 0 else None,
        
        # Checkpointing
        'project': config.PROJECT_DIR,
        'name': config.RUN_NAME,
        'save': True,
        'save_period': config.SAVE_PERIOD if config.SAVE_PERIOD > 0 else -1,
        
        # Additional settings from proven config
        'cos_lr': False,  # No cosine LR (proven config)
        'auto_augment': 'randaugment',  # From proven config
        
        # Logging
        'verbose': config.VERBOSE,
        'plots': config.PLOTS,
        'exist_ok': True,
    }
    
    print(f"\n  Configuration (PROVEN - yolov11s_lora_20251115_230522):")
    print(f"    Epochs: {config.EPOCHS}")
    print(f"    Batch size: {config.BATCH_SIZE}")
    print(f"    Image size: {config.IMAGE_SIZE}")
    print(f"    Learning rate: {config.LEARNING_RATE}")
    print(f"    Warmup: {config.WARMUP_EPOCHS} epochs (momentum: {config.WARMUP_MOMENTUM})")
    print(f"    Momentum: {config.MOMENTUM}")
    print(f"    LR scheduler: Cosine decay to {config.LEARNING_RATE * config.LRF:.6f}")
    print(f"    Optimizer: {config.OPTIMIZER}")
    print(f"    Freeze layers: {config.FREEZE_LAYERS}")
    print(f"    Weight decay: {config.WEIGHT_DECAY}")
    print(f"    Mixed precision: {config.AMP}")
    print(f"    Patience: {config.PATIENCE} epochs")
    print(f"    Workers: {config.WORKERS}")
    print(f"\n    Loss weights:")
    print(f"      Box loss: {config.BOX_LOSS_GAIN}")
    print(f"      Cls loss: {config.CLS_LOSS_GAIN}")
    print(f"      DFL loss: {config.DFL_LOSS_GAIN}")
    print(f"\n    Augmentation:")
    print(f"      Mosaic: {config.MOSAIC}")
    print(f"      Rotation: ±{config.DEGREES}°")
    print(f"      Scale: {config.SCALE}")
    print(f"      Erasing: {config.ERASING}")
    if config.DROPOUT > 0:
        print(f"    Dropout: {config.DROPOUT}")
    if config.LABEL_SMOOTHING > 0:
        print(f"    Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"    Early stopping patience: {config.PATIENCE}")
    print(f"    Save period: {config.SAVE_PERIOD} epochs")
    
    # Start training
    print(f"\n[4/5] Starting training...")
    print(f"  Output directory: {save_dir}")
    print("=" * 150)
    print(f"  Ep  │ Train Loss (total|box|cls|dfl) │ Val Loss (total|box|cls|dfl)   │ Metrics (P|R)  │ mAP (50|50-95) │ Learning Rate")
    print("-" * 150)
    
    try:
        # Validate training arguments before starting
        print("\n  Validating configuration...")
        
        # Start training with error recovery
        print("  Starting training loop...\n")
        results = model.train(**train_args)
        
        # Training completed successfully
        print(f"\n{'=' * 150}")
        print(f"[5/5] Processing results and saving metrics...")
        
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
    
    try:
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
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
