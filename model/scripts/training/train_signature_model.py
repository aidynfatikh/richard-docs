#!/usr/bin/env python3
"""
Signature-Only Detection Model Training Script
Trains a YOLOv11 model to detect only signatures using fine-tuning.
Based on the proven training configuration from train_yolov11_lora.py.
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

class SignatureTrainingConfig:
    """Training configuration for signature-only detection model"""
    
    # Dataset
    DATA_YAML = "data/datasets/signature/dataset.yaml"  # Path to signature dataset.yaml
    
    # Model - use lighter model for single-class detection
    MODEL_NAME = "yolo11n.pt"  # YOLOv11n (nano) - faster and lighter for single class
    
    # Fine-tuning mode (freeze early layers for faster training)
    FREEZE_LAYERS = 10  # Number of layers to freeze (0 to disable)
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 16  # Proven configuration
    IMAGE_SIZE = 1024  # Proven configuration
    LEARNING_RATE = 0.001  # Proven configuration
    
    # Training stability
    AMP = True  # Automatic mixed precision
    DROPOUT = 0.0  # Dropout rate
    LABEL_SMOOTHING = 0.0  # Label smoothing
    NBS = 64  # Nominal batch size
    
    # Augmentation (proven configuration)
    AUGMENT = True
    MOSAIC = 1.0  # Proven configuration
    MIXUP = 0.0  # Disable mixup
    COPY_PASTE = 0.0  # Proven configuration
    FLIPUD = 0.5  # Proven configuration
    FLIPLR = 0.5  # Horizontal flip
    DEGREES = 10.0  # Proven configuration
    TRANSLATE = 0.1  # Small translation
    SCALE = 0.5  # Proven configuration
    HSV_H = 0.015  # Hue augmentation
    HSV_S = 0.7  # Saturation augmentation
    HSV_V = 0.4  # Value/brightness augmentation
    
    # Additional augmentation
    PERSPECTIVE = 0.0  # No perspective
    SHEAR = 0.0  # No shear
    ERASING = 0.4  # Random erasing probability
    
    # Checkpointing and logging
    PROJECT_DIR = "runs/train"
    RUN_NAME = f"signature_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    SAVE_PERIOD = 5  # Save checkpoint every N epochs
    
    # Hardware
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    WORKERS = 4  # Proven configuration
    
    # Validation
    VAL_SPLIT = 0.1  # 90% train, 10% val
    
    # Additional training parameters
    PATIENCE = 20  # Early stopping patience
    OPTIMIZER = "SGD"  # SGD optimizer for stable convergence
    WARMUP_EPOCHS = 3  # Proven configuration
    WARMUP_MOMENTUM = 0.8  # Proven configuration
    WARMUP_BIAS_LR = 0.1  # Proven configuration
    LRF = 0.01  # Final learning rate factor
    MOMENTUM = 0.937  # SGD momentum
    WEIGHT_DECAY = 0.0005
    CLOSE_MOSAIC = 10  # Disable mosaic last N epochs
    
    # Loss weights (proven values)
    BOX_LOSS_GAIN = 7.5  # Proven configuration
    CLS_LOSS_GAIN = 0.5   # Proven configuration (less important for single class)
    DFL_LOSS_GAIN = 1.5   # Distribution focal loss gain
    
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
    """Append training metrics to CSV file."""
    try:
        file_exists = os.path.exists(log_path)
        if file_exists:
            with open(log_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('epoch'):
                    initialize_csv_log(log_path)
        
        # Extract metrics
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
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        df.to_csv(log_path, mode='a', header=write_header, index=False)
    except Exception as e:
        print(f"Warning: Could not log metrics to CSV: {e}")


def print_training_summary(results, save_dir):
    """Print comprehensive training summary."""
    print("\n" + "=" * 80)
    print("                    SIGNATURE MODEL TRAINING COMPLETE                     ")
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
                map_col = None
                for col in ['mAP50-95', 'mAP50_95', 6]:
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
    """Verify dataset.yaml exists and contains required fields."""
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
    
    # Check that we have exactly 1 class (signature)
    if len(data['names']) != 1 or data['names'][0] != 'signature':
        print(f"❌ Error: Expected 1 class 'signature', got: {data['names']}")
        return False
    
    # Get dataset directory
    dataset_dir = os.path.dirname(data_yaml)
    
    # Check train and val directories
    train_path = os.path.join(dataset_dir, data['train'])
    val_path = os.path.join(dataset_dir, data['val'])
    
    train_images = set(os.listdir(train_path)) if os.path.exists(train_path) else set()
    val_images = set(os.listdir(val_path)) if os.path.exists(val_path) else set()
    
    overlap = train_images & val_images
    
    print(f"✓ Signature dataset verified")
    print(f"  Classes: {data['names']}")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Train/Val overlap: {len(overlap)} (should be 0)")
    
    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} images appear in both train and val sets!")
    
    return True


def create_training_callbacks(csv_log_path):
    """Create callback functions for logging metrics during training."""
    def on_fit_epoch_end(trainer):
        """Log metrics after each epoch (after validation)"""
        try:
            epoch = trainer.epoch + 1
            
            # Get learning rate
            lr = 0
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                if hasattr(trainer.optimizer, 'param_groups') and len(trainer.optimizer.param_groups) > 0:
                    lr = trainer.optimizer.param_groups[0].get('lr', 0)
            
            # Get training losses
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
            
            # Get validation metrics
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            
            map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50', 0))
            map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95', 0))
            precision = metrics.get('metrics/precision(B)', metrics.get('precision', 0))
            recall = metrics.get('metrics/recall(B)', metrics.get('recall', 0))
            
            val_box = metrics.get('val/box_loss', 0)
            val_cls = metrics.get('val/cls_loss', 0)
            val_dfl = metrics.get('val/dfl_loss', 0)
            
            # Log to CSV
            if metrics:
                log_metrics_to_csv(csv_log_path, epoch, metrics, lr)
            
            # Calculate total losses
            train_total = train_box + train_cls + train_dfl
            val_total = val_box + val_cls + val_dfl
            
            # Print progress
            print(f"  {epoch:3d} │ {train_total:.4f} ({train_box:.3f}|{train_cls:.3f}|{train_dfl:.3f}) │ "
                  f"{val_total:.4f} ({val_box:.3f}|{val_cls:.3f}|{val_dfl:.3f}) │ "
                  f"P:{precision:.3f} R:{recall:.3f} │ "
                  f"mAP50:{map50:.4f} mAP:{map50_95:.4f} │ LR:{lr:.6f}")
        except Exception as e:
            if epoch % 10 == 0:
                print(f"  Warning: Callback error at epoch {epoch}: {str(e)[:50]}")
    
    return {
        'on_fit_epoch_end': on_fit_epoch_end
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_signature_model(config):
    """
    Main training function for signature-only detection model.
    
    Args:
        config: SignatureTrainingConfig object with all parameters
    """
    
    print("=" * 70)
    print("Signature-Only Detection Model Training")
    print("=" * 70)
    
    # Verify dataset
    print(f"\n[1/5] Verifying signature dataset...")
    if not verify_dataset(config.DATA_YAML):
        return
    
    # Initialize model
    print(f"\n[2/5] Loading {config.MODEL_NAME} model...")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Freeze layers: {config.FREEZE_LAYERS}")
    
    try:
        print("  Downloading weights..." if not os.path.exists(config.MODEL_NAME) else "  Loading weights...")
        model = YOLO(config.MODEL_NAME)
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        raise
    
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
    print(f"\n[3/5] Preparing training configuration...")
    train_args = {
        # Data
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
        'lrf': config.LRF,
        'weight_decay': config.WEIGHT_DECAY,
        'patience': config.PATIENCE,
        'amp': config.AMP,
        'dropout': config.DROPOUT,
        'nbs': config.NBS,
        'max_det': 300,
        
        # Loss weights
        'box': config.BOX_LOSS_GAIN,
        'cls': config.CLS_LOSS_GAIN,
        'dfl': config.DFL_LOSS_GAIN,
        
        # Fine-tuning
        'freeze': config.FREEZE_LAYERS if config.FREEZE_LAYERS > 0 else None,
        
        # Checkpointing
        'project': config.PROJECT_DIR,
        'name': config.RUN_NAME,
        'save': True,
        'save_period': config.SAVE_PERIOD if config.SAVE_PERIOD > 0 else -1,
        
        # Additional settings
        'cos_lr': False,
        'auto_augment': 'randaugment',
        
        # Logging
        'verbose': config.VERBOSE,
        'plots': config.PLOTS,
        'exist_ok': True,
    }
    
    print(f"\n  Configuration:")
    print(f"    Model: {config.MODEL_NAME} (optimized for single-class)")
    print(f"    Epochs: {config.EPOCHS}")
    print(f"    Batch size: {config.BATCH_SIZE}")
    print(f"    Image size: {config.IMAGE_SIZE}")
    print(f"    Learning rate: {config.LEARNING_RATE}")
    print(f"    Freeze layers: {config.FREEZE_LAYERS}")
    print(f"    Loss weights: Box={config.BOX_LOSS_GAIN}, Cls={config.CLS_LOSS_GAIN}, DFL={config.DFL_LOSS_GAIN}")
    
    # Start training
    print(f"\n[4/5] Starting training...")
    print(f"  Output directory: {save_dir}")
    print("=" * 150)
    print(f"  Ep  │ Train Loss (total|box|cls|dfl) │ Val Loss (total|box|cls|dfl)   │ Metrics (P|R)  │ mAP (50|50-95) │ Learning Rate")
    print("-" * 150)
    
    try:
        print("\n  Validating configuration...")
        print("  Starting training loop...\n")
        results = model.train(**train_args)
        
        # Training completed
        print(f"\n{'=' * 150}")
        print(f"[5/5] Processing results and saving metrics...")
        
        # Log final metrics
        if hasattr(results, 'results_dict'):
            final_lr = config.LEARNING_RATE
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
        config = SignatureTrainingConfig()
        
        # Check CUDA availability
        print("")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU: {gpu_name}")
        else:
            print("⚠️  CPU mode (training will be slower)")
        
        # Run training
        train_signature_model(config)
        
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
