#!/usr/bin/env python3
"""
Test Data Inference Pipeline

This script:
1. Converts test PDFs to PNG images (if not already done)
2. Runs main model inference (auto-selects best model)
3. Runs ensemble model inference (auto-selects best models)
4. Saves results in separate directories
"""

import os
import sys
import subprocess
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TEST_PDF_DIR = "data/raw/test/test"
TEST_IMAGES_DIR = "data/raw/test_images"
MAIN_OUTPUT_DIR = "outputs/inference/test_main"
SIGNATURE_OUTPUT_DIR = "outputs/inference/test_signature"
ENSEMBLE_OUTPUT_DIR = "outputs/inference/test_ensemble"

# Model paths (None = auto-detect best models)
MAIN_MODEL = None
SIGNATURE_MODEL = None
ENSEMBLE_MAIN_MODEL = None
ENSEMBLE_SEG_MODEL = None

# Use latest checkpoints instead of best models
USE_LATEST = False

# Python interpreter (use current environment)
PYTHON = sys.executable

# PDF conversion settings (matching build_dataset.py)
PDF_DPI = 200


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_best_model(base_dir, model_type="detection", use_latest=False, signature_only=False):
    """Find the best trained model by highest mAP or most recent timestamp
    
    Args:
        base_dir: Directory containing training runs
        model_type: Type of model ("detection" or "segmentation")
        use_latest: If True, select most recent run and load last.pt
        signature_only: If True, only look for signature_only models; if False, exclude them
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
            
            # Filter by signature_only
            is_sig_model = run_dir.name.startswith('signature_only')
            if signature_only and not is_sig_model:
                continue
            if not signature_only and is_sig_model:
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
            model_desc = "latest signature" if signature_only else f"latest {model_type}"
            print(f"  Auto-selected {model_desc} model: {latest_run}")
        
        return latest_model_path
    
    # Select best model by mAP
    best_model_path = None
    best_map = -1
    best_run = None
    
    for run_dir in Path(base_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        # Filter by signature_only
        is_sig_model = run_dir.name.startswith('signature_only')
        if signature_only and not is_sig_model:
            continue
        if not signature_only and is_sig_model:
            continue
        
        metrics_file = run_dir / "training_metrics.csv"
        weights_file = run_dir / "weights" / weight_name
        
        if not metrics_file.exists() or not weights_file.exists():
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            
            # Find best mAP
            if model_type == "segmentation" and 'mask_mAP50-95' in df.columns:
                max_map = df['mask_mAP50-95'].max()
            elif 'mAP50-95' in df.columns:
                max_map = df['mAP50-95'].max()
            elif len(df.columns) > 6:
                max_map = df.iloc[:, 6].max()
            else:
                continue
            
            if max_map > best_map:
                best_map = max_map
                best_model_path = str(weights_file)
                best_run = run_dir.name
        except Exception:
            continue
    
    if best_model_path:
        model_desc = "signature" if signature_only else model_type
        print(f"  Auto-selected best {model_desc} model: {best_run} (mAP: {best_map:.4f})")
    
    return best_model_path


# ============================================================================
# STEP 1: CONVERT PDFs TO PNG
# ============================================================================

def convert_pdfs_to_images():
    """Convert all PDFs in test directory to PNG images (matching build_dataset.py)."""
    
    print("=" * 80)
    print("STEP 1: Converting Test PDFs to PNG Images")
    print("=" * 80)
    
    # Check if PDF directory exists
    pdf_dir = Path(TEST_PDF_DIR)
    if not pdf_dir.exists():
        print(f"❌ Error: PDF directory not found: {TEST_PDF_DIR}")
        return False
    
    # Get all PDF files
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {TEST_PDF_DIR}")
        return False
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    # Create output directory
    output_dir = Path(TEST_IMAGES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {TEST_IMAGES_DIR}")
    
    # Check for existing images
    existing_images = list(output_dir.glob("*.png"))
    if existing_images:
        print(f"\n⚠️  Found {len(existing_images)} existing PNG images")
        print("  Skipping already converted images...")
    
    # Convert each PDF
    total_pages = 0
    converted_count = 0
    skipped_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        pdf_name = pdf_path.stem
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        try:
            # Get number of pages first
            images = convert_from_path(str(pdf_path), dpi=PDF_DPI)
            
            # Check which pages already exist
            pages_to_convert = []
            for page_num in range(1, len(images) + 1):
                output_name = f"{pdf_name}_page_{page_num}.png"
                output_path = output_dir / output_name
                
                if output_path.exists():
                    skipped_count += 1
                else:
                    pages_to_convert.append((page_num, images[page_num - 1]))
            
            # Convert missing pages
            if pages_to_convert:
                print(f"  Converting {len(pages_to_convert)} pages...")
                for page_num, image in pages_to_convert:
                    output_name = f"{pdf_name}_page_{page_num}.png"
                    output_path = output_dir / output_name
                    
                    # Save with high quality (matching build_dataset.py approach)
                    image.save(str(output_path), 'PNG', optimize=False)
                    converted_count += 1
                    total_pages += 1
                
                print(f"  ✓ Converted {len(pages_to_convert)} pages")
            else:
                print(f"  ✓ All {len(images)} pages already exist")
                total_pages += len(images)
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {str(e)}")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"✓ PDF Processing Complete")
    print(f"  Total PDFs: {len(pdf_files)}")
    print(f"  Total pages: {total_pages}")
    print(f"  Newly converted: {converted_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Output: {output_dir.absolute()}")
    print("=" * 80)
    
    return total_pages > 0


# ============================================================================
# STEP 2: RUN MAIN MODEL INFERENCE
# ============================================================================

def run_main_inference():
    """Run main YOLOv11s detection model inference with auto-selected best model."""
    
    print("\n" + "=" * 80)
    print("STEP 2: Running Main Model Inference")
    print("=" * 80)
    
    # Check if test images exist
    test_images_dir = Path(TEST_IMAGES_DIR)
    if not test_images_dir.exists():
        print(f"❌ Error: Test images directory not found: {TEST_IMAGES_DIR}")
        return False
    
    image_files = list(test_images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No images found in {TEST_IMAGES_DIR}")
        return False
    
    print(f"\nRunning inference on {len(image_files)} images...")
    
    # Auto-detect best model if not specified
    global MAIN_MODEL, USE_LATEST
    if MAIN_MODEL is None:
        model_type = "latest" if USE_LATEST else "best"
        print(f"Auto-detecting {model_type} detection model...")
        MAIN_MODEL = find_best_model("runs/train", "detection", use_latest=USE_LATEST)
        if MAIN_MODEL is None:
            print("❌ No trained detection models found")
            return False
    else:
        print(f"Using specified model: {MAIN_MODEL}")
    
    output_dir = MAIN_OUTPUT_DIR + ("_latest" if USE_LATEST else "")
    print(f"Output: {output_dir}")
    
    # Build command (model will be auto-detected by script if not provided)
    cmd = [
        PYTHON,
        "inference_main_model.py",
        "--source", TEST_IMAGES_DIR,
        "--output", output_dir
    ]
    
    if USE_LATEST:
        cmd.append("--latest")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ Main model inference completed successfully")
            print(f"  Results saved to: {Path(output_dir).absolute()}")
            return True
        else:
            print(f"\n❌ Main model inference failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running main model inference: {str(e)}")
        return False


# ============================================================================
# STEP 3: RUN SIGNATURE MODEL INFERENCE
# ============================================================================

def run_signature_inference():
    """Run signature-only model inference with auto-selected best model."""
    
    print("\n" + "=" * 80)
    print("STEP 3: Running Signature Model Inference")
    print("=" * 80)
    
    # Check if test images exist
    test_images_dir = Path(TEST_IMAGES_DIR)
    if not test_images_dir.exists():
        print(f"❌ Error: Test images directory not found: {TEST_IMAGES_DIR}")
        return False
    
    image_files = list(test_images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No images found in {TEST_IMAGES_DIR}")
        return False
    
    print(f"\nRunning signature inference on {len(image_files)} images...")
    
    # Auto-detect best signature model if not specified
    global SIGNATURE_MODEL, USE_LATEST
    if SIGNATURE_MODEL is None:
        model_type = "latest" if USE_LATEST else "best"
        print(f"Auto-detecting {model_type} signature model...")
        SIGNATURE_MODEL = find_best_model("runs/train", "detection", use_latest=USE_LATEST, signature_only=True)
        if SIGNATURE_MODEL is None:
            print("❌ No trained signature models found")
            return False
    else:
        print(f"Using specified model: {SIGNATURE_MODEL}")
    
    output_dir = SIGNATURE_OUTPUT_DIR + ("_latest" if USE_LATEST else "")
    print(f"Output: {output_dir}")
    
    # Build command
    cmd = [
        PYTHON,
        "inference_signature_model.py",
        "--source", TEST_IMAGES_DIR,
        "--output", output_dir
    ]
    
    if USE_LATEST:
        cmd.append("--latest")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ Signature model inference completed successfully")
            print(f"  Results saved to: {Path(output_dir).absolute()}")
            return True
        else:
            print(f"\n❌ Signature model inference failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running signature model inference: {str(e)}")
        return False


# ============================================================================
# STEP 4: RUN ENSEMBLE INFERENCE
# ============================================================================

def run_ensemble_inference():
    """Run ensemble (main + segmentation) model inference with auto-selected best models."""
    
    print("\n" + "=" * 80)
    print("STEP 4: Running Ensemble Model Inference")
    print("=" * 80)
    
    # Check if test images exist
    test_images_dir = Path(TEST_IMAGES_DIR)
    if not test_images_dir.exists():
        print(f"❌ Error: Test images directory not found: {TEST_IMAGES_DIR}")
        return False
    
    image_files = list(test_images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No images found in {TEST_IMAGES_DIR}")
        return False
    
    print(f"\nRunning ensemble inference on {len(image_files)} images...")
    
    # Auto-detect best models if not specified
    global ENSEMBLE_MAIN_MODEL, ENSEMBLE_SEG_MODEL, USE_LATEST
    
    if ENSEMBLE_MAIN_MODEL is None:
        model_type = "latest" if USE_LATEST else "best"
        print(f"Auto-detecting {model_type} detection model...")
        ENSEMBLE_MAIN_MODEL = find_best_model("runs/train", "detection", use_latest=USE_LATEST)
        if ENSEMBLE_MAIN_MODEL is None:
            print("❌ No trained detection models found")
            return False
    else:
        print(f"Using specified main model: {ENSEMBLE_MAIN_MODEL}")
    
    if ENSEMBLE_SEG_MODEL is None:
        model_type = "latest" if USE_LATEST else "best"
        print(f"Auto-detecting {model_type} segmentation model...")
        ENSEMBLE_SEG_MODEL = find_best_model("runs/segment", "segmentation", use_latest=USE_LATEST)
        if ENSEMBLE_SEG_MODEL is None:
            print("❌ No trained segmentation models found")
            return False
    else:
        print(f"Using specified seg model: {ENSEMBLE_SEG_MODEL}")
    
    output_dir = ENSEMBLE_OUTPUT_DIR + ("_latest" if USE_LATEST else "")
    print(f"Output: {output_dir}")
    
    # Build command (models will be auto-detected by script if not provided)
    cmd = [
        PYTHON,
        "inference_ensemble.py",
        "--images", TEST_IMAGES_DIR,
        "--output", output_dir
    ]
    
    if USE_LATEST:
        cmd.append("--latest")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ Ensemble inference completed successfully")
            print(f"  Results saved to: {Path(output_dir).absolute()}")
            return True
        else:
            print(f"\n❌ Ensemble inference failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running ensemble inference: {str(e)}")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test inference pipeline')
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use last.pt (latest checkpoint) instead of best.pt for all models'
    )
    args = parser.parse_args()
    
    global USE_LATEST
    USE_LATEST = args.latest
    
    model_type = "latest" if USE_LATEST else "best"
    
    print("\n" + "=" * 80)
    print("TEST DATA INFERENCE PIPELINE")
    print("=" * 80)
    print(f"\nModel selection: {model_type.upper()}")
    print("\nThis script will:")
    print("  1. Convert test PDFs to PNG images")
    print("  2. Run main model inference")
    print("  3. Run signature model inference")
    print("  4. Run ensemble model inference")
    print("=" * 80)
    
    # Step 1: Convert PDFs
    if not convert_pdfs_to_images():
        print("\n❌ Pipeline failed at PDF conversion step")
        sys.exit(1)
    
    # Step 2: Main model inference
    if not run_main_inference():
        print("\n❌ Pipeline failed at main model inference step")
        sys.exit(1)
    
    # Step 3: Signature model inference
    if not run_signature_inference():
        print("\n❌ Pipeline failed at signature model inference step")
        sys.exit(1)
    
    # Step 4: Ensemble inference
    if not run_ensemble_inference():
        print("\n❌ Pipeline failed at ensemble inference step")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ TEST INFERENCE PIPELINE COMPLETE")
    print("=" * 80)
    
    suffix = "_latest" if USE_LATEST else ""
    print("\n📊 Results:")
    print(f"  Test images: {Path(TEST_IMAGES_DIR).absolute()}")
    print(f"  Main model results: {Path(MAIN_OUTPUT_DIR + suffix).absolute()}")
    print(f"    - detections.json")
    print(f"    - metrics.csv")
    print(f"    - *.png (visualizations)")
    print(f"  Signature model results: {Path(SIGNATURE_OUTPUT_DIR + suffix).absolute()}")
    print(f"    - detections.json")
    print(f"    - metrics.csv")
    print(f"    - *.png (visualizations)")
    print(f"  Ensemble results: {Path(ENSEMBLE_OUTPUT_DIR + suffix).absolute()}")
    print(f"    - detections.json")
    print(f"    - metrics.csv")
    print(f"    - statistics.csv")
    print(f"    - visualizations/")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
