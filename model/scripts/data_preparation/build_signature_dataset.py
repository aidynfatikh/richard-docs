#!/usr/bin/env python3
"""
Signature-Only Dataset Preparation Script
Builds a YOLOv11 dataset containing only signature bounding boxes.
Based on the original build_dataset.py but filters for signatures only.
"""

import os
import json
import yaml
import shutil
import random
import cv2
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
JSON_PATH = "data/raw/selected_annotations.json"
PDF_DIR = "data/raw/pdfs/pdfs"  # Directory containing PDF files
IMAGES_DIR = "data/raw/images"  # Directory where converted images will be saved
ARCHIVE_DIR = "data/raw/archive"  # Directory with additional annotated images
OUTPUT_DIR = "data/datasets/signature"  # Output directory for signature-only dataset


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def find_archive_annotations(archive_dir, class_names=['signature']):
    """
    Find all images in archive directory with signature annotations.
    Converts YOLO format annotations and filters for signature class only.
    Returns list of dicts with: image, width, height, objects, is_archive=True
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return []
    
    annotations = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Map class IDs to names (assuming original dataset order: qr, signature, stamp)
    original_class_names = ['qr', 'signature', 'stamp']
    signature_id = original_class_names.index('signature') if 'signature' in original_class_names else 1
    
    # Recursively search for images with annotations
    for img_path in archive_dir.rglob('*'):
        if not img_path.is_file() or img_path.suffix not in image_extensions:
            continue
        
        # Check for corresponding .txt file
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists() or txt_path.stat().st_size == 0:
            continue
        
        # Load image to get dimensions
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
        except:
            continue
        
        # Parse YOLO format annotations (filter for signatures only)
        objects = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Skip if not signature class
                    if class_id != signature_id:
                        continue
                    
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    
                    # Convert from normalized YOLO to pixel coordinates
                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    width = width_norm * img_width
                    height = height_norm * img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    
                    objects.append({
                        'label': 'signature',
                        'bbox': [x_min, y_min, width, height]
                    })
        except:
            continue
        
        if objects:  # Only add if has signature annotations
            rel_path = img_path.relative_to(archive_dir)
            annotations.append({
                'image': str(rel_path),
                'image_path': str(img_path),
                'width': img_width,
                'height': img_height,
                'objects': objects,
                'is_archive': True
            })
    
    return annotations


def parse_annotations(json_path):
    """
    Load and parse the annotations JSON file.
    Filters for signature annotations only.
    Returns list of dicts with: image, width, height, objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    flat_annotations = []
    
    for pdf_name, pdf_data in data.items():
        for page_name, page_data in pdf_data.items():
            # Image name: pdf_name + page_number
            page_num = page_name.split('_')[-1]
            base_name = os.path.splitext(pdf_name)[0]
            image_name = f"{base_name}_{page_name}.png"
            
            page_size = page_data.get('page_size', {})
            img_width = page_size.get('width', 0)
            img_height = page_size.get('height', 0)
            
            # Parse annotations - filter for signatures only
            objects = []
            annotations_list = page_data.get('annotations', [])
            
            for ann_item in annotations_list:
                for ann_id, ann_data in ann_item.items():
                    category = ann_data.get('category', 'unknown')
                    
                    # Skip if not signature
                    if category != 'signature':
                        continue
                    
                    bbox_dict = ann_data.get('bbox', {})
                    
                    # Convert bbox dict to list [x, y, width, height]
                    bbox = [
                        bbox_dict.get('x', 0),
                        bbox_dict.get('y', 0),
                        bbox_dict.get('width', 0),
                        bbox_dict.get('height', 0)
                    ]
                    
                    objects.append({
                        'label': 'signature',
                        'bbox': bbox
                    })
            
            if objects:  # Only add if there are signature annotations
                flat_annotations.append({
                    'image': image_name,
                    'width': img_width,
                    'height': img_height,
                    'objects': objects
                })
    
    return flat_annotations


def bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert [x_min, y_min, width, height] to YOLO format.
    Returns: (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def create_directory_structure(output_dir):
    """Create YOLOv11 dataset directory structure."""
    dirs = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val'),
        os.path.join(output_dir, 'vis')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs


def split_dataset(annotations, train_ratio=0.8):
    """
    Randomly split annotations into train and validation sets.
    Ensures no image appears in both train and val by deduplicating first.
    """
    # Deduplicate by image name (keep first occurrence)
    seen_images = set()
    unique_annotations = []
    
    for ann in annotations:
        # Get base image name (without path for archive images)
        image_name = ann['image']
        if ann.get('is_archive', False):
            image_name = Path(image_name).name
        
        if image_name not in seen_images:
            seen_images.add(image_name)
            unique_annotations.append(ann)
    
    duplicates_removed = len(annotations) - len(unique_annotations)
    if duplicates_removed > 0:
        print(f"      Removed {duplicates_removed} duplicate images")
    
    # Now split the unique annotations
    random.shuffle(unique_annotations)
    split_idx = int(len(unique_annotations) * train_ratio)
    return unique_annotations[:split_idx], unique_annotations[split_idx:]


def process_annotation(ann, images_dir, output_dir, split):
    """
    Process a single annotation: copy image and create YOLO label file.
    For signature-only dataset, class_id is always 0.
    Returns: True if successful, False if image is missing.
    """
    image_name = ann['image']
    img_width = ann['width']
    img_height = ann['height']
    
    # Source image path
    if ann.get('is_archive', False) and 'image_path' in ann:
        src_image_path = ann['image_path']
        image_name = Path(image_name).name
    else:
        src_image_path = os.path.join(images_dir, image_name)
    
    # Check if image exists
    if not os.path.exists(src_image_path):
        print(f"Warning: Image not found: {src_image_path}")
        return False
    
    # Destination paths
    dst_image_path = os.path.join(output_dir, 'images', split, image_name)
    label_name = os.path.splitext(image_name)[0] + '.txt'
    label_path = os.path.join(output_dir, 'labels', split, label_name)
    
    # Copy image
    shutil.copy2(src_image_path, dst_image_path)
    
    # Create YOLO label file (all signatures have class_id=0)
    with open(label_path, 'w') as f:
        for obj in ann.get('objects', []):
            bbox = obj['bbox']
            
            # Validate bbox
            if not bbox or len(bbox) != 4:
                print(f"Warning: Skipping invalid bbox in {image_name}: {bbox}")
                continue
            
            # Validate bbox dimensions
            if bbox[2] <= 0 or bbox[3] <= 0:
                print(f"Warning: Skipping invalid bbox dimensions in {image_name}: {bbox}")
                continue
            
            x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)
            
            # Validate normalized coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"Warning: Skipping out-of-range bbox in {image_name}: "
                      f"x={x_center:.3f}, y={y_center:.3f}, w={width:.3f}, h={height:.3f}")
                continue
            
            # Write with class_id=0 (signature is the only class)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True


def visualize_samples(annotations, images_dir, output_dir):
    """Visualize all samples with signature annotations."""
    # Green color for signatures (BGR format)
    color = (0, 255, 0)
    
    vis_dir = os.path.join(output_dir, 'vis')
    
    for ann in annotations:
        image_name = ann['image']
        
        # For archive images, use just the filename
        if ann.get('is_archive', False):
            lookup_name = Path(image_name).name
        else:
            lookup_name = image_name
        
        # Find the label file in train or val
        label_name = os.path.splitext(lookup_name)[0] + '.txt'
        label_path = None
        for split in ['train', 'val']:
            candidate_path = os.path.join(output_dir, 'labels', split, label_name)
            if os.path.exists(candidate_path):
                label_path = candidate_path
                break
        
        if not label_path:
            continue
        
        # Use source path for archive images
        if ann.get('is_archive', False) and 'image_path' in ann:
            img_path = ann['image_path']
        else:
            img_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Read YOLO labels and draw boxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                
                # Convert to pixel coordinates
                x1 = int((x_center_norm - width_norm / 2) * img_width)
                y1 = int((y_center_norm - height_norm / 2) * img_height)
                x2 = int((x_center_norm + width_norm / 2) * img_width)
                y2 = int((y_center_norm + height_norm / 2) * img_height)
                
                # Clamp to bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label_text = 'signature'
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                label_y1 = max(text_height + 10, y1)
                cv2.rectangle(img, (x1, label_y1 - text_height - 10), 
                             (x1 + text_width, label_y1), color, -1)
                
                cv2.putText(img, label_text, (x1, label_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        vis_filename = lookup_name if ann.get('is_archive', False) else image_name
        vis_path = os.path.join(vis_dir, vis_filename)
        cv2.imwrite(vis_path, img)


def convert_pdfs_to_images(pdf_dir, output_dir, annotations):
    """
    Convert PDF pages to PNG images based on annotations.
    Resizes images to match the dimensions specified in annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all required PDF pages with their target dimensions
    pdf_pages = {}
    for ann in annotations:
        image_name = ann['image']
        parts = image_name.rsplit('_page_', 1)
        if len(parts) == 2:
            pdf_base = parts[0]
            page_num = int(parts[1].replace('.png', ''))
            target_width = ann['width']
            target_height = ann['height']
            
            if pdf_base not in pdf_pages:
                pdf_pages[pdf_base] = {}
            pdf_pages[pdf_base][page_num] = (target_width, target_height)
    
    print(f"      Found {len(pdf_pages)} PDFs with {len(annotations)} annotated pages")
    
    converted_count = 0
    for pdf_base, pages_info in pdf_pages.items():
        pdf_path = None
        for ext in ['.pdf', '.PDF']:
            candidate = os.path.join(pdf_dir, pdf_base + ext)
            if os.path.exists(candidate):
                pdf_path = candidate
                break
        
        if not pdf_path:
            print(f"      Warning: PDF not found for {pdf_base}")
            continue
        
        try:
            for page_num, (target_width, target_height) in pages_info.items():
                output_path = os.path.join(output_dir, f"{pdf_base}_page_{page_num}.png")
                
                if os.path.exists(output_path):
                    converted_count += 1
                    continue
                
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=200)
                if images:
                    img = images[0]
                    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                    img_resized.save(output_path, 'PNG')
                    converted_count += 1
                    print(f"      Converted: {pdf_base} page {page_num} -> {target_width}x{target_height}")
        
        except Exception as e:
            print(f"      Error converting {pdf_base}: {str(e)}")
    
    print(f"      Total images: {converted_count}")
    return converted_count


def create_dataset_yaml(output_dir):
    """Generate dataset.yaml configuration file for signature-only dataset."""
    dataset_config = {
        'train': 'images/train',
        'val': 'images/val',
        'names': ['signature']  # Single class
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build signature-only dataset')
    parser.add_argument(
        '--include-archive',
        action='store_true',
        help='Include data from archive directory'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Signature-Only Dataset Preparation")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse annotations from JSON (signatures only)
    print(f"\n[1/8] Loading signature annotations from: {JSON_PATH}")
    annotations = parse_annotations(JSON_PATH)
    print(f"      JSON signature annotations: {len(annotations)}")
    
    # Add archive annotations if requested (signatures only)
    if args.include_archive:
        print(f"      Loading archive signature annotations from: {ARCHIVE_DIR}")
        archive_annotations = find_archive_annotations(ARCHIVE_DIR)
        print(f"      Archive signature annotations: {len(archive_annotations)}")
        annotations.extend(archive_annotations)
    else:
        print(f"      Skipping archive data (use --include-archive to include)")
    
    total_signatures = sum(len(ann['objects']) for ann in annotations)
    print(f"      Total images with signatures: {len(annotations)}")
    print(f"      Total signature instances: {total_signatures}")
    
    # Convert PDFs to images
    print(f"\n[2/8] Converting PDF pages to images...")
    convert_pdfs_to_images(PDF_DIR, IMAGES_DIR, annotations)
    
    # Create directory structure
    print(f"\n[3/8] Creating dataset structure in: {OUTPUT_DIR}")
    create_directory_structure(OUTPUT_DIR)
    
    # Split dataset
    print("\n[3/5] Splitting dataset (90% train / 10% val)...")
    train_annotations, val_annotations = split_dataset(annotations)
    print(f"      Train: {len(train_annotations)} images")
    print(f"      Val: {len(val_annotations)} images")
    
    # Process training set
    print("\n[5/8] Processing training images and labels...")
    train_count = 0
    for ann in train_annotations:
        if process_annotation(ann, IMAGES_DIR, OUTPUT_DIR, 'train'):
            train_count += 1
    print(f"      Processed: {train_count}/{len(train_annotations)} images")
    
    # Process validation set
    print("\n[6/8] Processing validation images and labels...")
    val_count = 0
    for ann in val_annotations:
        if process_annotation(ann, IMAGES_DIR, OUTPUT_DIR, 'val'):
            val_count += 1
    print(f"      Processed: {val_count}/{len(val_annotations)} images")
    
    # Visualize samples
    print("\n[7/8] Creating visualizations...")
    all_annotations = train_annotations + val_annotations
    visualize_samples(all_annotations, IMAGES_DIR, OUTPUT_DIR)
    print(f"      Saved {len(all_annotations)} visualizations to vis/")
    
    # Create dataset.yaml
    print("\n[8/8] Creating dataset configuration...")
    yaml_path = create_dataset_yaml(OUTPUT_DIR)
    
    print(f"\n{'=' * 70}")
    print("Signature Dataset Preparation Complete!")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Training images:   {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Total images:      {train_count + val_count}")
    print(f"  Total signatures:  {total_signatures}")
    print(f"\nClasses: 1 (signature only)")
    print(f"\nDataset configuration: {yaml_path}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("\nReady for signature-only model training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
