#!/usr/bin/env python3
"""
YOLOv11 Dataset Preparation Script
Converts custom JSON annotations to YOLO format with train/val split and visualization.
"""

import os
import json
import yaml
import shutil
import random
import cv2
from pathlib import Path
from pdf2image import convert_from_path


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
JSON_PATH = "raw_data/selected_annotations.json"
PDF_DIR = "raw_data/pdfs/pdfs"  # Directory containing PDF files
IMAGES_DIR = "raw_data/images"  # Directory where converted images will be saved
OUTPUT_DIR = "dataset"


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def parse_annotations(json_path):
    """
    Load and parse the annotations JSON file.
    Converts nested structure to flat list format.
    Returns list of dicts with: image, width, height, objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    flat_annotations = []
    
    for pdf_name, pdf_data in data.items():
        for page_name, page_data in pdf_data.items():
            # Image name: pdf_name + page_number
            # Extract page number from "page_X" format
            page_num = page_name.split('_')[-1]
            # Create image name: pdfname_pageX.png
            base_name = os.path.splitext(pdf_name)[0]
            image_name = f"{base_name}_{page_name}.png"
            
            page_size = page_data.get('page_size', {})
            img_width = page_size.get('width', 0)
            img_height = page_size.get('height', 0)
            
            # Parse annotations
            objects = []
            annotations_list = page_data.get('annotations', [])
            
            for ann_item in annotations_list:
                # Each annotation is a dict with one key (annotation_XXX)
                for ann_id, ann_data in ann_item.items():
                    category = ann_data.get('category', 'unknown')
                    bbox_dict = ann_data.get('bbox', {})
                    
                    # Convert bbox dict to list [x, y, width, height]
                    bbox = [
                        bbox_dict.get('x', 0),
                        bbox_dict.get('y', 0),
                        bbox_dict.get('width', 0),
                        bbox_dict.get('height', 0)
                    ]
                    
                    objects.append({
                        'label': category,
                        'bbox': bbox
                    })
            
            if objects:  # Only add if there are annotations
                flat_annotations.append({
                    'image': image_name,
                    'width': img_width,
                    'height': img_height,
                    'objects': objects
                })
    
    return flat_annotations


def collect_class_names(annotations):
    """Extract all unique class labels and sort alphabetically."""
    class_names = set()
    for ann in annotations:
        for obj in ann.get('objects', []):
            class_names.add(obj['label'])
    return sorted(list(class_names))


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
        os.path.join(output_dir, 'train-vis')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs


def split_dataset(annotations, train_ratio=0.8):
    """Randomly split annotations into train and validation sets."""
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_ratio)
    return annotations[:split_idx], annotations[split_idx:]


def process_annotation(ann, class_to_id, images_dir, output_dir, split):
    """
    Process a single annotation: copy image and create YOLO label file.
    Returns: True if successful, False if image is missing.
    """
    image_name = ann['image']
    img_width = ann['width']
    img_height = ann['height']
    
    # Source image path
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
    
    # Create YOLO label file
    with open(label_path, 'w') as f:
        for obj in ann.get('objects', []):
            class_name = obj['label']
            class_id = class_to_id[class_name]
            bbox = obj['bbox']
            
            x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True


def visualize_samples(annotations, class_names, images_dir, output_dir, num_samples=20):
    """Visualize random training samples with bounding boxes."""
    # Generate random colors for each class
    colors = {}
    for idx, class_name in enumerate(class_names):
        random.seed(idx * 42)
        colors[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    # Select random samples
    samples = random.sample(annotations, min(num_samples, len(annotations)))
    
    vis_dir = os.path.join(output_dir, 'train-vis')
    
    for ann in samples:
        image_name = ann['image']
        img_path = os.path.join(output_dir, 'images', 'train', image_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Draw bounding boxes
        for obj in ann.get('objects', []):
            label = obj['label']
            bbox = obj['bbox']
            x_min, y_min, width, height = bbox
            
            x1 = int(x_min)
            y1 = int(y_min)
            x2 = int(x_min + width)
            y2 = int(y_min + height)
            
            color = colors[label]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label_text = label
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(img, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, image_name)
        cv2.imwrite(vis_path, img)


def convert_pdfs_to_images(pdf_dir, output_dir, annotations):
    """
    Convert PDF pages to PNG images based on annotations.
    Only converts the pages that have annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all required PDF pages
    pdf_pages = {}
    for ann in annotations:
        image_name = ann['image']
        # Parse image name: "pdfname_page_X.png"
        parts = image_name.rsplit('_page_', 1)
        if len(parts) == 2:
            pdf_base = parts[0]
            page_num = int(parts[1].replace('.png', ''))
            
            # Find matching PDF file (case-insensitive)
            if pdf_base not in pdf_pages:
                pdf_pages[pdf_base] = []
            pdf_pages[pdf_base].append(page_num)
    
    print(f"      Found {len(pdf_pages)} PDFs with {len(annotations)} annotated pages")
    
    converted_count = 0
    for pdf_base, pages in pdf_pages.items():
        # Find the PDF file (try different extensions and cases)
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
            # Convert only required pages
            for page_num in sorted(set(pages)):
                output_path = os.path.join(output_dir, f"{pdf_base}_page_{page_num}.png")
                
                if os.path.exists(output_path):
                    converted_count += 1
                    continue
                
                # Convert single page (page_num is 1-indexed)
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
                if images:
                    images[0].save(output_path, 'PNG')
                    converted_count += 1
                    print(f"      Converted: {pdf_base} page {page_num}")
        
        except Exception as e:
            print(f"      Error converting {pdf_base}: {str(e)}")
    
    print(f"      Total images: {converted_count}")
    return converted_count


def create_dataset_yaml(output_dir, class_names):
    """Generate dataset.yaml configuration file."""
    dataset_config = {
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    """Main execution function."""
    print("=" * 70)
    print("YOLOv11 Dataset Preparation")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse annotations
    print(f"\n[1/8] Loading annotations from: {JSON_PATH}")
    annotations = parse_annotations(JSON_PATH)
    print(f"      Total annotations: {len(annotations)}")
    
    # Collect class names
    print("\n[2/8] Collecting class labels...")
    class_names = collect_class_names(annotations)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"      Found {len(class_names)} classes: {class_names}")
    
    # Convert PDFs to images
    print(f"\n[3/8] Converting PDF pages to images...")
    convert_pdfs_to_images(PDF_DIR, IMAGES_DIR, annotations)
    
    # Create directory structure
    print(f"\n[4/8] Creating dataset structure in: {OUTPUT_DIR}")
    create_directory_structure(OUTPUT_DIR)
    
    # Split dataset
    print("\n[5/8] Splitting dataset (80% train / 20% val)...")
    train_annotations, val_annotations = split_dataset(annotations)
    print(f"      Train: {len(train_annotations)} images")
    print(f"      Val: {len(val_annotations)} images")
    
    # Process training set
    print("\n[6/8] Processing training images and labels...")
    train_count = 0
    for ann in train_annotations:
        if process_annotation(ann, class_to_id, IMAGES_DIR, OUTPUT_DIR, 'train'):
            train_count += 1
    print(f"      Processed: {train_count}/{len(train_annotations)} images")
    
    # Process validation set
    print("\n[7/8] Processing validation images and labels...")
    val_count = 0
    for ann in val_annotations:
        if process_annotation(ann, class_to_id, IMAGES_DIR, OUTPUT_DIR, 'val'):
            val_count += 1
    print(f"      Processed: {val_count}/{len(val_annotations)} images")
    
    # Visualize samples
    print("\n[8/8] Creating visualizations...")
    visualize_samples(train_annotations, class_names, IMAGES_DIR, OUTPUT_DIR)
    print("      Saved visualizations to train-vis/")
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(OUTPUT_DIR, class_names)
    print(f"\n{'=' * 70}")
    print("Dataset preparation complete!")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Training images:   {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Total images:      {train_count + val_count}")
    print(f"\nClasses ({len(class_names)}):")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print(f"\nDataset configuration: {yaml_path}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("\nReady for YOLOv11 training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
