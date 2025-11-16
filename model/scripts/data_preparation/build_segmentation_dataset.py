#!/usr/bin/env python3
"""
YOLOv11 Segmentation Dataset Preparation Script

Generates a segmentation dataset from existing bounding box annotations for stamps and signatures.
Detects overlapping boxes, crops regions with margins, and creates pixel-accurate masks.

Classes:
- stamp: mask value = 1
- signature: mask value = 2

Purpose: Build a second model to more accurately classify stamps and signatures,
and potentially mask out signatures that are inside stamps.
"""

import os
import json
import yaml
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================
JSON_PATH = "data/raw/selected_annotations.json"
IMAGES_DIR = "data/raw/images"  # Directory with converted PDF images
ARCHIVE_DIR = "data/raw/archive"  # Directory with additional annotated images
OUTPUT_DIR = "data/datasets/segmentation"

# Margin to add around crops (as fraction of box size)
CROP_MARGIN = 0.15

# Minimum IoU to consider boxes as overlapping
OVERLAP_THRESHOLD = 0.0  # Any intersection counts as overlap

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x, y, width, height].
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    if inter_area == 0:
        return 0.0
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def boxes_overlap(box1: List[float], box2: List[float]) -> bool:
    """Check if two boxes overlap (any intersection)."""
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2
    
    # Check if boxes intersect
    return not (x1_max <= x2_min or x2_max <= x1_min or 
                y1_max <= y2_min or y2_max <= y1_min)


def get_combined_bbox(boxes: List[List[float]]) -> List[float]:
    """
    Get bounding box that contains all provided boxes.
    Returns [x_min, y_min, width, height].
    """
    if not boxes:
        return [0, 0, 0, 0]
    
    x_mins = [box[0] for box in boxes]
    y_mins = [box[1] for box in boxes]
    x_maxs = [box[0] + box[2] for box in boxes]
    y_maxs = [box[1] + box[3] for box in boxes]
    
    x_min = min(x_mins)
    y_min = min(y_mins)
    x_max = max(x_maxs)
    y_max = max(y_maxs)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def add_margin_to_bbox(bbox: List[float], margin: float, img_width: int, img_height: int) -> List[int]:
    """
    Add margin around bbox and clip to image bounds.
    Margin is relative to bbox size.
    Returns [x_min, y_min, x_max, y_max] in pixels.
    """
    x, y, w, h = bbox
    
    # Calculate margin in pixels (average of width and height)
    margin_pixels = margin * (w + h) / 2
    
    x_min = max(0, int(x - margin_pixels))
    y_min = max(0, int(y - margin_pixels))
    x_max = min(img_width, int(x + w + margin_pixels))
    y_max = min(img_height, int(y + h + margin_pixels))
    
    return [x_min, y_min, x_max, y_max]


def create_mask(crop_bbox: List[int], boxes_with_labels: List[Tuple[List[float], str]], 
                class_to_id: Dict[str, int]) -> np.ndarray:
    """
    Create a mask image for the crop region.
    
    Args:
        crop_bbox: [x_min, y_min, x_max, y_max] of the crop region
        boxes_with_labels: List of (bbox, label) tuples in original image coordinates
        class_to_id: Mapping from class name to mask value
    
    Returns:
        Binary mask as numpy array with values 0 (background), 1 (stamp), 2 (signature)
    """
    x_min, y_min, x_max, y_max = crop_bbox
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    
    # Create mask (0 = background)
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    
    # Sort boxes: stamps first (class_id=1), then signatures (class_id=2)
    # This ensures signatures overwrite stamps in overlap regions (semantically correct)
    sorted_boxes = sorted(boxes_with_labels, key=lambda x: class_to_id.get(x[1], 0))
    
    # Draw each box on the mask in sorted order
    for bbox, label in sorted_boxes:
        if label not in class_to_id:
            continue
        
        class_id = class_to_id[label]
        
        # Convert bbox to crop coordinates
        box_x, box_y, box_w, box_h = bbox
        box_x_min = int(box_x)
        box_y_min = int(box_y)
        box_x_max = int(box_x + box_w)
        box_y_max = int(box_y + box_h)
        
        # Convert to crop-relative coordinates
        rel_x_min = max(0, box_x_min - x_min)
        rel_y_min = max(0, box_y_min - y_min)
        rel_x_max = min(crop_width, box_x_max - x_min)
        rel_y_max = min(crop_height, box_y_max - y_min)
        
        # Fill the box region in the mask
        if rel_x_max > rel_x_min and rel_y_max > rel_y_min:
            mask[rel_y_min:rel_y_max, rel_x_min:rel_x_max] = class_id
    
    return mask


def find_archive_annotations(archive_dir: str, target_classes: List[str] = ['stamp', 'signature']) -> List[Dict]:
    """
    Find all images in archive directory that have corresponding .txt annotation files.
    Only includes stamps and signatures (excludes QR codes).
    Returns list of dicts with: image, width, height, objects, is_archive=True
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return []
    
    # Class mapping (assuming standard order: qr=0, signature=1, stamp=2)
    class_id_to_name = {0: 'qr', 1: 'signature', 2: 'stamp'}
    
    annotations = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
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
        
        # Parse YOLO format annotations
        objects = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    label = class_id_to_name.get(class_id, 'unknown')
                    
                    # Skip if not stamp or signature
                    if label not in target_classes:
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
                        'label': label,
                        'bbox': [x_min, y_min, width, height]
                    })
        except:
            continue
        
        if objects:
            # Use relative path from archive dir to maintain uniqueness
            rel_path = img_path.relative_to(archive_dir)
            annotations.append({
                'image': str(rel_path),
                'image_path': str(img_path),  # Store full path for copying
                'width': img_width,
                'height': img_height,
                'objects': objects,
                'is_archive': True
            })
    
    return annotations


def parse_annotations(json_path: str) -> List[Dict]:
    """
    Load and parse the annotations JSON file.
    Returns list of dicts with: image, width, height, objects
    Only includes stamps and signatures (excludes QR codes).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    flat_annotations = []
    
    for pdf_name, pdf_data in data.items():
        for page_name, page_data in pdf_data.items():
            page_num = page_name.split('_')[-1]
            base_name = os.path.splitext(pdf_name)[0]
            image_name = f"{base_name}_{page_name}.png"
            
            page_size = page_data.get('page_size', {})
            img_width = page_size.get('width', 0)
            img_height = page_size.get('height', 0)
            
            # Parse annotations - only stamps and signatures
            objects = []
            annotations_list = page_data.get('annotations', [])
            
            for ann_item in annotations_list:
                for ann_id, ann_data in ann_item.items():
                    category = ann_data.get('category', 'unknown')
                    
                    # Skip QR codes for segmentation dataset
                    if category not in ['stamp', 'signature']:
                        continue
                    
                    bbox_dict = ann_data.get('bbox', {})
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
            
            if objects:  # Only add if there are stamp/signature annotations
                flat_annotations.append({
                    'image': image_name,
                    'width': img_width,
                    'height': img_height,
                    'objects': objects
                })
    
    return flat_annotations


def find_overlap_groups(all_boxes: List[Tuple[List[float], str]]) -> List[List[int]]:
    """
    Find groups of overlapping boxes using Union-Find algorithm.
    
    Args:
        all_boxes: List of (bbox, label) tuples
        
    Returns:
        List of groups, where each group is a list of indices
    """
    n = len(all_boxes)
    
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Find all overlapping pairs and union them
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(all_boxes[i][0], all_boxes[j][0]):
                union(i, j)
    
    # Group indices by their root parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return list(groups.values())


def process_image_annotations(ann: Dict, class_to_id: Dict[str, int]) -> List[Dict]:
    """
    Process annotations for a single image.
    Detect ALL overlapping boxes (including 3+ overlaps) and group them together.
    
    Returns list of crop info dicts with:
        - crop_bbox: [x_min, y_min, x_max, y_max]
        - boxes: List of (bbox, label) tuples included in this crop
        - crop_type: 'overlap', 'individual_stamp', or 'individual_signature'
    """
    img_width = ann['width']
    img_height = ann['height']
    objects = ann['objects']
    
    # Get all boxes with their labels
    all_boxes = [(obj['bbox'], obj['label']) for obj in objects]
    
    if len(all_boxes) == 0:
        return []
    
    # Find groups of overlapping boxes
    overlap_groups = find_overlap_groups(all_boxes)
    
    crops = []
    
    # Process each group
    for group_indices in overlap_groups:
        # Get boxes in this group
        group_boxes = [all_boxes[i] for i in group_indices]
        
        # Determine crop type
        if len(group_boxes) == 1:
            # Individual box
            bbox, label = group_boxes[0]
            crop_type = f'individual_{label}'
        else:
            # Multiple overlapping boxes
            crop_type = f'overlap_{len(group_boxes)}boxes'
        
        # Get combined bounding box for all boxes in group
        bboxes_only = [bbox for bbox, label in group_boxes]
        combined_bbox = get_combined_bbox(bboxes_only)
        crop_bbox = add_margin_to_bbox(combined_bbox, CROP_MARGIN, img_width, img_height)
        
        crops.append({
            'crop_bbox': crop_bbox,
            'boxes': group_boxes,
            'crop_type': crop_type
        })
    
    return crops


def create_directory_structure(output_dir: str):
    """Create segmentation dataset directory structure."""
    dirs = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'masks', 'train'),
        os.path.join(output_dir, 'masks', 'val'),
        os.path.join(output_dir, 'train-vis')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def split_annotations(annotations: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
    """Split annotations into train and validation sets (default 90/10)."""
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_ratio)
    return annotations[:split_idx], annotations[split_idx:]


def process_and_save_crops(annotations: List[Dict], class_to_id: Dict[str, int], 
                           images_dir: str, output_dir: str, split: str) -> int:
    """
    Process all annotations and save crops with masks.
    Returns total number of crops created.
    """
    total_crops = 0
    
    for ann in annotations:
        image_name = ann['image']
        
        # Use image_path if available (for archive images)
        if ann.get('is_archive', False) and 'image_path' in ann:
            img_path = ann['image_path']
            # Use simpler name for output (just filename)
            base_name = os.path.splitext(Path(image_name).name)[0]
        else:
            img_path = os.path.join(images_dir, image_name)
            base_name = os.path.splitext(image_name)[0]
        
        # Load image
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Process annotations to get crops
        crops = process_image_annotations(ann, class_to_id)
        
        # Save each crop and its mask
        # For archive images, use just the filename
        if ann.get('is_archive', False):
            base_name = os.path.splitext(Path(image_name).name)[0]
        else:
            base_name = os.path.splitext(image_name)[0]
        
        for crop_idx, crop_info in enumerate(crops):
            crop_bbox = crop_info['crop_bbox']
            boxes = crop_info['boxes']
            crop_type = crop_info['crop_type']
            
            x_min, y_min, x_max, y_max = crop_bbox
            
            # Crop image
            crop_img = img[y_min:y_max, x_min:x_max]
            
            if crop_img.size == 0:
                continue
            
            # Create mask
            mask = create_mask(crop_bbox, boxes, class_to_id)
            
            # Generate unique filename (remove any path separators from base_name)
            safe_base_name = base_name.replace('/', '_').replace('\\', '_')
            crop_name = f"{safe_base_name}_crop{crop_idx:03d}_{crop_type}.png"
            
            # Save crop image
            crop_img_path = os.path.join(output_dir, 'images', split, crop_name)
            cv2.imwrite(crop_img_path, crop_img)
            
            # Save mask
            mask_path = os.path.join(output_dir, 'masks', split, crop_name)
            cv2.imwrite(mask_path, mask)
            
            total_crops += 1
    
    return total_crops


def create_visualizations(output_dir: str, split: str, max_samples: int = 50):
    """
    Create visualization of crops with masks overlaid.
    Saves to train-vis/ folder.
    """
    images_dir = os.path.join(output_dir, 'images', split)
    masks_dir = os.path.join(output_dir, 'masks', split)
    vis_dir = os.path.join(output_dir, 'train-vis')
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    # Sample randomly if too many
    if len(image_files) > max_samples:
        image_files = random.sample(image_files, max_samples)
    
    # Define colors for visualization (BGR)
    colors = {
        0: (0, 0, 0),         # Background - black
        1: (0, 0, 255),       # Stamp - red
        2: (0, 255, 0)        # Signature - green
    }
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_file)
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Create colored overlay
        overlay = img.copy()
        
        # Apply colors for each class
        for class_id, color in colors.items():
            if class_id == 0:  # Skip background
                continue
            class_mask = (mask == class_id)
            overlay[class_mask] = color
        
        # Blend with original image
        alpha = 0.4
        vis_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # Add legend
        legend_height = 60
        legend = np.ones((legend_height, vis_img.shape[1], 3), dtype=np.uint8) * 255
        
        cv2.putText(legend, "Stamp", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(legend, "Signature", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine
        final_vis = np.vstack([vis_img, legend])
        
        # Save
        vis_path = os.path.join(vis_dir, f"vis_{split}_{img_file}")
        cv2.imwrite(vis_path, final_vis)


def analyze_overlap_statistics(annotations: List[Dict], split_name: str):
    """
    Analyze and print statistics about overlapping boxes.
    
    Args:
        annotations: List of annotation dictionaries
        split_name: Name of the split (e.g., "Training", "Validation")
    """
    overlap_counts = {}  # {num_boxes_in_group: count}
    individual_stamps = 0
    individual_signatures = 0
    
    for ann in annotations:
        all_boxes = [(obj['bbox'], obj['label']) for obj in ann['objects']]
        if not all_boxes:
            continue
        
        overlap_groups = find_overlap_groups(all_boxes)
        
        for group_indices in overlap_groups:
            group_size = len(group_indices)
            
            if group_size == 1:
                # Individual box
                label = all_boxes[group_indices[0]][1]
                if label == 'stamp':
                    individual_stamps += 1
                else:
                    individual_signatures += 1
            else:
                # Overlapping group
                overlap_counts[group_size] = overlap_counts.get(group_size, 0) + 1
    
    print(f"\n      {split_name} Overlap Statistics:")
    print(f"        Individual stamps: {individual_stamps}")
    print(f"        Individual signatures: {individual_signatures}")
    
    if overlap_counts:
        print(f"        Overlapping groups:")
        for size in sorted(overlap_counts.keys()):
            print(f"          {size} boxes overlapping: {overlap_counts[size]} groups")
    else:
        print(f"        No overlapping groups found")


def create_dataset_yaml(output_dir: str):
    """Create dataset.yaml configuration file for YOLOv11-seg."""
    config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'stamp',
            1: 'signature'
        }
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build YOLOv11 segmentation dataset')
    parser.add_argument(
        '--include-archive',
        action='store_true',
        help='Include data from archive directory'
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLOv11 Segmentation Dataset Builder")
    print("=" * 80)
    print("\nPurpose: Build a segmentation model to accurately classify stamps and")
    print("signatures, including handling signatures inside stamps.\n")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Class mapping for masks
    class_to_id = {
        'stamp': 1,
        'signature': 2
    }
    
    # Parse annotations from JSON
    print(f"[1/7] Loading annotations from: {JSON_PATH}")
    annotations = parse_annotations(JSON_PATH)
    print(f"      JSON images with stamps/signatures: {len(annotations)}")
    
    # Add archive annotations if requested
    if args.include_archive:
        print(f"      Loading archive annotations from: {ARCHIVE_DIR}")
        archive_annotations = find_archive_annotations(ARCHIVE_DIR, target_classes=['stamp', 'signature'])
        print(f"      Archive images with stamps/signatures: {len(archive_annotations)}")
        annotations.extend(archive_annotations)
    else:
        print(f"      Skipping archive data (use --include-archive to include)")
    
    print(f"      Total images with stamps/signatures: {len(annotations)}")
    
    # Count objects
    total_stamps = sum(len([o for o in ann['objects'] if o['label'] == 'stamp']) 
                      for ann in annotations)
    total_signatures = sum(len([o for o in ann['objects'] if o['label'] == 'signature']) 
                          for ann in annotations)
    print(f"      Total stamps: {total_stamps}")
    print(f"      Total signatures: {total_signatures}")
    
    # Create directory structure
    print(f"\n[2/7] Creating dataset structure in: {OUTPUT_DIR}")
    create_directory_structure(OUTPUT_DIR)
    
    # Split dataset
    print("\n[3/7] Splitting dataset (90% train / 10% val)...")
    train_annotations, val_annotations = split_annotations(annotations)
    print(f"      Train: {len(train_annotations)} images")
    print(f"      Val: {len(val_annotations)} images")
    
    # Process training set
    print("\n[4/7] Processing training crops and masks...")
    train_crops = process_and_save_crops(train_annotations, class_to_id, 
                                        IMAGES_DIR, OUTPUT_DIR, 'train')
    print(f"      Created {train_crops} training crops")
    
    # Analyze overlap statistics for training
    analyze_overlap_statistics(train_annotations, "Training")
    
    # Process validation set
    print("\n[5/7] Processing validation crops and masks...")
    val_crops = process_and_save_crops(val_annotations, class_to_id, 
                                      IMAGES_DIR, OUTPUT_DIR, 'val')
    print(f"      Created {val_crops} validation crops")
    
    # Analyze overlap statistics for validation
    analyze_overlap_statistics(val_annotations, "Validation")
    
    # Create visualizations
    print("\n[6/7] Creating visualizations...")
    create_visualizations(OUTPUT_DIR, 'train', max_samples=50)
    create_visualizations(OUTPUT_DIR, 'val', max_samples=20)
    print(f"      Saved visualization samples to train-vis/")
    
    # Create dataset.yaml
    print("\n[7/7] Creating dataset configuration...")
    yaml_path = create_dataset_yaml(OUTPUT_DIR)
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("Segmentation Dataset Creation Complete!")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Training crops:   {train_crops}")
    print(f"  Validation crops: {val_crops}")
    print(f"  Total crops:      {train_crops + val_crops}")
    print(f"\nClasses:")
    print(f"  1: stamp (red in visualizations)")
    print(f"  2: signature (green in visualizations)")
    print(f"\nConfiguration:")
    print(f"  Crop margin: {CROP_MARGIN * 100}% of box size")
    print(f"  Overlap detection: Any intersection counts")
    print(f"\nOutput:")
    print(f"  Dataset directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Config file: {yaml_path}")
    print(f"  Visualizations: {os.path.join(OUTPUT_DIR, 'train-vis/')}")
    print(f"\nNext Steps:")
    print(f"  1. Review visualizations in train-vis/ folder")
    print(f"  2. Train YOLOv11-seg model using this dataset")
    print(f"  3. Use the model to accurately segment stamps and signatures")
    print("=" * 80)


if __name__ == "__main__":  
    main()
