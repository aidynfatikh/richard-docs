"""
YOLO Model Service
Handles loading and inference for document element detection (stamps, signatures, QR codes)
Incorporates advanced detection grouping for overlapping signatures.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Set
import cv2
import numpy as np
from ultralytics import YOLO


class DocumentDetector:
    """
    Document element detector using YOLOv11.
    Detects stamps, signatures, and QR codes.
    Includes signature grouping for overlapping detections.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25, device='cpu', 
                 enable_grouping=True, group_iou_threshold=0.3, iou_threshold=0.45):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO weights. If None, uses default path.
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run on ('cpu', 'cuda', '0', '1', etc.)
            enable_grouping: Enable signature grouping (default: True)
            group_iou_threshold: IoU threshold for grouping signatures (default: 0.3)
            iou_threshold: NMS IoU threshold (default: 0.45)
        """
        if model_path is None:
            # Default to best model from training
            model_path = self._find_best_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_grouping = enable_grouping
        self.group_iou_threshold = group_iou_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Model loaded. Classes: {self.class_names}")
        if self.enable_grouping:
            print(f"Signature grouping enabled (IoU threshold: {self.group_iou_threshold})")
    
    def _find_best_model(self):
        """Find the best.pt model in the runs directory."""
        model_dir = Path(__file__).parent.parent.parent / "model" / "runs" / "train"
        
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Training runs directory not found: {model_dir}\n"
                "Please train a model first or specify model_path explicitly."
            )
        
        # Find most recent training run
        runs = sorted(model_dir.glob("yolov11s_*"), key=os.path.getmtime, reverse=True)
        
        if not runs:
            raise FileNotFoundError(f"No training runs found in {model_dir}")
        
        best_model = runs[0] / "weights" / "best.pt"
        
        if not best_model.exists():
            raise FileNotFoundError(f"best.pt not found in {runs[0]}/weights/")
        
        return str(best_model)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU (Intersection over Union) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def group_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Group closely stacked detections (especially signatures).
        Multiple overlapping boxes of the same class are merged into one.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of detections with grouped boxes merged
        """
        if not self.enable_grouping or not detections:
            return detections
        
        # Separate by class
        by_class = {}
        for det in detections:
            cls_name = det['class_name']
            if cls_name not in by_class:
                by_class[cls_name] = []
            by_class[cls_name].append(det)
        
        grouped_detections = []
        
        # Process each class separately
        for cls_name, dets in by_class.items():
            # Only group signatures (they tend to have multiple overlapping boxes)
            if cls_name != 'signature':
                grouped_detections.extend(dets)
                continue
            
            if len(dets) <= 1:
                grouped_detections.extend(dets)
                continue
            
            # Build connectivity graph
            n = len(dets)
            graph = {i: set() for i in range(n)}
            
            for i in range(n):
                for j in range(i + 1, n):
                    iou = self.calculate_iou(dets[i]['box'], dets[j]['box'])
                    if iou >= self.group_iou_threshold:
                        graph[i].add(j)
                        graph[j].add(i)
            
            # Find connected components using DFS
            visited = set()
            components = []
            
            def dfs(node: int, component: Set[int]):
                if node in visited:
                    return
                visited.add(node)
                component.add(node)
                for neighbor in graph[node]:
                    dfs(neighbor, component)
            
            for i in range(n):
                if i not in visited:
                    component = set()
                    dfs(i, component)
                    components.append(component)
            
            # Merge detections in each component
            for component in components:
                if len(component) == 1:
                    # Single detection, keep as is
                    idx = list(component)[0]
                    grouped_detections.append(dets[idx])
                else:
                    # Multiple detections, merge them
                    indices = list(component)
                    boxes = [dets[i]['box'] for i in indices]
                    confidences = [dets[i]['confidence'] for i in indices]
                    
                    # Calculate bounding box that fits all
                    x1_min = min(box[0] for box in boxes)
                    y1_min = min(box[1] for box in boxes)
                    x2_max = max(box[2] for box in boxes)
                    y2_max = max(box[3] for box in boxes)
                    
                    # Use maximum confidence
                    max_conf = max(confidences)
                    
                    # Create merged detection
                    merged = {
                        'box': [x1_min, y1_min, x2_max, y2_max],
                        'confidence': max_conf,
                        'class_id': dets[indices[0]]['class_id'],
                        'class_name': cls_name,
                        'grouped': True,
                        'group_count': len(component)
                    }
                    grouped_detections.append(merged)
        
        return grouped_detections
    
    def detect(self, image, iou_threshold=None, image_size=640):
        """
        Run detection on an image with signature grouping.
        
        Args:
            image: Input image (numpy array or path)
            iou_threshold: NMS IoU threshold (uses instance default if None)
            image_size: Input size for model (default: 640)
        
        Returns:
            dict: Detection results with boxes, labels, confidences
        """
        start_time = time.time()
        
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        # Get image dimensions
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        img_height, img_width = img.shape[:2]
        
        # Run inference
        results = self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse raw detections
        raw_detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(float, box)
                class_name = self.class_names[cls_id]
                
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_name": class_name,
                    "class_id": int(cls_id)
                }
                raw_detections.append(detection)
        
        # Apply grouping logic
        original_count = len(raw_detections)
        grouped_detections = self.group_detections(raw_detections)
        grouped_count = original_count - len(grouped_detections)
        
        # Categorize detections by type
        stamps = []
        signatures = []
        qrs = []
        
        for i, det in enumerate(grouped_detections):
            class_name = det['class_name']
            x1, y1, x2, y2 = det['box']
            
            detection_obj = {
                "id": f"{class_name}_{i+1}",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],  # [x_min, y_min, x_max, y_max]
                "confidence": det['confidence'],
                "class_name": class_name,
                "class_id": det['class_id']
            }
            
            # Add grouping metadata if applicable
            if det.get('grouped', False):
                detection_obj['grouped'] = True
                detection_obj['group_count'] = det.get('group_count', 2)
            
            # Categorize by type
            if class_name == "stamp":
                stamps.append(detection_obj)
            elif class_name == "signature":
                signatures.append(detection_obj)
            elif class_name == "qr":
                qrs.append(detection_obj)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = {
            "image_size": {
                "width_px": img_width,
                "height_px": img_height
            },
            "stamps": stamps,
            "signatures": signatures,
            "qrs": qrs,
            "summary": {
                "total_stamps": len(stamps),
                "total_signatures": len(signatures),
                "total_qrs": len(qrs),
                "total_detections": len(stamps) + len(signatures) + len(qrs),
                "raw_detections": original_count,
                "grouped_detections": grouped_count if grouped_count > 0 else None
            },
            "meta": {
                "model_version": os.path.basename(self.model_path),
                "inference_time_ms": round(inference_time, 2),
                "confidence_threshold": self.confidence_threshold,
                "grouping_enabled": self.enable_grouping,
                "group_iou_threshold": self.group_iou_threshold if self.enable_grouping else None
            }
        }
        
        return result
    
    def detect_batch(self, images, iou_threshold=0.45, image_size=1024):
        """
        Run detection on multiple images.
        
        Args:
            images: List of images (numpy arrays or paths)
            iou_threshold: NMS IoU threshold
            image_size: Input size for model
        
        Returns:
            list: List of detection results
        """
        return [self.detect(img, iou_threshold, image_size) for img in images]


# Global model instance (singleton pattern for API)
_detector_instance = None


def get_detector(model_path=None, confidence_threshold=0.25, device='cpu',
                enable_grouping=True, group_iou_threshold=0.3):
    """
    Get or create detector instance (singleton).
    
    Args:
        model_path: Path to model weights
        confidence_threshold: Detection threshold
        device: Device to run on
        enable_grouping: Enable signature grouping
        group_iou_threshold: IoU threshold for grouping
    
    Returns:
        DocumentDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = DocumentDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            enable_grouping=enable_grouping,
            group_iou_threshold=group_iou_threshold
        )
    
    return _detector_instance
