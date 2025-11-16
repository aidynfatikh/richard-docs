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
    
    def __init__(self, model_path=None, confidence_threshold=0.15, device='cpu', 
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
            print(f"Smart grouping enabled (distance: 50px, IoU: {self.group_iou_threshold}, containment: 0.75)")
    
    def _find_best_model(self):
        """Find the best production model."""
        # Use the best production model (76% mAP@50, 95% signature recall)
        model_path = Path(__file__).parent.parent.parent / "model" / "runs" / "train" / "yolov11s_lora_20251115_230142" / "weights" / "best.pt"
        
        if model_path.exists():
            return str(model_path)
        
        # Fallback: find most recent training run
        model_dir = Path(__file__).parent.parent.parent / "model" / "runs" / "train"
        
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Training runs directory not found: {model_dir}\n"
                "Please train a model first or specify model_path explicitly."
            )
        
        runs = sorted(model_dir.glob("yolov11s_*"), key=os.path.getmtime, reverse=True)
        
        if not runs:
            raise FileNotFoundError(f"No training runs found in {model_dir}")
        
        best_model = runs[0] / "weights" / "best.pt"
        
        if not best_model.exists():
            raise FileNotFoundError(f"best.pt not found in {runs[0]}/weights/")
        
        return str(best_model)
    
    def calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate center-to-center distance between two boxes"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min < x1_max or y2_min < y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_containment(self, box1: List[float], box2: List[float]) -> float:
        """Calculate how much box1 is contained within box2 (0.0 to 1.0)"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min < x1_max or y2_min < y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def boxes_are_close(self, box1: Dict, box2: Dict, distance_threshold: float = 50) -> bool:
        """Check if two boxes should be grouped together"""
        
        # Calculate distance between centers
        distance = self.calculate_distance(box1['box'], box2['box'])
        
        # Calculate IoU
        iou = self.calculate_iou(box1['box'], box2['box'])
        
        # Calculate containment (for signatures inside other signatures)
        containment1 = self.calculate_containment(box1['box'], box2['box'])
        containment2 = self.calculate_containment(box2['box'], box1['box'])
        max_containment = max(containment1, containment2)
        
        # Group if:
        # 1. Close enough (centers nearby)
        # 2. Overlapping enough (IoU)
        # 3. One box is mostly inside another (>60% containment - catches partial detections)
        return (distance < distance_threshold or 
                iou > self.group_iou_threshold or 
                max_containment > 0.6)
    
    def group_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Group nearby/overlapping detections of the SAME CLASS into unified detections.
        Uses two-pass strategy matching production inference.py:
        1. First pass: Remove boxes almost completely inside other boxes (duplicates)
        2. Second pass: Merge remaining nearby/overlapping boxes
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of detections with grouped boxes merged
        """
        if not self.enable_grouping or not detections:
            return detections
        
        # Group by class
        by_class = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in by_class:
                by_class[class_id] = []
            by_class[class_id].append(det)
        
        # Process each class separately
        merged_detections = []
        
        for class_id, class_detections in by_class.items():
            # Sort by confidence (highest first) then by area (largest first)
            class_detections = sorted(class_detections, 
                                     key=lambda x: (x['confidence'], 
                                                  (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1])), 
                                     reverse=True)
            
            # PASS 1: Remove boxes that are almost completely inside other boxes
            kept_detections = []
            for i, det1 in enumerate(class_detections):
                is_redundant = False
                
                for j, det2 in enumerate(class_detections):
                    if i == j:
                        continue
                    
                    containment = self.calculate_containment(det1['box'], det2['box'])
                    
                    # If >75% of this box is inside another box, check which one to keep
                    if containment > 0.75:
                        if det2['confidence'] > det1['confidence']:
                            is_redundant = True
                            break
                        elif det2['confidence'] == det1['confidence']:
                            # Same confidence - keep larger box
                            area1 = (det1['box'][2] - det1['box'][0]) * (det1['box'][3] - det1['box'][1])
                            area2 = (det2['box'][2] - det2['box'][0]) * (det2['box'][3] - det2['box'][1])
                            if area2 > area1:
                                is_redundant = True
                                break
                
                if not is_redundant:
                    kept_detections.append(det1)
            
            # PASS 2: Merge remaining nearby/overlapping boxes
            assigned = set()
            
            for i, det1 in enumerate(kept_detections):
                if i in assigned:
                    continue
                
                # Start new group for this detection
                group = [det1]
                assigned.add(i)
                
                # Find all same-class detections that overlap/are close to this one
                for j, det2 in enumerate(kept_detections):
                    if j in assigned:
                        continue
                    
                    # Check if det2 should be grouped with any detection in current group
                    should_group = False
                    for group_det in group:
                        if self.boxes_are_close(group_det, det2, distance_threshold=50):
                            should_group = True
                            break
                    
                    if should_group:
                        group.append(det2)
                        assigned.add(j)
                
                # Merge group into single detection
                if len(group) == 1:
                    # Single detection, keep as-is
                    merged_detections.append(group[0])
                else:
                    # Multiple overlapping detections - merge into one
                    all_boxes = [d['box'] for d in group]
                    x1 = min(box[0] for box in all_boxes)
                    y1 = min(box[1] for box in all_boxes)
                    x2 = max(box[2] for box in all_boxes)
                    y2 = max(box[3] for box in all_boxes)
                    
                    # Use highest confidence
                    max_conf = max(d['confidence'] for d in group)
                    
                    merged_detections.append({
                        'class_id': class_id,
                        'class_name': det1['class_name'],
                        'confidence': max_conf,
                        'box': [x1, y1, x2, y2],
                        'grouped': True,
                        'group_count': len(group)
                    })
        
        return merged_detections
    
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
    
    def detect_batch_optimized(self, images: List[np.ndarray], iou_threshold: Optional[float] = None, image_size: int = 640) -> List[Dict]:
        """
        Optimized batch detection using YOLO's native batch inference.
        Processes multiple images in a single forward pass for better performance.
        
        Args:
            images: List of image numpy arrays (BGR format)
            iou_threshold: NMS IoU threshold (uses instance default if None)
            image_size: Input size for model (default: 640)
        
        Returns:
            List of detection result dictionaries (same format as detect())
        """
        if not images:
            return []
        
        start_time = time.time()
        
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        # Run batch inference - YOLO handles batching internally
        results_batch = self.model.predict(
            source=images,
            conf=self.confidence_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False
        )
        
        # Process each result
        batch_results = []
        for img_idx, (img, results) in enumerate(zip(images, results_batch)):
            img_height, img_width = img.shape[:2]
            
            # Parse detections for this image
            raw_detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
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
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": det['confidence'],
                    "class_name": class_name,
                    "class_id": det['class_id']
                }
                
                if det.get('grouped', False):
                    detection_obj['grouped'] = True
                    detection_obj['group_count'] = det.get('group_count', 2)
                
                if class_name == "stamp":
                    stamps.append(detection_obj)
                elif class_name == "signature":
                    signatures.append(detection_obj)
                elif class_name == "qr":
                    qrs.append(detection_obj)
            
            # Build result dictionary
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
                    "confidence_threshold": self.confidence_threshold,
                    "grouping_enabled": self.enable_grouping,
                    "group_iou_threshold": self.group_iou_threshold if self.enable_grouping else None
                }
            }
            
            batch_results.append(result)
        
        # Add batch timing info
        total_inference_time = (time.time() - start_time) * 1000
        avg_time_per_image = total_inference_time / len(images)
        
        for i, result in enumerate(batch_results):
            result["meta"]["batch_inference_time_ms"] = round(total_inference_time, 2)
            result["meta"]["avg_time_per_image_ms"] = round(avg_time_per_image, 2)
            result["meta"]["batch_size"] = len(images)
            result["meta"]["image_index"] = i
        
        return batch_results


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


class RealTimeDetector:
    """
    Optimized detector for real-time video stream detection.
    Designed for 5-10 FPS performance on CPU with reduced latency.

    Optimizations:
    - Smaller input size (416px vs 640px)
    - Disabled signature grouping
    - Lightweight response format
    - Lower NMS threshold
    """

    def __init__(self, model_path=None, confidence_threshold=0.25, device='cpu',
                 image_size=416, iou_threshold=0.40):
        """
        Initialize real-time detector.

        Args:
            model_path: Path to YOLO weights. If None, uses default path.
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run on ('cpu', 'cuda', '0', '1', etc.)
            image_size: Input size for inference (default: 416 for speed)
            iou_threshold: NMS IoU threshold (default: 0.40, slightly lower for speed)
        """
        if model_path is None:
            # Use same model finder as DocumentDetector
            model_path = self._find_best_model()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size
        self.iou_threshold = iou_threshold

        print(f"Loading Real-Time YOLO model from: {model_path}")
        print(f"Real-Time Optimizations: image_size={image_size}, grouping=disabled")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Real-Time model loaded. Classes: {self.class_names}")

    def _find_best_model(self):
        """Find the best production model."""
        # Use the best production model (76% mAP@50, 95% signature recall)
        model_path = Path(__file__).parent.parent.parent / "model" / "runs" / "train" / "yolov11s_lora_20251115_230142" / "weights" / "best.pt"
        
        if model_path.exists():
            return str(model_path)
        
        # Fallback: find most recent training run
        model_dir = Path(__file__).parent.parent.parent / "model" / "runs" / "train"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Training runs directory not found: {model_dir}\n"
                "Please train a model first or specify model_path explicitly."
            )

        runs = sorted(model_dir.glob("yolov11s_*"), key=os.path.getmtime, reverse=True)

        if not runs:
            raise FileNotFoundError(f"No training runs found in {model_dir}")

        best_model = runs[0] / "weights" / "best.pt"

        if not best_model.exists():
            raise FileNotFoundError(f"best.pt not found in {runs[0]}/weights/")

        return str(best_model)

    def detect_frame(self, image):
        """
        Run detection on a video frame (optimized for speed).
        Returns lightweight response for real-time streaming.

        Args:
            image: Input image (numpy array)

        Returns:
            dict: Lightweight detection results {
                "coordinates": [...],
                "counts": {...},
                "inference_time_ms": float
            }
        """
        start_time = time.time()

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Run inference with optimized parameters
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,  # Smaller size for speed
            device=self.device,
            verbose=False,
            half=False  # Disable FP16 for CPU stability
        )[0]

        # Parse detections into lightweight format (coordinates only)
        coordinates = []
        counts = {"stamp": 0, "signature": 0, "qr": 0}

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)  # Convert to int for JSON
                class_name = self.class_names[cls_id]

                detection = {
                    "coordinates": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "normalized_coordinates": {
                        "x1": round(x1 / img_width, 6),
                        "y1": round(y1 / img_height, 6),
                        "x2": round(x2 / img_width, 6),
                        "y2": round(y2 / img_height, 6)
                    },
                    "confidence": round(float(conf), 3),  # Round to 3 decimals
                    "class": class_name
                }

                coordinates.append(detection)
                counts[class_name] = counts.get(class_name, 0) + 1

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "coordinates": coordinates,
            "counts": counts,
            "image_size": {"width": img_width, "height": img_height},
            "inference_time_ms": round(inference_time, 2)
        }


# Global real-time detector instance (singleton)
_realtime_detector_instance = None


def get_realtime_detector(model_path=None, confidence_threshold=0.25, device='cpu',
                          image_size=416, iou_threshold=0.40):
    """
    Get or create real-time detector instance (singleton).

    Args:
        model_path: Path to model weights
        confidence_threshold: Detection threshold
        device: Device to run on
        image_size: Input size for inference (default: 416)
        iou_threshold: NMS IoU threshold (default: 0.40)

    Returns:
        RealTimeDetector instance
    """
    global _realtime_detector_instance

    if _realtime_detector_instance is None:
        _realtime_detector_instance = RealTimeDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=image_size,
            iou_threshold=iou_threshold
        )

    return _realtime_detector_instance
