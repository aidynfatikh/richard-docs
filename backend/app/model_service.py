"""
YOLO Model Service
Handles loading and inference for document element detection (stamps, signatures, QR codes)
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class DocumentDetector:
    """
    Document element detector using YOLOv11.
    Detects stamps, signatures, and QR codes.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25, device='cpu'):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO weights. If None, uses default path.
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run on ('cpu', 'cuda', '0', '1', etc.)
        """
        if model_path is None:
            # Default to best model from training
            model_path = self._find_best_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Model loaded. Classes: {self.class_names}")
    
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
    
    def detect(self, image, iou_threshold=0.45, image_size=1024):
        """
        Run detection on an image.
        
        Args:
            image: Input image (numpy array or path)
            iou_threshold: NMS IoU threshold
            image_size: Input size for model
        
        Returns:
            dict: Detection results with boxes, labels, confidences
        """
        start_time = time.time()
        
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
            verbose=False
        )[0]
        
        # Parse detections by category
        stamps = []
        signatures = []
        qrs = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(float, box)
                class_name = self.class_names[cls_id]
                
                detection = {
                    "id": f"{class_name}_{i+1}",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],  # [x_min, y_min, x_max, y_max]
                    "confidence": float(conf),
                    "class_name": class_name,
                    "class_id": int(cls_id)
                }
                
                # Categorize by type
                if class_name == "stamp":
                    stamps.append(detection)
                elif class_name == "signature":
                    signatures.append(detection)
                elif class_name == "qr":
                    qrs.append(detection)
        
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
                "total_detections": len(stamps) + len(signatures) + len(qrs)
            },
            "meta": {
                "model_version": os.path.basename(self.model_path),
                "inference_time_ms": round(inference_time, 2),
                "confidence_threshold": self.confidence_threshold
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


def get_detector(model_path=None, confidence_threshold=0.25, device='cpu'):
    """
    Get or create detector instance (singleton).
    
    Args:
        model_path: Path to model weights
        confidence_threshold: Detection threshold
        device: Device to run on
    
    Returns:
        DocumentDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = DocumentDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
    
    return _detector_instance
