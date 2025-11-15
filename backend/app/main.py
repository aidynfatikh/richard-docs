from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
from typing import Optional, List
import time

from app.model_service import get_detector
from app.document_processor import get_processor

app = FastAPI(
    title="Document Detection API",
    description="API for detecting stamps, signatures, and QR codes in documents",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services on startup
detector = None
processor = None

@app.on_event("startup")
async def startup_event():
    """Load the YOLO model and document processor when the API starts."""
    global detector, processor
    try:
        print("Initializing document detector...")
        detector = get_detector(
            model_path=None,  # Auto-finds best model
            confidence_threshold=0.25,
            device='cpu',  # Change to 'cuda' or '0' if GPU available
            enable_grouping=True,  # Enable signature grouping
            group_iou_threshold=0.3  # IoU threshold for grouping
        )
        print("✓ Model loaded successfully")
        
        print("Initializing document processor...")
        processor = get_processor(dpi=200)  # 200 DPI for PDF rendering
        print("✓ Document processor initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not load services: {e}")
        print("API will return errors until services are available")


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "Document Detection API",
        "status": "running",
        "model_loaded": detector is not None
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    health_info = {
        "status": "healthy" if detector is not None else "model_not_loaded",
        "model_loaded": detector is not None,
        "model_path": detector.model_path if detector else None,
        "classes": detector.class_names if detector else None
    }
    
    if detector is not None:
        health_info["grouping_enabled"] = detector.enable_grouping
        health_info["group_iou_threshold"] = detector.group_iou_threshold
        health_info["confidence_threshold"] = detector.confidence_threshold
    
    return health_info


@app.post("/detect")
async def detect_elements(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Detect document elements (stamps, signatures, QR codes) in an uploaded document.
    Supports images (JPG, PNG, WEBP, TIFF, BMP) and PDFs (multi-page).
    
    Args:
        file: Document file (image or PDF)
        confidence: Confidence threshold (0-1), default 0.25
    
    Returns:
        JSON with detected elements and their bounding boxes.
        For PDFs with multiple pages, returns array of results per page.
    """
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )
    
    # Read file contents
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {str(e)}"
        )
    
    # Check if format is supported
    format_info = processor.get_format_info(file.filename or "document")
    if not format_info['is_supported']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {format_info['extension']}. "
                   f"Supported: JPG, PNG, WEBP, TIFF, BMP, PDF"
        )
    
    # Process document (handles both images and PDFs)
    try:
        pages = processor.process_file(contents, file.filename or "document")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    
    # Update detector confidence if provided
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence
    
    # Run detection on all pages
    try:
        start_time = time.time()
        results = []
        
        for img, page_num, base64_img in pages:
            page_start = time.time()
            
            print(f"Processing page {page_num}: base64_img exists = {base64_img is not None}, length = {len(base64_img) if base64_img else 0}")
            
            # Run detection
            result = detector.detect(img)
            
            # Add page-specific metadata
            result["meta"]["page_number"] = page_num
            result["meta"]["page_processing_time_ms"] = round((time.time() - page_start) * 1000, 2)
            
            # Add base64 image if available (for PDFs)
            if base64_img:
                result["page_image"] = base64_img
                print(f"Added page_image to result for page {page_num}, prefix: {base64_img[:50]}")
            else:
                print(f"No base64_img for page {page_num}")
            
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        # Restore original confidence
        detector.confidence_threshold = original_confidence
        
        # Return single result for images, array for multi-page PDFs
        if len(results) == 1:
            results[0]["meta"]["total_processing_time_ms"] = round(total_time, 2)
            results[0]["meta"]["confidence_threshold"] = confidence
            results[0]["meta"]["is_pdf"] = format_info['is_pdf']
            return results[0]
        else:
            # Multi-page PDF
            return {
                "document_type": "pdf",
                "total_pages": len(results),
                "pages": results,
                "summary": {
                    "total_detections": sum(r["summary"]["total_detections"] for r in results),
                    "total_stamps": sum(r["summary"]["total_stamps"] for r in results),
                    "total_signatures": sum(r["summary"]["total_signatures"] for r in results),
                    "total_qrs": sum(r["summary"]["total_qrs"] for r in results),
                },
                "meta": {
                    "total_processing_time_ms": round(total_time, 2),
                    "confidence_threshold": confidence,
                    "is_pdf": True
                }
            }
        
    except Exception as e:
        # Restore confidence on error
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)