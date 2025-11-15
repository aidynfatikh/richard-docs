from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
from typing import Optional
import time

from app.model_service import get_detector

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

# Initialize model on startup
detector = None

@app.on_event("startup")
async def startup_event():
    """Load the YOLO model when the API starts."""
    global detector
    try:
        print("Initializing document detector...")
        detector = get_detector(
            model_path=None,  # Auto-finds best model
            confidence_threshold=0.25,
            device='cpu'  # Change to 'cuda' or '0' if GPU available
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("API will return errors until model is available")


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
    return {
        "status": "healthy" if detector is not None else "model_not_loaded",
        "model_loaded": detector is not None,
        "model_path": detector.model_path if detector else None,
        "classes": detector.class_names if detector else None
    }


@app.post("/detect")
async def detect_elements(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25
):
    """
    Detect document elements (stamps, signatures, QR codes) in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
        confidence: Confidence threshold (0-1), default 0.25
    
    Returns:
        JSON with detected elements and their bounding boxes
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Read and decode image
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid image."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}"
        )
    
    # Update detector confidence if provided
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence
    
    # Run detection
    try:
        start_time = time.time()
        result = detector.detect(img)
        total_time = (time.time() - start_time) * 1000
        
        # Add request metadata
        result["meta"]["total_processing_time_ms"] = round(total_time, 2)
        result["meta"]["confidence_threshold"] = confidence
        
        # Restore original confidence
        detector.confidence_threshold = original_confidence
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)