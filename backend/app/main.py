from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
from typing import Optional, List
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.model_service import get_detector, get_realtime_detector
from app.document_processor import get_processor
from app.document_classifier import is_document_photo
import sys
import os
# Add document_scan module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'document_scan'))
from scan import DocScanner
import base64

app = FastAPI(
    title="Document Detection API",
    description="API for detecting stamps, signatures, and QR codes in documents",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers including ngrok-skip-browser-warning
    expose_headers=["*"],  # Expose all response headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Initialize services on startup
detector = None
realtime_detector = None
processor = None
doc_scanner = DocScanner()

@app.on_event("startup")
async def startup_event():
    """Load the YOLO model and document processor when the API starts."""
    global detector, realtime_detector, processor, doc_scanner
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

        print("Initializing real-time detector...")
        realtime_detector = get_realtime_detector(
            model_path=None,  # Uses same model as detector
            confidence_threshold=0.25,
            device='cpu',  # Change to 'cuda' or '0' if GPU available
            image_size=416,  # Smaller size for speed
            iou_threshold=0.40  # Lower NMS threshold for speed
        )
        print("✓ Real-time detector loaded successfully")

        print("Initializing document processor...")
        processor = get_processor(dpi=200)  # 200 DPI for PDF rendering
        print("✓ Document processor initialized")

        print("Initializing document scanner...")
        doc_scanner = DocScanner(interactive=False)
        print("✓ Document scanner initialized")
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



@app.post("/scan-document")
async def scan_document(file: UploadFile = File(...)):
    """
    Scan a document image, detect boundaries, apply perspective transform
    and return the cleaned B/W scanned version.
    """

    # гарантированная проверка
    if doc_scanner is None:
        raise HTTPException(
            status_code=503,
            detail="Document scanner not loaded"
        )

    # читаем файл
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Empty file")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # запускаем OCR-сканер
    try:
        result = doc_scanner.scan_image_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal scanning error: {str(e)}"
        )

    # если DocScanner вернул ошибку
    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Document scanning failed")
        )

    # конвертация результата в base64
    transformed_b64 = base64.b64encode(result["transformed_image"]).decode("utf-8")

    return {
        "success": True,
        "corners": result["corners"],
        "transformed_image": f"data:image/jpeg;base64,{transformed_b64}"
    }


@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25,
    force_scan: Optional[bool] = False,
    skip_scan: Optional[bool] = False
):
    """
    Intelligent document processing endpoint.
    Automatically classifies upload as camera photo vs digital document,
    then applies appropriate processing pipeline.

    Workflow:
    1. Classify document (camera photo or digital document)
    2. If camera photo → apply perspective correction (DocScanner)
    3. If digital document → use as-is
    4. Run object detection (stamps, signatures, QR codes)
    5. Return results with classification metadata

    Args:
        file: Uploaded document file (image or PDF)
        confidence: Detection confidence threshold (0-1), default 0.25
        force_scan: Force perspective correction even if classified as digital
        skip_scan: Skip perspective correction even if classified as camera photo

    Returns:
        JSON with:
        - classification: camera_photo or digital_document
        - scan_applied: whether perspective correction was applied
        - detections: stamps, signatures, QR codes
        - metadata: processing details
    """
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )

    # Read file contents
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
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

    start_time = time.time()
    processing_metadata = {
        'filename': file.filename,
        'file_size_bytes': len(contents),
        'format': format_info['extension']
    }

    # Step 1: Classify document (only for images, not PDFs)
    classification_result = None
    should_scan = False

    if not format_info['is_pdf']:
        print(f"\n[Classifier] Analyzing document type for {file.filename}...")
        is_camera_photo, classification_info = is_document_photo(contents)

        classification_result = classification_info
        processing_metadata['classification'] = classification_info

        # Decision logic
        if force_scan:
            should_scan = True
            processing_metadata['scan_reason'] = 'force_scan_flag'
        elif skip_scan:
            should_scan = False
            processing_metadata['scan_reason'] = 'skip_scan_flag'
        else:
            should_scan = is_camera_photo
            processing_metadata['scan_reason'] = f"classifier_decision_{classification_info['classification']}"

        print(f"[Classifier] Result: {classification_info['classification']} "
              f"(confidence: {classification_info['confidence']:.2f})")
        print(f"[Classifier] Decision: {'APPLY' if should_scan else 'SKIP'} perspective correction")
    else:
        # PDFs are never scanned (they're digital by definition)
        processing_metadata['classification'] = {
            'classification': 'digital_document',
            'confidence': 1.0,
            'reason': 'pdf_format'
        }
        should_scan = False if not force_scan else True
        processing_metadata['scan_reason'] = 'pdf_never_scanned' if not force_scan else 'force_scan_override'

    # Step 2: Apply perspective correction if needed
    processed_contents = contents
    scan_metadata = None
    transformed_image_b64 = None  # Will hold base64-encoded scanned image

    if should_scan and doc_scanner is not None:
        print(f"[Scanner] Applying perspective correction...")
        try:
            scan_result = doc_scanner.scan_image_bytes(contents)

            if scan_result['success']:
                # Use scanned image for detection
                processed_contents = scan_result['transformed_image']

                # Encode transformed image to base64 for frontend display
                transformed_b64 = base64.b64encode(scan_result['transformed_image']).decode("utf-8")
                transformed_image_b64 = f"data:image/jpeg;base64,{transformed_b64}"

                scan_metadata = {
                    'applied': True,
                    'corners_detected': scan_result.get('corners'),
                    'scan_success': True
                }
                print(f"[Scanner] ✓ Perspective correction applied successfully")
            else:
                # Scan failed, fall back to original
                print(f"[Scanner] ⚠ Scan failed: {scan_result.get('error', 'Unknown error')}")
                print(f"[Scanner] Falling back to original image")
                scan_metadata = {
                    'applied': False,
                    'error': scan_result.get('error'),
                    'scan_success': False,
                    'fallback_to_original': True
                }
        except Exception as e:
            print(f"[Scanner] ⚠ Exception during scanning: {str(e)}")
            scan_metadata = {
                'applied': False,
                'error': str(e),
                'scan_success': False,
                'fallback_to_original': True
            }
    else:
        scan_metadata = {
            'applied': False,
            'reason': 'not_camera_photo' if not should_scan else 'scanner_not_loaded'
        }

    processing_metadata['scan'] = scan_metadata

    # Step 3: Process document (handles both images and PDFs)
    try:
        pages = processor.process_file(processed_contents, file.filename or "document")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

    # Step 4: Run detection on all pages
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    try:
        results = []

        for img, page_num, base64_img in pages:
            page_start = time.time()

            # Run detection
            result = detector.detect(img)

            # Add page-specific metadata
            result["meta"]["page_number"] = page_num
            result["meta"]["page_processing_time_ms"] = round((time.time() - page_start) * 1000, 2)

            # Add base64 image if available (for PDFs)
            if base64_img:
                result["page_image"] = base64_img

            results.append(result)

        total_time = (time.time() - start_time) * 1000

        # Restore original confidence
        detector.confidence_threshold = original_confidence

        # Step 5: Build comprehensive response
        if len(results) == 1:
            # Single page/image
            response = results[0]
            response["meta"]["total_processing_time_ms"] = round(total_time, 2)
            response["meta"]["confidence_threshold"] = confidence
            response["meta"]["is_pdf"] = format_info['is_pdf']
            response["processing"] = processing_metadata

            # Include transformed image if perspective correction was applied
            if transformed_image_b64:
                response["transformed_image"] = transformed_image_b64

            return response
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
                },
                "processing": processing_metadata
            }

    except Exception as e:
        # Restore confidence on error
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/classify-document")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify document as camera photo vs digital document.
    Returns classification with detailed metadata - no processing applied.

    This endpoint is useful for:
    - Testing the classifier
    - Getting classification confidence before processing
    - Debugging classification decisions

    Args:
        file: Image file to classify

    Returns:
        JSON with classification result and detailed indicators
    """
    # Read file
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {str(e)}"
        )

    # Classify
    try:
        is_camera_photo, classification_info = is_document_photo(contents)

        return {
            "filename": file.filename,
            "file_size_bytes": len(contents),
            "is_camera_photo": is_camera_photo,
            "classification": classification_info['classification'],
            "confidence": classification_info['confidence'],
            "recommendation": classification_info['recommendation'],
            "details": classification_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


async def process_single_file_batch(
    file: UploadFile,
    idx: int,
    confidence: float,
    force_scan: bool = False,
    skip_scan: bool = False
):
    """Process a single file for batch detection with intelligent processing."""
    try:
        contents = await file.read()
        if not contents:
            return {
                "file_index": idx,
                "filename": file.filename,
                "success": False,
                "error": "Empty file"
            }
        
        # Check format support
        format_info = processor.get_format_info(file.filename or "document")
        if not format_info['is_supported']:
            return {
                "file_index": idx,
                "filename": file.filename,
                "success": False,
                "error": f"Unsupported format: {format_info['extension']}"
            }
        
        # Handle PDFs with multi-page support
        if format_info['is_pdf']:
            try:
                pages_data = processor.process_pdf_bytes(contents)
                
                # Process each page in parallel
                page_results = []
                for page_num, page_data in enumerate(pages_data):
                    img = cv2.imdecode(
                        np.frombuffer(base64.b64decode(page_data['image'].split(',')[1]), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    detection_result = detector.detect(img)
                    detection_result['page_number'] = page_num + 1
                    detection_result['annotated_image'] = page_data['image']
                    page_results.append(detection_result)
                
                # Aggregate summary
                total_detections = sum(p['summary']['total_detections'] for p in page_results)
                total_stamps = sum(p['summary']['total_stamps'] for p in page_results)
                total_signatures = sum(p['summary']['total_signatures'] for p in page_results)
                total_qrs = sum(p['summary']['total_qrs'] for p in page_results)
                
                return {
                    "file_index": idx,
                    "filename": file.filename,
                    "success": True,
                    "document_type": "multi_page_pdf",
                    "total_pages": len(pages_data),
                    "pages": page_results,
                    "summary": {
                        "total_detections": total_detections,
                        "total_stamps": total_stamps,
                        "total_signatures": total_signatures,
                        "total_qrs": total_qrs
                    }
                }
            except Exception as e:
                return {
                    "file_index": idx,
                    "filename": file.filename,
                    "success": False,
                    "error": f"PDF processing failed: {str(e)}"
                }
        
        # Handle images with optional scanning
        else:
            # Classify document
            is_camera_photo, classification_info = is_document_photo(contents)
            
            should_scan = False
            if force_scan:
                should_scan = True
            elif not skip_scan and is_camera_photo:
                should_scan = True
            
            processed_contents = contents
            scan_applied = False
            
            # Apply perspective correction if needed
            if should_scan and doc_scanner is not None:
                try:
                    scan_result = doc_scanner.scan_image_bytes(contents)
                    if scan_result['success']:
                        processed_contents = scan_result['transformed_image']
                        scan_applied = True
                except Exception:
                    pass  # Fall back to original
            
            # Decode and detect
            nparr = np.frombuffer(processed_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "file_index": idx,
                    "filename": file.filename,
                    "success": False,
                    "error": "Failed to decode image"
                }
            
            detection_result = detector.detect(img)
            detection_result["file_index"] = idx
            detection_result["filename"] = file.filename
            detection_result["success"] = True
            detection_result["classification"] = classification_info['classification']
            detection_result["scan_applied"] = scan_applied
            
            return detection_result
            
    except Exception as e:
        return {
            "file_index": idx,
            "filename": file.filename,
            "success": False,
            "error": str(e)
        }


@app.post("/batch-detect")
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = 0.25,
    force_scan: Optional[bool] = False,
    skip_scan: Optional[bool] = False,
    max_workers: Optional[int] = 10
):
    """
    HIGH-PERFORMANCE batch detection for 1000+ files.
    Supports images and PDFs with intelligent processing and parallel execution.

    Args:
        files: List of document files (images/PDFs)
        confidence: Confidence threshold (0-1), default 0.25
        force_scan: Force perspective correction on all images
        skip_scan: Skip perspective correction even for camera photos
        max_workers: Maximum parallel workers (default 10, max 50)

    Returns:
        JSON with batch results and aggregate statistics
    """
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    if len(files) > 2000:  # Increased limit for hackathon
        raise HTTPException(
            status_code=400,
            detail="Too many files. Maximum 2000 files per batch."
        )

    # Limit max_workers
    max_workers = min(max_workers, 50)

    # Update detector confidence if provided
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    try:
        start_time = time.time()
        
        # Process files in parallel with semaphore to control concurrency
        tasks = [
            process_single_file_batch(file, idx, confidence, force_scan, skip_scan)
            for idx, file in enumerate(files)
        ]
        
        # Process in batches to avoid overwhelming the system
        batch_size = max_workers
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "file_index": i + j,
                        "filename": files[i + j].filename,
                        "success": False,
                        "error": str(result)
                    })
                else:
                    results.append(result)
        
        total_time = (time.time() - start_time) * 1000

        # Restore original confidence
        detector.confidence_threshold = original_confidence

        # Calculate overall summary
        successful_results = [r for r in results if r.get("success", False)]
        
        # Aggregate stats (handle both single-page and multi-page)
        total_detections = 0
        total_stamps = 0
        total_signatures = 0
        total_qrs = 0
        
        for r in successful_results:
            if "summary" in r:
                total_detections += r["summary"].get("total_detections", 0)
                total_stamps += r["summary"].get("total_stamps", 0)
                total_signatures += r["summary"].get("total_signatures", 0)
                total_qrs += r["summary"].get("total_qrs", 0)

        return {
            "total_files": len(files),
            "successful_detections": len(successful_results),
            "failed_detections": len(files) - len(successful_results),
            "results": results,
            "summary": {
                "total_detections": total_detections,
                "total_stamps": total_stamps,
                "total_signatures": total_signatures,
                "total_qrs": total_qrs,
            },
            "meta": {
                "total_processing_time_ms": round(total_time, 2),
                "avg_time_per_file_ms": round(total_time / len(files), 2),
                "confidence_threshold": confidence,
                "parallel_workers": max_workers
            }
        }

    except Exception as e:
        # Restore confidence on error
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream detection.

    Protocol:
    - Client sends: Base64-encoded JPEG frames
    - Server responds: JSON with detections, counts, and performance metrics

    Message Format:
    Client -> Server: {"frame": "base64_encoded_jpeg_string"}
    Server -> Client: {
        "detections": [...],
        "counts": {"stamp": 0, "signature": 0, "qr": 0},
        "image_size": {"width": 640, "height": 480},
        "inference_time_ms": 123.45
    }
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    if realtime_detector is None:
        await websocket.send_json({
            "error": "Real-time detector not initialized",
            "message": "Server is still starting up. Please wait and reconnect."
        })
        await websocket.close()
        return

    frame_count = 0
    total_inference_time = 0

    try:
        while True:
            # Receive frame from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "message": "Expected JSON with 'frame' field containing base64 image"
                })
                continue

            if "frame" not in message:
                await websocket.send_json({
                    "error": "Missing 'frame' field",
                    "message": "Message must contain 'frame' field with base64-encoded image"
                })
                continue

            # Decode base64 frame
            try:
                # Handle data URL format (data:image/jpeg;base64,...)
                frame_data = message["frame"]
                if "base64," in frame_data:
                    frame_data = frame_data.split("base64,")[1]

                # Decode base64 to bytes
                img_bytes = base64.b64decode(frame_data)

                # Convert to numpy array
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({
                        "error": "Failed to decode image",
                        "message": "Could not decode base64 data as image"
                    })
                    continue

            except Exception as e:
                await websocket.send_json({
                    "error": "Image decoding error",
                    "message": str(e)
                })
                continue

            # Run detection on frame
            try:
                result = realtime_detector.detect_frame(frame)

                # Track performance metrics
                frame_count += 1
                total_inference_time += result["inference_time_ms"]
                avg_inference_time = total_inference_time / frame_count

                # Add performance metrics to response
                result["meta"] = {
                    "frame_count": frame_count,
                    "avg_inference_time_ms": round(avg_inference_time, 2)
                }

                # Send results back to client
                await websocket.send_json(result)

                # Log every 10 frames for monitoring
                if frame_count % 10 == 0:
                    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
                    print(f"[WebSocket] Frame {frame_count}: "
                          f"{len(result['coordinates'])} detections, "
                          f"{result['inference_time_ms']:.1f}ms, "
                          f"avg FPS: {fps:.1f}")

            except Exception as e:
                print(f"[WebSocket] Detection error: {e}")
                await websocket.send_json({
                    "error": "Detection failed",
                    "message": str(e)
                })
                continue

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected. Total frames processed: {frame_count}")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
        try:
            await websocket.send_json({
                "error": "Server error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
