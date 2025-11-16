from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
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
    allow_origins=[
        "*",  # Allow all origins for development
        "https://docs.richardsai.tech",  # Production frontend
        "http://localhost:5173",  # Local Vite dev server
        "http://localhost:3000",  # Alternative local port
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers including ngrok-skip-browser-warning
    expose_headers=["*"],  # Expose all response headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Middleware to add CORS headers for ngrok
class NgrokCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        return response

app.add_middleware(NgrokCORSMiddleware)

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
            model_path=None,  # Auto-finds best model (yolov11s_lora_20251115_230142)
            confidence_threshold=0.15,
            device='cpu',  # Change to 'cuda' or '0' if GPU available
            enable_grouping=True,  # Enable smart grouping (distance + IoU + containment)
            group_iou_threshold=0.2  # IoU threshold for grouping (matches inference.py)
        )
        print("✓ Model loaded successfully")

        print("Initializing real-time detector...")
        realtime_detector = get_realtime_detector(
            model_path=None,  # Uses same model as detector
            confidence_threshold=0.15,
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
    confidence: Optional[float] = 0.15
):
    """
    Detect document elements (stamps, signatures, QR codes) in an uploaded document.
    Supports images (JPG, PNG, WEBP, TIFF, BMP) and PDFs (multi-page).
    
    Args:
        file: Document file (image or PDF)
        confidence: Confidence threshold (0-1), default 0.15
    
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
        # Use lazy base64 encoding - only encode for response
        pages = await asyncio.to_thread(
            processor.process_file, 
            contents, 
            file.filename or "document"
        )
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
        
        # Extract images for batch processing
        images = [img for img, _, _ in pages]
        
        # Use batch inference for better performance
        if len(images) > 1:
            # Batch inference for multi-page PDFs
            results = await asyncio.to_thread(
                detector.detect_batch_optimized,
                images,
                iou_threshold=0.45
            )
        else:
            # Single image detection
            results = [await asyncio.to_thread(detector.detect, images[0], iou_threshold=0.45)]
        
        # Add page metadata and base64 images
        for i, (result, (_, page_num, base64_img)) in enumerate(zip(results, pages)):
            result["meta"]["page_number"] = page_num
            
            # Generate base64 for PDFs if not already present
            if base64_img:
                result["page_image"] = base64_img
            elif format_info['is_pdf']:
                # Encode page image for PDF pages
                img = images[i]
                success, encoded_img = cv2.imencode('.png', img)
                if success:
                    img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                    result["page_image"] = f"data:image/png;base64,{img_base64}"
        
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
                    "is_pdf": True,
                    "batch_optimized": True
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
    confidence: Optional[float] = 0.15,
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
        confidence: Detection confidence threshold (0-1), default 0.15
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
        pages = await asyncio.to_thread(
            processor.process_file, 
            processed_contents, 
            file.filename or "document"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

    # Step 4: Run detection on all pages with batch optimization
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    try:
        # Extract images for batch processing
        images = [img for img, _, _ in pages]
        
        # Use batch inference for multi-page documents
        if len(images) > 1:
            results = await asyncio.to_thread(
                detector.detect_batch_optimized,
                images,
                iou_threshold=0.45
            )
        else:
            results = [await asyncio.to_thread(detector.detect, images[0], iou_threshold=0.45)]
        
        # Add page metadata and base64 images
        for i, (result, (_, page_num, base64_img)) in enumerate(zip(results, pages)):
            result["meta"]["page_number"] = page_num
            
            # Add base64 image if available (for PDFs)
            if base64_img:
                result["page_image"] = base64_img

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


def prepare_document_for_intelligent_processing(file_data, force_scan: bool, skip_scan: bool):
    """
    Prepare a document for intelligent batch processing with scanning and classification.
    Runs the same pipeline as /process-document but returns intermediate artifacts
    so detections can be executed in batch.
    """
    idx = file_data["idx"]
    filename = file_data["filename"]
    contents = file_data["contents"]
    format_info = file_data["format_info"]

    processing_metadata = {
        "filename": filename,
        "file_size_bytes": len(contents),
        "format": format_info["extension"]
    }

    start_time = time.time()
    transformed_image_b64 = None
    should_scan = False

    try:
        # Step 1: Classification (images only)
        if not format_info['is_pdf']:
            is_camera_photo, classification_info = is_document_photo(contents)
            processing_metadata['classification'] = classification_info

            if force_scan:
                should_scan = True
                processing_metadata['scan_reason'] = 'force_scan_flag'
            elif skip_scan:
                should_scan = False
                processing_metadata['scan_reason'] = 'skip_scan_flag'
            else:
                should_scan = is_camera_photo
                processing_metadata['scan_reason'] = f"classifier_decision_{classification_info['classification']}"
        else:
            processing_metadata['classification'] = {
                'classification': 'digital_document',
                'confidence': 1.0,
                'reason': 'pdf_format'
            }
            should_scan = bool(force_scan)
            processing_metadata['scan_reason'] = 'force_scan_override' if force_scan else 'pdf_never_scanned'

        # Step 2: Perspective correction if needed
        processed_contents = contents
        scan_metadata = {
            'applied': False,
            'reason': 'not_camera_photo'
        }

        if should_scan:
            if doc_scanner is None:
                scan_metadata = {
                    'applied': False,
                    'reason': 'scanner_not_loaded'
                }
            else:
                try:
                    scan_result = doc_scanner.scan_image_bytes(contents)
                    if scan_result['success']:
                        processed_contents = scan_result['transformed_image']
                        transformed_b64 = base64.b64encode(scan_result['transformed_image']).decode("utf-8")
                        transformed_image_b64 = f"data:image/jpeg;base64,{transformed_b64}"
                        scan_metadata = {
                            'applied': True,
                            'corners_detected': scan_result.get('corners'),
                            'scan_success': True
                        }
                    else:
                        scan_metadata = {
                            'applied': False,
                            'error': scan_result.get('error'),
                            'scan_success': False,
                            'fallback_to_original': True
                        }
                except Exception as scan_error:
                    scan_metadata = {
                        'applied': False,
                        'error': str(scan_error),
                        'scan_success': False,
                        'fallback_to_original': True
                    }
        else:
            scan_metadata = {
                'applied': False,
                'reason': processing_metadata.get('scan_reason', 'scan_not_required')
            }

        processing_metadata['scan'] = scan_metadata

        # Step 3: Convert to images/pages (handles PDFs and images uniformly)
        pages = processor.process_file(processed_contents, filename or "document")
        prepared_pages = []

        for img, page_num, base64_img in pages:
            prepared_pages.append({
                "page_number": page_num,
                "image": img,
                "base64_image": base64_img
            })

        if not prepared_pages:
            raise ValueError("No pages generated from document")

        processing_metadata['page_count'] = len(prepared_pages)
        processing_metadata['preprocessing_time_ms'] = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "file_index": idx,
            "filename": filename,
            "document_type": "pdf" if format_info['is_pdf'] or len(prepared_pages) > 1 else "image",
            "format_info": format_info,
            "processing": processing_metadata,
            "transformed_image": transformed_image_b64,
            "pages": prepared_pages
        }

    except Exception as e:
        return {
            "success": False,
            "file_index": idx,
            "filename": filename,
            "error": str(e)
        }


async def read_and_prepare_file(file: UploadFile, idx: int):
    """Read file and prepare metadata (fast I/O operation)."""
    try:
        contents = await file.read()
        if not contents:
            return None, {"file_index": idx, "filename": file.filename, "success": False, "error": "Empty file"}
        
        format_info = processor.get_format_info(file.filename or "document")
        if not format_info['is_supported']:
            return None, {"file_index": idx, "filename": file.filename, "success": False, "error": f"Unsupported format: {format_info['extension']}"}
        
        return {
            "idx": idx,
            "filename": file.filename,
            "contents": contents,
            "format_info": format_info
        }, None
    except Exception as e:
        return None, {"file_index": idx, "filename": file.filename, "success": False, "error": str(e)}


def process_single_file_cpu(file_data, force_scan: bool, skip_scan: bool):
    """Process a single file (CPU-intensive, runs in parallel)."""
    idx = file_data["idx"]
    filename = file_data["filename"]
    contents = file_data["contents"]
    format_info = file_data["format_info"]
    
    images = []
    metadata_list = []
    
    try:
        # Handle PDFs
        if format_info['is_pdf']:
            pages_data = processor.pdf_to_images(contents, enable_base64=False, parallel=True)
            for img, page_num, _ in pages_data:
                images.append(img)
                metadata_list.append({
                    "file_index": idx,
                    "filename": filename,
                    "document_type": "pdf",
                    "page_number": page_num,
                    "total_pages": len(pages_data)
                })
        # Handle images
        else:
            # Classify with fast_mode (EXIF-only, 10x faster)
            is_camera_photo, classification_info = is_document_photo(contents, fast_mode=True)
            should_scan = force_scan or (not skip_scan and is_camera_photo)
            
            processed_contents = contents
            scan_applied = False
            
            if should_scan and doc_scanner is not None:
                try:
                    scan_result = doc_scanner.scan_image_bytes(contents)
                    if scan_result['success']:
                        processed_contents = scan_result['transformed_image']
                        scan_applied = True
                except Exception:
                    pass
            
            # Decode image
            nparr = np.frombuffer(processed_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
                metadata_list.append({
                    "file_index": idx,
                    "filename": filename,
                    "document_type": "image",
                    "classification": classification_info['classification'],
                    "scan_applied": scan_applied
                })
            else:
                metadata_list.append({
                    "file_index": idx,
                    "filename": filename,
                    "success": False,
                    "error": "Failed to decode image",
                    "skip_detection": True
                })
    except Exception as e:
        metadata_list.append({
            "file_index": idx,
            "filename": filename,
            "success": False,
            "error": str(e),
            "skip_detection": True
        })
    
    return images, metadata_list


def process_files_batch_cpu(file_data_list, force_scan: bool, skip_scan: bool):
    """CPU-intensive batch processing with parallel workers."""
    import multiprocessing
    from functools import partial
    
    # Use ThreadPoolExecutor for I/O-bound operations (image decoding, classification)
    # ThreadPool is better than ProcessPool here because:
    # 1. No pickling overhead (large image data)
    # 2. Shared memory access to processor, doc_scanner
    # 3. Better for I/O-bound ops (cv2.imdecode, EXIF reading)
    
    max_workers = min(multiprocessing.cpu_count(), len(file_data_list), 16)
    
    all_images = []
    all_metadata = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process files in parallel
        process_func = partial(process_single_file_cpu, force_scan=force_scan, skip_scan=skip_scan)
        results = list(executor.map(process_func, file_data_list))
    
    # Merge results in order
    for images, metadata_list in results:
        all_images.extend(images)
        all_metadata.extend(metadata_list)
    
    return all_images, all_metadata


def run_batch_detection(all_images, all_metadata, iou_threshold=0.45):
    """Run YOLO batch detection on all images at once."""
    if not all_images:
        return []
    
    # Single batch inference for ALL images (much faster)
    detection_results = detector.detect_batch_optimized(all_images, iou_threshold=iou_threshold, image_size=640)
    
    # Merge detection results with metadata and encode images
    results = []
    for img, detection_result, metadata in zip(all_images, detection_results, all_metadata):
        if metadata.get("skip_detection"):
            results.append(metadata)
        else:
            result = {
                "file_index": metadata["file_index"],
                "filename": metadata["filename"],
                "success": True,
                **detection_result
            }
            
            # Add extra metadata
            if metadata["document_type"] == "pdf":
                result["page_number"] = metadata["page_number"]
                result["document_type"] = "pdf_page"
                
                # Encode PDF page image to base64 for frontend display
                success, encoded_img = cv2.imencode('.png', img)
                if success:
                    img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                    result["page_image"] = f"data:image/png;base64,{img_base64}"
            else:
                result["classification"] = metadata["classification"]
                result["scan_applied"] = metadata["scan_applied"]
            
            results.append(result)
    
    return results


@app.post("/batch-detect")
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = 0.15,
    force_scan: Optional[bool] = False,
    skip_scan: Optional[bool] = False,
    max_workers: Optional[int] = 10
):
    """
    ULTRA-OPTIMIZED batch detection for 1000+ files.
    Uses true batch inference - processes ALL images in single YOLO forward pass.
    
    Performance improvements:
    - Parallel file I/O (async reads)
    - Single batch YOLO inference (vs sequential)
    - Efficient image preprocessing pipeline
    - Minimal memory overhead

    Args:
        files: List of document files (images/PDFs)
        confidence: Confidence threshold (0-1), default 0.15
        force_scan: Force perspective correction on all images
        skip_scan: Skip perspective correction even for camera photos
        max_workers: Maximum parallel I/O workers (default 10, max 50)

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

    if len(files) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Too many files. Maximum 2000 files per batch."
        )

    max_workers = min(max_workers, 50)

    # Update detector confidence
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    try:
        start_time = time.time()
        
        # Phase 1: Parallel file reading (I/O bound - async)
        print(f"[Batch] Phase 1: Reading {len(files)} files...")
        read_start = time.time()
        
        read_tasks = [read_and_prepare_file(file, idx) for idx, file in enumerate(files)]
        read_results = await asyncio.gather(*read_tasks)
        
        # Separate successful reads from errors
        file_data_list = []
        error_results = []
        
        for file_data, error in read_results:
            if error:
                error_results.append(error)
            else:
                file_data_list.append(file_data)
        
        read_time = (time.time() - read_start) * 1000
        print(f"[Batch] Phase 1 complete: {len(file_data_list)} files read in {read_time:.0f}ms")
        
        if not file_data_list:
            return {
                "total_files": len(files),
                "successful_detections": 0,
                "failed_detections": len(error_results),
                "results": error_results,
                "summary": {"total_detections": 0, "total_stamps": 0, "total_signatures": 0, "total_qrs": 0},
                "meta": {"total_processing_time_ms": 0, "error": "All files failed to read"}
            }
        
        # Phase 2: CPU-intensive preprocessing (runs in thread pool)
        print(f"[Batch] Phase 2: Preprocessing {len(file_data_list)} files...")
        prep_start = time.time()
        
        all_images, all_metadata = await asyncio.to_thread(
            process_files_batch_cpu, file_data_list, force_scan, skip_scan
        )
        
        prep_time = (time.time() - prep_start) * 1000
        print(f"[Batch] Phase 2 complete: {len(all_images)} images prepared in {prep_time:.0f}ms")
        
        # Phase 3: Single batch YOLO inference (GPU/CPU)
        print(f"[Batch] Phase 3: Running batch inference on {len(all_images)} images...")
        detect_start = time.time()
        
        results = await asyncio.to_thread(run_batch_detection, all_images, all_metadata, iou_threshold=0.45)
        results.extend(error_results)
        
        detect_time = (time.time() - detect_start) * 1000
        print(f"[Batch] Phase 3 complete: Detection finished in {detect_time:.0f}ms")
        
        total_time = (time.time() - start_time) * 1000

        # Restore original confidence
        detector.confidence_threshold = original_confidence

        # Phase 4: Group PDF pages and aggregate stats
        print(f"[Batch] Phase 4: Aggregating results...")
        
        # Group PDF pages by file
        pdf_groups = {}
        final_results = []
        
        for r in results:
            if r.get("document_type") == "pdf_page":
                file_idx = r["file_index"]
                if file_idx not in pdf_groups:
                    pdf_groups[file_idx] = []
                pdf_groups[file_idx].append(r)
            else:
                final_results.append(r)
        
        # Merge PDF pages into single results
        for file_idx, pages in pdf_groups.items():
            if not pages:
                continue
            
            first_page = pages[0]
            total_detections = sum(p["summary"]["total_detections"] for p in pages)
            total_stamps = sum(p["summary"]["total_stamps"] for p in pages)
            total_signatures = sum(p["summary"]["total_signatures"] for p in pages)
            total_qrs = sum(p["summary"]["total_qrs"] for p in pages)
            
            final_results.append({
                "file_index": file_idx,
                "filename": first_page["filename"],
                "success": True,
                "document_type": "multi_page_pdf",
                "total_pages": len(pages),
                "pages": pages,
                "summary": {
                    "total_detections": total_detections,
                    "total_stamps": total_stamps,
                    "total_signatures": total_signatures,
                    "total_qrs": total_qrs
                }
            })
        
        # Sort by file_index
        final_results.sort(key=lambda x: x.get("file_index", 0))
        
        # Calculate overall summary
        successful_results = [r for r in final_results if r.get("success", False)]
        
        total_detections = sum(r["summary"]["total_detections"] for r in successful_results if "summary" in r)
        total_stamps = sum(r["summary"]["total_stamps"] for r in successful_results if "summary" in r)
        total_signatures = sum(r["summary"]["total_signatures"] for r in successful_results if "summary" in r)
        total_qrs = sum(r["summary"]["total_qrs"] for r in successful_results if "summary" in r)
        
        print(f"[Batch] Complete: {len(files)} files → {len(all_images)} images → {total_detections} detections in {total_time:.0f}ms")
        print(f"[Batch] Performance: {1000*len(files)/total_time:.1f} files/sec, {1000*len(all_images)/total_time:.1f} images/sec")

        return {
            "total_files": len(files),
            "total_images_processed": len(all_images),
            "successful_detections": len(successful_results),
            "failed_detections": len(files) - len(successful_results),
            "results": final_results,
            "summary": {
                "total_detections": total_detections,
                "total_stamps": total_stamps,
                "total_signatures": total_signatures,
                "total_qrs": total_qrs,
            },
            "meta": {
                "total_processing_time_ms": round(total_time, 2),
                "read_time_ms": round(read_time, 2),
                "preprocessing_time_ms": round(prep_time, 2),
                "detection_time_ms": round(detect_time, 2),
                "avg_time_per_file_ms": round(total_time / len(files), 2),
                "avg_time_per_image_ms": round(total_time / max(len(all_images), 1), 2),
                "throughput_files_per_sec": round(1000 * len(files) / total_time, 2),
                "throughput_images_per_sec": round(1000 * len(all_images) / total_time, 2),
                "confidence_threshold": confidence,
                "batch_optimization": "true_batch_inference"
            }
        }

    except Exception as e:
        # Restore confidence on error
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )


@app.post("/batch-process-document")
async def batch_process_document(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = 0.15,
    force_scan: Optional[bool] = False,
    skip_scan: Optional[bool] = False,
    max_workers: Optional[int] = 10
):
    """
    Intelligent batch document processing that combines the best of /process-document
    (classification + DocScanner + per-page metadata) with the throughput of /batch-detect.
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

    if len(files) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Too many files. Maximum 2000 files per batch."
        )

    max_workers = max(1, min(max_workers or 10, 32))

    start_time = time.time()
    error_results = []

    # Phase 1: Async file reads
    read_start = time.time()
    read_tasks = [read_and_prepare_file(file, idx) for idx, file in enumerate(files)]
    read_results = await asyncio.gather(*read_tasks)

    file_data_list = []
    for file_data, error in read_results:
        if error:
            error_results.append(error)
        else:
            file_data_list.append(file_data)

    read_time = (time.time() - read_start) * 1000

    if not file_data_list:
        total_time = (time.time() - start_time) * 1000
        return {
            "total_files": len(files),
            "successful_documents": 0,
            "failed_documents": len(error_results),
            "results": error_results,
            "summary": {
                "total_detections": 0,
                "total_stamps": 0,
                "total_signatures": 0,
                "total_qrs": 0
            },
            "meta": {
                "total_processing_time_ms": round(total_time, 2),
                "read_time_ms": round(read_time, 2),
                "preprocessing_time_ms": 0,
                "detection_time_ms": 0,
                "confidence_threshold": confidence,
                "documents_ready_for_detection": 0,
                "pages_processed": 0,
                "max_workers": max_workers
            }
        }

    # Phase 2: Document classification + scanning in thread pool
    prep_start = time.time()
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        prep_futures = [
            loop.run_in_executor(
                executor,
                prepare_document_for_intelligent_processing,
                file_data,
                force_scan,
                skip_scan
            )
            for file_data in file_data_list
        ]
        prep_results = await asyncio.gather(*prep_futures)
    finally:
        executor.shutdown(wait=True)

    prep_time = (time.time() - prep_start) * 1000

    prepared_docs = []
    for result in prep_results:
        if result.get("success"):
            prepared_docs.append(result)
        else:
            error_results.append(result)

    if not prepared_docs:
        total_time = (time.time() - start_time) * 1000
        return {
            "total_files": len(files),
            "successful_documents": 0,
            "failed_documents": len(error_results),
            "results": error_results,
            "summary": {
                "total_detections": 0,
                "total_stamps": 0,
                "total_signatures": 0,
                "total_qrs": 0
            },
            "meta": {
                "total_processing_time_ms": round(total_time, 2),
                "read_time_ms": round(read_time, 2),
                "preprocessing_time_ms": round(prep_time, 2),
                "detection_time_ms": 0,
                "confidence_threshold": confidence,
                "documents_ready_for_detection": 0,
                "pages_processed": 0,
                "max_workers": max_workers
            }
        }

    # Phase 3: Batch detection
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    detection_time = 0
    try:
        batched_images = []
        image_refs = []

        for doc_idx, doc in enumerate(prepared_docs):
            for page_idx, page in enumerate(doc["pages"]):
                batched_images.append(page["image"])
                image_refs.append((doc_idx, page_idx))

        if batched_images:
            detect_start = time.time()
            detection_results = await asyncio.to_thread(
                detector.detect_batch_optimized,
                batched_images,
                0.45
            )

            for detection, (doc_idx, page_idx) in zip(detection_results, image_refs):
                doc = prepared_docs[doc_idx]
                page = doc["pages"][page_idx]
                metadata = detection.setdefault("meta", {})
                metadata["page_number"] = page["page_number"]
                metadata["file_index"] = doc["file_index"]
                metadata["filename"] = doc["filename"]

                if page.get("base64_image"):
                    detection["page_image"] = page["base64_image"]

                page["detection"] = detection
                page.pop("image", None)

            detection_time = (time.time() - detect_start) * 1000
        else:
            detection_time = 0
    except Exception as e:
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing detection failed: {str(e)}"
        )
    finally:
        detector.confidence_threshold = original_confidence

    # Phase 4: Aggregate per-document results
    overall_summary = {
        "total_detections": 0,
        "total_stamps": 0,
        "total_signatures": 0,
        "total_qrs": 0
    }

    total_pages_processed = 0
    final_results = []

    for doc in prepared_docs:
        doc_pages_payload = []
        doc_summary = {
            "total_detections": 0,
            "total_stamps": 0,
            "total_signatures": 0,
            "total_qrs": 0
        }

        for page in doc["pages"]:
            total_pages_processed += 1
            detection = page.get("detection")
            if detection:
                doc_pages_payload.append(detection)
                summary = detection.get("summary", {})
                doc_summary["total_detections"] += summary.get("total_detections", 0)
                doc_summary["total_stamps"] += summary.get("total_stamps", 0)
                doc_summary["total_signatures"] += summary.get("total_signatures", 0)
                doc_summary["total_qrs"] += summary.get("total_qrs", 0)

        overall_summary["total_detections"] += doc_summary["total_detections"]
        overall_summary["total_stamps"] += doc_summary["total_stamps"]
        overall_summary["total_signatures"] += doc_summary["total_signatures"]
        overall_summary["total_qrs"] += doc_summary["total_qrs"]

        doc_payload = {
            "file_index": doc["file_index"],
            "filename": doc["filename"],
            "success": True,
            "document_type": doc["document_type"],
            "processing": doc["processing"],
            "summary": doc_summary
        }

        if doc.get("transformed_image"):
            doc_payload["transformed_image"] = doc["transformed_image"]

        if doc["document_type"] == "image" and len(doc_pages_payload) == 1:
            doc_payload["result"] = doc_pages_payload[0]
        else:
            doc_payload["pages"] = doc_pages_payload

        final_results.append(doc_payload)

    final_results.extend(error_results)
    final_results.sort(key=lambda x: x.get("file_index", 0))

    successful_documents = sum(1 for r in final_results if r.get("success"))
    failed_documents = len(final_results) - successful_documents
    total_time = (time.time() - start_time) * 1000

    return {
        "total_files": len(files),
        "successful_documents": successful_documents,
        "failed_documents": failed_documents,
        "results": final_results,
        "summary": overall_summary,
        "meta": {
            "total_processing_time_ms": round(total_time, 2),
            "read_time_ms": round(read_time, 2),
            "preprocessing_time_ms": round(prep_time, 2),
            "detection_time_ms": round(detection_time, 2),
            "confidence_threshold": confidence,
            "documents_ready_for_detection": len(prepared_docs),
            "pages_processed": total_pages_processed,
            "max_workers": max_workers
        }
    }


@app.post("/batch-detect-hq")
async def batch_detect_high_quality(
    files: List[UploadFile] = File(...),
    confidence: Optional[float] = 0.15,
    force_scan: Optional[bool] = False,
    skip_scan: Optional[bool] = False,
    max_workers: Optional[int] = 10
):
    """
    HIGH-QUALITY batch detection for accuracy-critical processing.
    Uses full classification (EXIF + visual analysis) and higher resolution.
    
    Differences from fast batch-detect:
    - Full visual classification (not EXIF-only) - more accurate but slower
    - Higher resolution inference (1024px vs 640px) - better small object detection
    - Complete metadata and detailed classification info
    
    Best for:
    - Legal documents requiring maximum accuracy
    - Documents with small or faint signatures/stamps
    - Quality over speed requirements
    
    Args:
        files: List of document files (images/PDFs)
        confidence: Confidence threshold (0-1), default 0.15
        force_scan: Force perspective correction on all images
        skip_scan: Skip perspective correction even for camera photos
        max_workers: Maximum parallel I/O workers (default 10, max 50)

    Returns:
        JSON with batch results and aggregate statistics
    """
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 1000:  # Lower limit for HQ mode
        raise HTTPException(
            status_code=400,
            detail="Too many files for high-quality mode. Maximum 1000 files per batch."
        )

    max_workers = min(max_workers, 50)
    original_confidence = detector.confidence_threshold
    if confidence != original_confidence:
        detector.confidence_threshold = confidence

    try:
        start_time = time.time()
        
        # Phase 1: Parallel file reading
        print(f"[Batch-HQ] Phase 1: Reading {len(files)} files...")
        read_start = time.time()
        
        read_tasks = [read_and_prepare_file(file, idx) for idx, file in enumerate(files)]
        read_results = await asyncio.gather(*read_tasks)
        
        file_data_list = []
        error_results = []
        
        for file_data, error in read_results:
            if error:
                error_results.append(error)
            else:
                file_data_list.append(file_data)
        
        read_time = (time.time() - read_start) * 1000
        print(f"[Batch-HQ] Phase 1 complete: {len(file_data_list)} files read in {read_time:.0f}ms")
        
        if not file_data_list:
            return {
                "total_files": len(files),
                "successful_detections": 0,
                "failed_detections": len(error_results),
                "results": error_results,
                "summary": {"total_detections": 0, "total_stamps": 0, "total_signatures": 0, "total_qrs": 0},
                "meta": {"total_processing_time_ms": 0, "error": "All files failed to read", "quality_mode": "high"}
            }
        
        # Phase 2: High-quality preprocessing (full classification, no fast_mode)
        print(f"[Batch-HQ] Phase 2: High-quality preprocessing {len(file_data_list)} files...")
        prep_start = time.time()
        
        # Modified preprocessing function for HQ mode
        def process_single_file_hq(file_data):
            idx = file_data["idx"]
            filename = file_data["filename"]
            contents = file_data["contents"]
            format_info = file_data["format_info"]
            
            images = []
            metadata_list = []
            
            try:
                if format_info['is_pdf']:
                    pages_data = processor.pdf_to_images(contents, enable_base64=False, parallel=True)
                    for img, page_num, _ in pages_data:
                        images.append(img)
                        metadata_list.append({
                            "file_index": idx,
                            "filename": filename,
                            "document_type": "pdf",
                            "page_number": page_num,
                            "total_pages": len(pages_data)
                        })
                else:
                    # Full classification (fast_mode=False for accuracy)
                    is_camera_photo, classification_info = is_document_photo(contents, fast_mode=False)
                    should_scan = force_scan or (not skip_scan and is_camera_photo)
                    
                    processed_contents = contents
                    scan_applied = False
                    
                    if should_scan and doc_scanner is not None:
                        try:
                            scan_result = doc_scanner.scan_image_bytes(contents)
                            if scan_result['success']:
                                processed_contents = scan_result['transformed_image']
                                scan_applied = True
                        except Exception:
                            pass
                    
                    nparr = np.frombuffer(processed_contents, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        images.append(img)
                        metadata_list.append({
                            "file_index": idx,
                            "filename": filename,
                            "document_type": "image",
                            "classification": classification_info['classification'],
                            "classification_confidence": classification_info['confidence'],
                            "classification_details": classification_info,
                            "scan_applied": scan_applied
                        })
                    else:
                        metadata_list.append({
                            "file_index": idx,
                            "filename": filename,
                            "success": False,
                            "error": "Failed to decode image",
                            "skip_detection": True
                        })
            except Exception as e:
                metadata_list.append({
                    "file_index": idx,
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "skip_detection": True
                })
            
            return images, metadata_list
        
        # Parallel preprocessing with ThreadPoolExecutor
        import multiprocessing
        from functools import partial
        max_prep_workers = min(multiprocessing.cpu_count(), len(file_data_list), 16)
        
        all_images = []
        all_metadata = []
        
        with ThreadPoolExecutor(max_workers=max_prep_workers) as executor:
            results = list(executor.map(process_single_file_hq, file_data_list))
        
        for images, metadata_list in results:
            all_images.extend(images)
            all_metadata.extend(metadata_list)
        
        prep_time = (time.time() - prep_start) * 1000
        print(f"[Batch-HQ] Phase 2 complete: {len(all_images)} images prepared in {prep_time:.0f}ms")
        
        # Phase 3: High-resolution batch detection (1024px)
        print(f"[Batch-HQ] Phase 3: Running high-res inference on {len(all_images)} images...")
        detect_start = time.time()
        
        detection_results = await asyncio.to_thread(
            detector.detect_batch_optimized, 
            all_images, 
            iou_threshold=0.45,  # Match inference.py NMS settings
            image_size=1024      # Match training resolution
        )
        
        # Merge results with base64 encoding
        results = []
        for img, detection_result, metadata in zip(all_images, detection_results, all_metadata):
            if metadata.get("skip_detection"):
                results.append(metadata)
            else:
                result = {
                    "file_index": metadata["file_index"],
                    "filename": metadata["filename"],
                    "success": True,
                    **detection_result
                }
                
                if metadata["document_type"] == "pdf":
                    result["page_number"] = metadata["page_number"]
                    result["document_type"] = "pdf_page"
                    success, encoded_img = cv2.imencode('.png', img)
                    if success:
                        img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                        result["page_image"] = f"data:image/png;base64,{img_base64}"
                else:
                    result["classification"] = metadata["classification"]
                    result["classification_confidence"] = metadata.get("classification_confidence")
                    result["classification_details"] = metadata.get("classification_details")
                    result["scan_applied"] = metadata["scan_applied"]
                
                results.append(result)
        
        results.extend(error_results)
        detect_time = (time.time() - detect_start) * 1000
        print(f"[Batch-HQ] Phase 3 complete: Detection finished in {detect_time:.0f}ms")
        
        # Phase 4: Aggregate results
        pdf_groups = {}
        final_results = []
        
        for r in results:
            if r.get("document_type") == "pdf_page":
                file_idx = r["file_index"]
                if file_idx not in pdf_groups:
                    pdf_groups[file_idx] = []
                pdf_groups[file_idx].append(r)
            else:
                final_results.append(r)
        
        for file_idx, pages in pdf_groups.items():
            if not pages:
                continue
            first_page = pages[0]
            final_results.append({
                "file_index": file_idx,
                "filename": first_page["filename"],
                "success": True,
                "document_type": "multi_page_pdf",
                "total_pages": len(pages),
                "pages": pages,
                "summary": {
                    "total_detections": sum(p["summary"]["total_detections"] for p in pages),
                    "total_stamps": sum(p["summary"]["total_stamps"] for p in pages),
                    "total_signatures": sum(p["summary"]["total_signatures"] for p in pages),
                    "total_qrs": sum(p["summary"]["total_qrs"] for p in pages)
                }
            })
        
        final_results.sort(key=lambda x: x.get("file_index", 0))
        successful_results = [r for r in final_results if r.get("success", False)]
        
        total_detections = sum(r["summary"]["total_detections"] for r in successful_results if "summary" in r)
        total_stamps = sum(r["summary"]["total_stamps"] for r in successful_results if "summary" in r)
        total_signatures = sum(r["summary"]["total_signatures"] for r in successful_results if "summary" in r)
        total_qrs = sum(r["summary"]["total_qrs"] for r in successful_results if "summary" in r)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"[Batch-HQ] Complete: {len(files)} files → {len(all_images)} images → {total_detections} detections in {total_time:.0f}ms")
        print(f"[Batch-HQ] Performance: {1000*len(files)/total_time:.1f} files/sec, {1000*len(all_images)/total_time:.1f} images/sec")

        detector.confidence_threshold = original_confidence

        return {
            "total_files": len(files),
            "total_images_processed": len(all_images),
            "successful_detections": len(successful_results),
            "failed_detections": len(files) - len(successful_results),
            "results": final_results,
            "summary": {
                "total_detections": total_detections,
                "total_stamps": total_stamps,
                "total_signatures": total_signatures,
                "total_qrs": total_qrs,
            },
            "meta": {
                "total_processing_time_ms": round(total_time, 2),
                "read_time_ms": round(read_time, 2),
                "preprocessing_time_ms": round(prep_time, 2),
                "detection_time_ms": round(detect_time, 2),
                "avg_time_per_file_ms": round(total_time / len(files), 2),
                "avg_time_per_image_ms": round(total_time / max(len(all_images), 1), 2),
                "throughput_files_per_sec": round(1000 * len(files) / total_time, 2),
                "throughput_images_per_sec": round(1000 * len(all_images) / total_time, 2),
                "confidence_threshold": confidence,
                "quality_mode": "high",
                "inference_resolution": "1024px",
                "classification_mode": "full_analysis"
            }
        }

    except Exception as e:
        detector.confidence_threshold = original_confidence
        raise HTTPException(
            status_code=500,
            detail=f"High-quality batch detection failed: {str(e)}"
        )


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream detection.

    **HIGH QUALITY MODE**: Uses the same detector as batch endpoints with:
    - Image size: 1024px (same as /batch-detect-hq)
    - IoU threshold: 0.45 (same as /detect)
    - Smart signature grouping enabled
    - Full detection pipeline for maximum accuracy

    Performance: ~2-5 FPS on CPU (prioritizes accuracy over speed)

    Protocol:
    - Client sends: Base64-encoded JPEG frames
    - Server responds: JSON with detections, counts, and performance metrics

    Message Format:
    Client -> Server: {"frame": "base64_encoded_jpeg_string"}
    Server -> Client: {
        "stamps": [...],
        "signatures": [...],
        "qrs": [...],
        "summary": {"total_stamps": 0, "total_signatures": 0, "total_qrs": 0},
        "image_size": {"width_px": 640, "height_px": 480},
        "classification": {...},
        "meta": {
            "inference_time_ms": 123.45,
            "quality_mode": "high_quality_1024px",
            "detector_type": "main_batch_detector"
        }
    }
    """
    await websocket.accept()
    print("[WebSocket HQ] Client connected - HIGH QUALITY MODE (1024px)")

    if detector is None:
        await websocket.send_json({
            "error": "Detector not initialized",
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
                # Run document classification
                classification_start = time.time()
                is_camera, classification_info = is_document_photo(img_bytes)
                classification_time = (time.time() - classification_start) * 1000

                # Run detection using MAIN DETECTOR (high quality, same as batch endpoints)
                # Use detector.detect() instead of realtime_detector for better accuracy
                detection_start = time.time()
                detection_result = await asyncio.to_thread(
                    detector.detect,
                    frame,
                    iou_threshold=0.45,  # Match batch detector settings
                    image_size=1024      # HIGH QUALITY - match batch-detect-hq
                )
                detection_time = (time.time() - detection_start) * 1000

                # detector.detect() already returns categorized format
                # Extract stamps, signatures, qrs directly
                stamps = detection_result.get("stamps", [])
                signatures = detection_result.get("signatures", [])
                qrs = detection_result.get("qrs", [])

                # Build response matching /detect format
                # Extract image size and convert to frontend-compatible format
                img_size = detection_result["image_size"]
                result = {
                    "image_size": {
                        "width": img_size["width_px"],
                        "height": img_size["height_px"]
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
                    "classification": {
                        "is_camera_photo": is_camera,
                        "confidence": classification_info.get("confidence", 0),
                        "reasons": classification_info.get("reasons", []),
                        "exif_data": classification_info.get("exif_data", {}),
                        "visual_features": classification_info.get("visual_features", {}),
                        "classification_time_ms": round(classification_time, 2)
                    }
                }

                # Track performance metrics
                frame_count += 1
                total_inference_time += detection_time
                avg_inference_time = total_inference_time / frame_count

                # Add performance metrics to response
                result["meta"] = {
                    "frame_count": frame_count,
                    "avg_inference_time_ms": round(avg_inference_time, 2),
                    "inference_time_ms": round(detection_time, 2),
                    "total_processing_time_ms": round(detection_time + classification_time, 2),
                    "quality_mode": "high_quality_1024px",  # Indicate using high quality detector
                    "detector_type": "main_batch_detector"  # Using same detector as batch endpoints
                }

                # Send results back to client
                await websocket.send_json(result)

                # Log every 10 frames for monitoring
                if frame_count % 10 == 0:
                    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
                    total_dets = result['summary']['total_detections']
                    print(f"[WebSocket HQ] Frame {frame_count}: "
                          f"{total_dets} detections "
                          f"(stamps: {len(stamps)}, sigs: {len(signatures)}, qrs: {len(qrs)}), "
                          f"{detection_time:.1f}ms, "
                          f"avg FPS: {fps:.1f}, "
                          f"camera: {is_camera}, "
                          f"mode: HIGH_QUALITY_1024px")

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
