# Backend API - Document Detection

FastAPI backend for detecting stamps, signatures, and QR codes in document images using YOLOv11.

## Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Ensure model is trained:**
The API auto-detects the best model from `../model/runs/train/`. Make sure you've trained a model first.

3. **Run the server:**
```bash
cd app
python main.py
```

Or with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Document Detection API",
  "status": "running",
  "model_loaded": true
}
```

### `GET /health`
Detailed health check with model info.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/path/to/best.pt",
  "classes": {
    "0": "qr",
    "1": "signature",
    "2": "stamp"
  }
}
```

### `POST /detect`
Detect document elements in an uploaded image.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `confidence` (optional): Confidence threshold (0-1), default: 0.25

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@document.jpg" \
  -F "confidence=0.3"
```

**Example with Python:**
```python
import requests

with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f},
        data={"confidence": 0.3}
    )
    
print(response.json())
```

**Response:**
```json
{
  "image_size": {
    "width_px": 1684,
    "height_px": 1190
  },
  "stamps": [
    {
      "id": "stamp_1",
      "bbox": [709, 1184, 918, 1402],
      "confidence": 0.94,
      "class_name": "stamp",
      "class_id": 2
    }
  ],
  "signatures": [
    {
      "id": "signature_1",
      "bbox": [702, 1227, 888, 1368],
      "confidence": 0.91,
      "class_name": "signature",
      "class_id": 1
    }
  ],
  "qrs": [],
  "summary": {
    "total_stamps": 1,
    "total_signatures": 1,
    "total_qrs": 0,
    "total_detections": 2
  },
  "meta": {
    "model_version": "best.pt",
    "inference_time_ms": 145.32,
    "confidence_threshold": 0.3,
    "total_processing_time_ms": 156.78
  }
}
```

## Response Format

The API returns detections categorized by type:

### Bounding Box Format
All bounding boxes are in `[x_min, y_min, x_max, y_max]` format (absolute pixel coordinates).

### Detection Object
Each detection contains:
- `id`: Unique identifier (e.g., "stamp_1")
- `bbox`: Bounding box `[x_min, y_min, x_max, y_max]`
- `confidence`: Detection confidence (0-1)
- `class_name`: Type ("stamp", "signature", or "qr")
- `class_id`: Numeric class ID

### Summary
- `total_stamps`: Number of stamps detected
- `total_signatures`: Number of signatures detected
- `total_qrs`: Number of QR codes detected
- `total_detections`: Total detections

### Metadata
- `model_version`: Model filename
- `inference_time_ms`: Model inference time
- `total_processing_time_ms`: Total request processing time
- `confidence_threshold`: Threshold used

## CORS

CORS is enabled for all origins by default. For production, update the `allow_origins` in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    ...
)
```

## Frontend Integration

Example JavaScript fetch:

```javascript
async function detectElements(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('confidence', 0.25);
  
  const response = await fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result;
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Invalid image or bad request
- `500`: Server error during detection
- `503`: Model not loaded

Error response format:
```json
{
  "detail": "Error message"
}
```
