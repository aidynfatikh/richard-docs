# Running the Document Detection Application

## Complete Setup Guide

### Prerequisites
- Python 3.8+ with conda/virtualenv
- Node.js 16+
- Trained YOLOv11 model (in `model/runs/train/`)

---

## Backend Setup & Run

### Option 1: Direct Python (Development)

```bash
# Navigate to backend
cd backend

# Activate your Python environment (ML environment)
conda activate ML  # or your environment name

# Install dependencies (if not done)
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at: `http://localhost:8000`

### Option 2: Docker (Production)

```bash
# From project root
docker-compose up backend
```

### Backend Features
- ✅ YOLOv11 object detection for stamps, signatures, and QR codes
- ✅ Signature grouping with IoU-based clustering
- ✅ FastAPI REST API
- ✅ CORS enabled for frontend integration
- ✅ Auto-detection of best trained model

### API Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health with model info
- `POST /detect` - Detect document elements
  - Parameters: `file` (image), `confidence` (optional, default: 0.25)

---

## Frontend Setup & Run

```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Set environment variable (optional, defaults to localhost:8000)
export VITE_API_URL=http://localhost:8000

# Run development server
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### Frontend Features
- ✅ Drag & drop file upload
- ✅ Multiple document processing
- ✅ Real-time progress tracking
- ✅ **Image visualization with bounding boxes**
- ✅ Color-coded detections:
  - 🔴 Stamps (Red)
  - 🟢 Signatures (Green)
  - 🔵 QR Codes (Blue)
- ✅ Grouping indicators for merged signatures
- ✅ Detailed statistics and metadata

---

## Full Stack Quick Start

### Terminal 1 - Backend
```bash
cd /home/fatikh/projects/richard-docs/backend
conda activate ML
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Frontend
```bash
cd /home/fatikh/projects/richard-docs/frontend
npm run dev
```

### Access the Application
1. Open browser: `http://localhost:5173`
2. Upload document images
3. Click "Analyze Documents"
4. View results with visualized bounding boxes!

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  - File upload UI                                           │
│  - Canvas-based bbox visualization                          │
│  - Results dashboard                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST /detect
                     │ (multipart/form-data)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  - Image processing                                         │
│  - YOLOv11 inference                                        │
│  - Signature grouping                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  YOLOv11 Model (PyTorch)                     │
│  - Trained on document dataset                              │
│  - Detects: stamps, signatures, QR codes                    │
│  - LoRA fine-tuned                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Backend Issues

**Model not found:**
```bash
# Check if model exists
ls -la model/runs/train/*/weights/best.pt

# If no model, train first:
cd model
python train_yolov11_lora.py
```

**Port already in use:**
```bash
# Change port
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Update frontend .env
export VITE_API_URL=http://localhost:8001
```

### Frontend Issues

**API connection failed:**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Verify `VITE_API_URL` environment variable

**Canvas not displaying:**
- Check browser console for errors
- Ensure image files are valid formats
- Clear browser cache and reload

---

## Testing

### Test Backend Directly
```bash
cd backend
python test_detection.py
```

### Test API Endpoint
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/document.jpg" \
  -F "confidence=0.25"
```

### Test Frontend Build
```bash
cd frontend
npm run build
npm run preview
```

---

## Configuration

### Backend (`backend/app/main.py`)
```python
detector = get_detector(
    confidence_threshold=0.25,    # Detection confidence
    device='cpu',                 # 'cpu' or 'cuda'
    enable_grouping=True,         # Enable signature grouping
    group_iou_threshold=0.3       # IoU for grouping
)
```

### Frontend (`frontend/src/config/api.config.ts`)
```typescript
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  TIMEOUT: 30000,
  MAX_FILE_SIZE: 50 * 1024 * 1024,  // 50MB
};
```

---

## Features

### ✅ Implemented
- YOLOv11 object detection
- Signature grouping (overlapping detections merged)
- Canvas-based bbox visualization
- Multi-file batch processing
- Progress tracking
- Color-coded detection types
- Confidence scores display
- Grouping metadata
- Responsive design
- Error handling

### 🎯 Key Improvements
1. **Signature Grouping**: Multiple overlapping signature boxes are automatically merged using IoU-based clustering
2. **Visual Feedback**: Bounding boxes drawn directly on images with labels
3. **Performance**: Fast inference with model caching
4. **User Experience**: Drag & drop, progress bars, detailed statistics

---

## Development Notes

- Backend uses singleton pattern for model loading
- Frontend canvas dynamically scales images
- Grouping uses DFS algorithm for connected components
- All coordinates in absolute pixel values
- CORS enabled for local development

Enjoy analyzing documents! 🚀
