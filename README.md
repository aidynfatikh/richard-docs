# 🎯 Richard Document Processor

AI-powered document annotation detector for legal documents. Detects **signatures, stamps, and QR codes** with 97.46% accuracy.

---

## 📊 Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **mAP@50** | **97.46%** | Detection accuracy at IoU 0.5 |
| **mAP@50-95** | **74.22%** | Strict accuracy (IoU 0.5-0.95) |
| **Precision** | **94.64%** | Low false positives (5.36%) |
| **Recall** | **93.55%** | Catches 93.55% of annotations |
| **Speed (CPU)** | **145ms** | Per document processing |
| **Speed (GPU)** | **30ms** | Per document processing |
| **Batch (1000 docs)** | **28s** | Parallel batch processing |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA (optional, for GPU acceleration)

---

## 🔧 Backend Setup

### 1. Install Requirements
```bash
cd backend
pip install -r requirements.txt
```

**Requirements:**
```
fastapi==0.121.2
PyMuPDF==1.24.0
ultralytics==8.3.228
opencv-python==4.11.0.86
Pillow==12.0.0
numpy==2.3.4
uvicorn[standard]
websockets
```

### 2. Start Backend Server
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server will run at: `http://localhost:8000`

API docs available at: `http://localhost:8000/docs`

---

## 🎨 Frontend Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

**Key Dependencies:**
```json
{
  "react": "^19.1.1",
  "react-router-dom": "^7.9.6",
  "tailwindcss": "^4.1.13",
  "vite": "^7.1.2"
}
```

### 2. Start Development Server
```bash
cd frontend
npm run dev
```

Frontend will run at: `http://localhost:5173`

### 3. Build for Production
```bash
npm run build
npm run preview
```

---

## 🧠 Model Training

### Train YOLOv11s with LoRA

```bash
cd model
python scripts/training/train_yolov11_lora.py \
  --data data/datasets/main/data.yaml \
  --epochs 60 \
  --batch 16 \
  --imgsz 1024 \
  --freeze 10 \
  --close_mosaic 10
```

**Training Parameters:**
- **Model:** YOLOv11s (9.4M params)
- **Image Size:** 1024×1024
- **Epochs:** 60
- **Batch Size:** 16
- **Optimizer:** SGD (momentum=0.937)
- **Learning Rate:** 0.001 → 0.00001 (cosine decay)
- **Frozen Layers:** 10 (transfer learning)
- **Close Mosaic:** Last 10 epochs (51-60)

**Training Results:**
- Epoch 2: 70.31% mAP@50 (fast start with transfer learning)
- Epoch 21: 95.43% mAP@50
- Epoch 50: 98.97% mAP@50 (peak with mosaic)
- **Epoch 60: 97.46% mAP@50, 74.22% mAP@50-95** (final)

Results saved to: `model/runs/train/yolov11s_lora_YYYYMMDD_HHMMSS/`

---

## 🔍 Model Inference

### Single Document Detection

```bash
cd model
python scripts/inference/inference_main_model.py \
  --source data/test_images/document.jpg \
  --weights runs/train/best_model/weights/best.pt \
  --conf 0.25 \
  --imgsz 1024 \
  --save-json
```

**Output:**
```json
{
  "detections": [
    {"class": "signature", "bbox": [100, 200, 300, 400], "confidence": 0.95},
    {"class": "stamp", "bbox": [500, 100, 700, 300], "confidence": 0.92},
    {"class": "qr_code", "bbox": [800, 500, 900, 600], "confidence": 0.98}
  ],
  "counts": {
    "signatures": 1,
    "stamps": 1,
    "qr_codes": 1
  },
  "processing_time_ms": 145
}
```

### Batch Processing (1000 documents)

```bash
python scripts/inference/inference_main_model.py \
  --source data/test_images/ \
  --batch \
  --save-json
```

**Performance:**
- **1000 documents in 28 seconds**
- Parallel PDF rendering
- Automatic multi-page handling
- Smart detection grouping

---

## 🌐 API Endpoints

### REST API

**Single Document Detection:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@document.pdf"
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/batch-detect" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf"
```

**Document Processing (with classification):**
```bash
curl -X POST "http://localhost:8000/process-document" \
  -F "file=@camera_photo.jpg"
```

### WebSocket (Real-time)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detect');
ws.send(imageBlob); // Send camera frame
ws.onmessage = (event) => {
  const detections = JSON.parse(event.data);
  // Process detections (8-10 FPS)
};
```

---

## 📁 Project Structure

```
richard-docs/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # API endpoints
│   │   ├── model_service.py   # YOLO model wrapper
│   │   ├── document_processor.py  # PDF/image handling
│   │   ├── document_classifier.py # Camera vs scan detection
│   │   └── document_scan/
│   │       └── scan.py        # Perspective correction
│   └── requirements.txt
│
├── frontend/                   # React + TypeScript
│   ├── src/
│   │   ├── pages/             # HomePage, SolutionPage, RealtimeScanPage
│   │   ├── components/        # UI components
│   │   ├── services/          # API & WebSocket services
│   │   └── types/             # TypeScript definitions
│   └── package.json
│
├── model/                      # Training & inference
│   ├── scripts/
│   │   ├── training/          # train_yolov11_lora.py
│   │   ├── inference/         # inference_main_model.py
│   │   ├── testing/           # test scripts
│   │   └── data_preparation/  # dataset builders
│   ├── data/
│   │   ├── datasets/          # YOLO format datasets
│   │   └── raw/               # raw images & annotations
│   ├── runs/                  # training outputs
│   └── weights/               # pretrained models
│
└── README.md
```

---

## 🔬 Technical Features

### 1. Document Classification
- **EXIF metadata analysis** (camera model detection)
- **Visual contour detection** (document edges, perspective)
- **85% accuracy** → applies perspective correction only when needed

### 2. Perspective Correction (DocScanner)
- Edge detection (Canny)
- 4-point perspective transform
- CLAHE enhancement + sharpening

### 3. Smart Detection Grouping
- **Pass 1:** Remove boxes 75% contained in others
- **Pass 2:** Merge nearby boxes (distance < 50px, IoU > 0.2)
- **Result:** 3x cleaner outputs

### 4. Multi-Interface Support
- **REST API:** `/detect`, `/batch-detect`, `/process-document`
- **WebSocket:** Real-time video stream detection (8-10 FPS)
- **Frontend:** React + TypeScript with live camera

---

## 📈 Dataset

- **Size:** 57 annotated legal documents
- **Split:** 51 train / 6 validation (90/10)
- **Classes:** 3 (QR codes, signatures, stamps)
- **Augmentation:** Mosaic, rotation (±10°), scale (0.5-1.5x), color jitter

---

## 🛠️ Development

### Run Tests
```bash
cd backend
python -m pytest tests/
```

### Lint & Format (Frontend)
```bash
cd frontend
npm run lint
npm run format
```

### Check Errors
```bash
npm run lint:fix
npm run format:check
```

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using YOLOv11s, FastAPI, and React**