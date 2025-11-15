# Quick Start - Document Classifier

## Installation

All required dependencies are already in `requirements.txt`:

```bash
cd backend
pip install -r requirements.txt
```

## Test the Classifier

### 1. Test with Python Script

```bash
# Test a single image
python test_classifier.py path/to/photo.jpg

# Test all sample images
python test_classifier.py --test-all

# Show API usage examples
python test_classifier.py --api-examples
```

### 2. Start the API Server

```bash
# From backend directory
cd app
python main.py

# Or with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server will start at: http://localhost:8000

## Quick API Tests

### Test 1: Classify a Document (No Processing)

```bash
# iPhone/Android camera photo
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@camera_photo.jpg' | jq

# Expected output:
# {
#   "classification": "camera_photo",
#   "confidence": 0.85,
#   "recommendation": "apply_perspective_correction",
#   ...
# }
```

```bash
# PDF or scanned document
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@document.pdf' | jq

# Expected output:
# {
#   "classification": "digital_document",
#   "confidence": 0.92,
#   "recommendation": "use_as_is",
#   ...
# }
```

### Test 2: Intelligent Processing (Auto-Classification + Detection)

```bash
# Upload any document - system decides automatically
curl -X POST http://localhost:8000/process-document \
  -F 'file=@document.jpg' \
  -F 'confidence=0.25' | jq

# Response includes:
# - Classification result
# - Whether perspective correction was applied
# - Object detection results (stamps, signatures, QR codes)
# - All processing metadata
```

### Test 3: Force Perspective Correction

```bash
# Force DocScanner even if classified as digital
curl -X POST http://localhost:8000/process-document \
  -F 'file=@scan.png' \
  -F 'force_scan=true' | jq
```

### Test 4: Skip Perspective Correction

```bash
# Skip DocScanner even if classified as camera photo
curl -X POST http://localhost:8000/process-document \
  -F 'file=@photo.jpg' \
  -F 'skip_scan=true' | jq
```

## Python Usage Example

```python
import requests

# Test classification
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify-document',
        files={'file': f}
    )

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Recommendation: {result['recommendation']}")

if result['is_camera_photo']:
    print("→ This is a camera photo")
    print("→ Perspective correction will be applied")
else:
    print("→ This is a digital document")
    print("→ Will be used as-is")
```

## Integration Example

```python
# Direct Python usage (without API)
from app.document_classifier import is_document_photo

# Read image
with open('document.jpg', 'rb') as f:
    image_bytes = f.read()

# Classify
is_camera_photo, classification_info = is_document_photo(image_bytes)

print(f"Classification: {classification_info['classification']}")
print(f"Confidence: {classification_info['confidence']:.1%}")

# EXIF indicators
print("\nEXIF Analysis:")
for key, value in classification_info['indicators']['exif'].items():
    print(f"  {key}: {value}")

# Visual indicators
print("\nVisual Analysis:")
for key, value in classification_info['indicators']['visual'].items():
    print(f"  {key}: {value}")

# Make decision
if is_camera_photo:
    print("\n→ Apply perspective correction (DocScanner)")
else:
    print("\n→ Use document as-is (no correction needed)")
```

## Understanding the Results

### Classification Response Fields

```json
{
  "classification": "camera_photo" | "digital_document",
  "confidence": 0.85,  // 0.0 to 1.0
  "recommendation": "apply_perspective_correction" | "use_as_is",

  "scores": {
    "exif": 0.80,      // EXIF analysis score
    "visual": 0.65,    // Visual heuristics score
    "final": 0.85      // Combined weighted score
  },

  "indicators": {
    "exif": {
      "has_exif": true,
      "make": "Apple",
      "model": "iPhone 14 Pro",
      "is_phone_camera": true,
      "is_phone_model": true,
      "has_iso": true,
      "has_gps": true
    },
    "visual": {
      "document_contour": true,
      "contour_area_ratio": 0.78,
      "perspective_distortion": true,
      "visible_background": true,
      "has_focus_variation": true
    }
  }
}
```

### Decision Logic

```
final_score > 0.5  →  Camera Photo  →  Apply DocScanner
final_score ≤ 0.5  →  Digital Doc   →  Use As-Is
```

**Confidence Levels:**
- `0.8 - 1.0`: Very confident
- `0.6 - 0.8`: Confident
- `0.4 - 0.6`: Uncertain (near threshold)
- `0.2 - 0.4`: Probably digital
- `0.0 - 0.2`: Very likely digital

## Workflow Diagram

```
User Upload
    │
    ▼
┌─────────────────────────────────────┐
│  1. Classify Document               │
│     - Extract EXIF metadata         │
│     - Analyze visual features       │
│     - Calculate confidence score    │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐   ┌─────────────┐
│Camera Photo │   │Digital Doc  │
│confidence   │   │confidence   │
│   > 0.5     │   │   ≤ 0.5     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 │
┌─────────────┐          │
│2. Apply     │          │
│  DocScanner │          │
│  - Detect   │          │
│    corners  │          │
│  - Perspective         │
│    correction│         │
└──────┬──────┘          │
       │                 │
       └────────┬────────┘
                │
                ▼
     ┌─────────────────────┐
     │ 3. Object Detection │
     │    - Stamps         │
     │    - Signatures     │
     │    - QR Codes       │
     └─────────────────────┘
                │
                ▼
          Final Result
```

## Common Scenarios

### Scenario 1: iPhone Photo of Document
```bash
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@iphone_photo.jpg'
```

**Expected Result:**
- Classification: `camera_photo`
- Confidence: ~0.85 (high)
- EXIF: Apple, iPhone model detected
- Visual: Perspective distortion, background visible
- Recommendation: Apply perspective correction ✓

### Scenario 2: Scanned PDF
```bash
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@scanned.pdf'
```

**Expected Result:**
- Classification: `digital_document`
- Confidence: ~0.95 (very high)
- Reason: PDF format
- Recommendation: Use as-is ✓

### Scenario 3: Screenshot
```bash
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@screenshot.png'
```

**Expected Result:**
- Classification: `digital_document`
- Confidence: ~0.90 (high)
- EXIF: No camera metadata or software tag detected
- Visual: No contours, edges touch boundaries
- Recommendation: Use as-is ✓

### Scenario 4: WhatsApp Photo (EXIF Stripped)
```bash
curl -X POST http://localhost:8000/classify-document \
  -F 'file=@whatsapp_image.jpg'
```

**Expected Result:**
- Classification: `camera_photo` (if visual indicators strong)
- Confidence: ~0.60 (moderate - relies on visual)
- EXIF: Stripped/missing
- Visual: Perspective distortion, background, blur variance
- Recommendation: Apply perspective correction ✓

**If misclassified:**
```bash
# Force correction
curl -X POST http://localhost:8000/process-document \
  -F 'file=@whatsapp_image.jpg' \
  -F 'force_scan=true'
```

## Troubleshooting

### Problem: Camera photos classified as digital

**Solution 1:** Lower confidence threshold
```python
from app.document_classifier import DocumentClassifier

classifier = DocumentClassifier(confidence_threshold=0.4)
```

**Solution 2:** Use force_scan parameter
```bash
curl -X POST http://localhost:8000/process-document \
  -F 'file=@photo.jpg' \
  -F 'force_scan=true'
```

### Problem: Digital documents classified as camera photos

**Solution 1:** Raise confidence threshold
```python
classifier = DocumentClassifier(confidence_threshold=0.6)
```

**Solution 2:** Use skip_scan parameter
```bash
curl -X POST http://localhost:8000/process-document \
  -F 'file=@document.png' \
  -F 'skip_scan=true'
```

### Problem: Classification too slow

**Solution:** Image is too large, resize before uploading
```python
from PIL import Image

img = Image.open('large_image.jpg')
img.thumbnail((1024, 1024))
img.save('resized.jpg', quality=85)
```

## Next Steps

1. **Test with your images**: Try uploading various types of documents
2. **Check logs**: Watch server console for classification details
3. **Review results**: Examine confidence scores and indicators
4. **Adjust threshold**: Tune `confidence_threshold` if needed
5. **Integrate**: Use `/process-document` in your application

For detailed documentation, see [CLASSIFIER_README.md](CLASSIFIER_README.md)
