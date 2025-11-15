# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
# Also install build tools for compiling ocrd-fork-pylsd
RUN apt-get update && apt-get install -y \
    # Build tools
    gcc \
    g++ \
    make \
    python3-dev \
    # Runtime libraries for OpenCV
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory structure
RUN mkdir -p /app/model/runs/train/yolov11s_lora_20251115_144408/weights

# Copy model weights into the image
COPY model/runs/train/yolov11s_lora_20251115_144408/weights/best.pt \
     /app/model/runs/train/yolov11s_lora_20251115_144408/weights/best.pt

# Copy application code
COPY backend/app/ /app/app/
COPY backend/document_scan/ /app/document_scan/

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application with hot-reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
