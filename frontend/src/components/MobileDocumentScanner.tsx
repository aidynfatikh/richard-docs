import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api.service';
import { getCameraConstraints } from '../utils/deviceDetection';
import type { ScanDocumentResponse, BatchDetectionResponse } from '../types/api.types';

interface ScannedDocument {
  id: string;
  transformedImage: string;
  originalBlob: Blob;
  timestamp: number;
}

type ScanMode = 'realtime' | 'auto';

export default function MobileDocumentScanner() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<number>();
  const stabilityTimerRef = useRef<number>();

  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string>('');
  const [scanMode, setScanMode] = useState<ScanMode>('auto');
  const [scannedDocs, setScannedDocs] = useState<ScannedDocument[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [documentDetected, setDocumentDetected] = useState(false);
  const [stabilityCounter, setStabilityCounter] = useState(0);
  const [showPreview, setShowPreview] = useState(false);
  const [processingBatch, setProcessingBatch] = useState(false);
  const [cameraStarted, setCameraStarted] = useState(false);

  // Don't auto-start camera - wait for user interaction
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      setError('');
      console.log('Requesting camera access...');

      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not available. Please use HTTPS or a modern browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia(
        getCameraConstraints('environment')
      );

      console.log('Camera stream obtained:', stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        // Wait for video to be ready and play
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          videoRef.current?.play().then(() => {
            console.log('Video playing');
            setCameraStarted(true);
            setIsReady(true);
          }).catch((playErr) => {
            console.error('Error playing video:', playErr);
            setError('Failed to start video playback. Please try again.');
          });
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      if ((err as any).name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if ((err as any).name === 'NotFoundError') {
        setError('No camera found on this device.');
      } else if ((err as any).name === 'OverconstrainedError') {
        setError('Camera constraints not supported. Trying fallback...');
        // Try with simpler constraints
        tryFallbackCamera();
      } else {
        setError(`Unable to access camera: ${(err as any).message || 'Unknown error'}`);
      }
    }
  };

  const tryFallbackCamera = async () => {
    try {
      // Try with basic constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().then(() => {
            setCameraStarted(true);
            setIsReady(true);
            setError('');
          });
        };
      }
    } catch (fallbackErr) {
      console.error('Fallback camera error:', fallbackErr);
      setError('Camera not available. Please check your device.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
    if (stabilityTimerRef.current) {
      clearTimeout(stabilityTimerRef.current);
    }
  };

  // Start document detection when camera is ready
  useEffect(() => {
    if (isReady && scanMode === 'auto') {
      startDocumentDetection();
    } else {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, [isReady, scanMode]);

  const startDocumentDetection = () => {
    // Check for document presence every 200ms
    detectionIntervalRef.current = window.setInterval(() => {
      detectDocumentInFrame();
    }, 200);
  };

  const detectDocumentInFrame = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || isProcessing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Draw current video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Simple edge detection to determine if document is present
    // This is a client-side heuristic - actual scanning happens server-side
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const hasDocument = detectEdges(imageData);

    setDocumentDetected(hasDocument);

    if (hasDocument) {
      // Increment stability counter
      setStabilityCounter((prev) => {
        const newCount = prev + 1;

        // If document has been stable for >0.5s (3 frames at 200ms interval)
        if (newCount >= 3 && !isProcessing) {
          // Auto-capture
          captureDocument();
          return 0; // Reset counter
        }

        return newCount;
      });
    } else {
      setStabilityCounter(0);
    }
  }, [isProcessing]);

  // Simple edge detection heuristic
  const detectEdges = (imageData: ImageData): boolean => {
    const data = imageData.data;
    let edgePixels = 0;
    const threshold = 30;

    // Sample every 10th pixel for performance
    for (let i = 0; i < data.length; i += 40) {
      const r1 = data[i];
      const g1 = data[i + 1];
      const b1 = data[i + 2];

      const r2 = data[i + 4] || r1;
      const g2 = data[i + 5] || g1;
      const b2 = data[i + 6] || b1;

      const diff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);

      if (diff > threshold) {
        edgePixels++;
      }
    }

    // If enough edges detected, likely a document
    return edgePixels > 100;
  };

  const captureDocument = async () => {
    if (!canvasRef.current || !videoRef.current || isProcessing) return;

    setIsProcessing(true);

    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) return;

      // Capture current frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Convert to blob
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((b) => {
          if (b) resolve(b);
          else reject(new Error('Failed to create blob'));
        }, 'image/jpeg', 0.95);
      });

      // Send to scan-document API
      const file = new File([blob], `scan-${Date.now()}.jpg`, { type: 'image/jpeg' });
      const result = await apiService.scanDocument(file);

      if (result.success && result.data) {
        // Add to scanned documents
        const scannedDoc: ScannedDocument = {
          id: Date.now().toString(),
          transformedImage: result.data.transformed_image,
          originalBlob: blob,
          timestamp: Date.now(),
        };

        setScannedDocs((prev) => [...prev, scannedDoc]);

        // Brief flash to indicate capture
        setShowPreview(true);
        setTimeout(() => setShowPreview(false), 500);
      } else {
        console.error('Scan failed:', result.error);
      }
    } catch (err) {
      console.error('Capture error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const manualCapture = () => {
    captureDocument();
  };

  const removeDocument = (id: string) => {
    setScannedDocs((prev) => prev.filter((doc) => doc.id !== id));
  };

  const processBatch = async () => {
    if (scannedDocs.length === 0) return;

    setProcessingBatch(true);

    try {
      // Convert scanned documents to files
      const files = scannedDocs.map((doc) =>
        apiService.dataUrlToFile(doc.transformedImage, `document-${doc.id}.jpg`)
      );

      // Send to batch-detect
      const result = await apiService.batchDetect(files, 0.25);

      if (result.success && result.data) {
        // Navigate to results page with the batch data
        navigate('/solution', { state: { batchResults: result.data } });
      } else {
        setError(result.error?.detail || 'Batch processing failed');
      }
    } catch (err) {
      setError('Failed to process documents');
      console.error(err);
    } finally {
      setProcessingBatch(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 text-white p-4 flex justify-between items-center z-10">
        <h1 className="text-lg font-semibold">Document Scanner</h1>
        {cameraStarted && (
          <div className="flex gap-2">
            <button
              onClick={() => setScanMode(scanMode === 'auto' ? 'realtime' : 'auto')}
              className="px-3 py-1 bg-blue-600 rounded text-sm"
            >
              {scanMode === 'auto' ? 'Auto' : 'Real-time'}
            </button>
          </div>
        )}
      </div>

      {/* Camera not started - show start button */}
      {!cameraStarted && (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-white">
          <svg
            className="w-24 h-24 mb-6 text-blue-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <h2 className="text-2xl font-bold mb-2">Ready to Scan?</h2>
          <p className="text-gray-400 text-center mb-8 max-w-sm">
            Click the button below to start your camera and begin scanning documents
          </p>
          <button
            onClick={startCamera}
            className="px-8 py-4 bg-blue-600 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors"
          >
            Start Camera
          </button>
          {error && (
            <div className="mt-6 bg-red-500/20 border border-red-500 text-red-200 px-4 py-3 rounded max-w-sm text-center text-sm">
              <p className="font-semibold mb-2">⚠️ {error}</p>
              {error.includes('HTTPS') && (
                <p className="text-xs mt-2 text-gray-300">
                  Camera access requires HTTPS. Make sure you're accessing the site via https:// or localhost
                </p>
              )}
              {error.includes('permission') && (
                <p className="text-xs mt-2 text-gray-300">
                  Go to your browser settings and enable camera permissions for this site
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Camera View */}
      {cameraStarted && (
        <div className="flex-1 relative overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />

        <canvas ref={canvasRef} className="hidden" />

        {/* Document detection overlay */}
        {scanMode === 'realtime' && documentDetected && (
          <div className="absolute inset-4 border-4 border-green-500 rounded-lg pointer-events-none" />
        )}

        {showPreview && (
          <div className="absolute inset-0 bg-white opacity-50 pointer-events-none" />
        )}

        {/* Capture button (manual mode) */}
        {scanMode === 'realtime' && (
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
            <button
              onClick={manualCapture}
              disabled={isProcessing}
              className="w-16 h-16 rounded-full bg-white border-4 border-blue-500 active:scale-95 transition-transform disabled:opacity-50"
            />
          </div>
        )}

        {/* Auto-capture indicator */}
        {scanMode === 'auto' && documentDetected && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-4 py-2 rounded-full text-sm">
            Document detected... {stabilityCounter}/3
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-2 rounded text-sm max-w-xs text-center">
            {error}
          </div>
        )}

        {/* Processing indicator */}
        {isProcessing && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-4 py-2 rounded text-sm">
            Scanning...
          </div>
        )}
        </div>
      )}

      {/* Scanned documents preview */}
      {cameraStarted && scannedDocs.length > 0 && (
        <div className="bg-gray-900 p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-white font-semibold">
              Scanned Documents ({scannedDocs.length})
            </h2>
            <button
              onClick={processBatch}
              disabled={processingBatch}
              className="px-4 py-2 bg-green-600 text-white rounded font-semibold disabled:opacity-50"
            >
              {processingBatch ? 'Processing...' : 'Done & Analyze'}
            </button>
          </div>

          <div className="flex gap-2 overflow-x-auto pb-2">
            {scannedDocs.map((doc) => (
              <div key={doc.id} className="relative flex-shrink-0">
                <img
                  src={doc.transformedImage}
                  alt="Scanned document"
                  className="w-24 h-32 object-cover rounded border-2 border-gray-700"
                />
                <button
                  onClick={() => removeDocument(doc.id)}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs font-bold"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
