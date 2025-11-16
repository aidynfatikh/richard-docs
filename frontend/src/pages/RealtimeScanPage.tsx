import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { RealtimeDetectionOverlay } from '../components/RealtimeDetectionOverlay';
import { RealtimeDetectionService, captureFrameFromVideo, calculateFrameInterval } from '../services/websocket.service';
import type { RealtimeDetection, DetectionCounts, WebSocketState } from '../types/websocket.types';
import { WEB_SOCKET_STATE } from '../types/websocket.types';
import { API_CONFIG } from '../config/api.config';

const TARGET_FPS = 3; // Process 3 frames per second for CPU optimization
const FRAME_QUALITY = 0.7; // JPEG quality (0-1)

export function RealtimeScanPage() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsServiceRef = useRef<RealtimeDetectionService | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wsState, setWsState] = useState<WebSocketState>(WEB_SOCKET_STATE.DISCONNECTED);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');

  // Detection state
  const [coordinates, setCoordinates] = useState<RealtimeDetection[]>([]);
  const [counts, setCounts] = useState<DetectionCounts>({ stamp: 0, signature: 0, qr: 0 });
  const [imageSize, setImageSize] = useState({ width: 640, height: 480 });
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);

  // Initialize camera and WebSocket
  useEffect(() => {
    let mounted = true;

    const initialize = async () => {
      try {
        // Request camera access
        await startCamera(facingMode);

        // Initialize WebSocket connection
        const apiUrl = import.meta.env.VITE_API_URL || API_CONFIG.BASE_URL;
        const wsService = new RealtimeDetectionService(apiUrl);
        wsServiceRef.current = wsService;

        // Register callbacks
        wsService.onStateChange((state) => {
          if (mounted) {
            setWsState(state);
          }
        });

        wsService.onDetection((result) => {
          if (mounted) {
            // Transform categorized detections back to coordinates format for overlay
            const allDetections: RealtimeDetection[] = [];

            // Add stamps
            result.stamps.forEach(det => {
              allDetections.push({
                coordinates: {
                  x1: det.bbox[0],
                  y1: det.bbox[1],
                  x2: det.bbox[2],
                  y2: det.bbox[3]
                },
                confidence: det.confidence,
                class: 'stamp',
                grouped: det.grouped,
                group_count: det.group_count
              });
            });

            // Add signatures
            result.signatures.forEach(det => {
              allDetections.push({
                coordinates: {
                  x1: det.bbox[0],
                  y1: det.bbox[1],
                  x2: det.bbox[2],
                  y2: det.bbox[3]
                },
                confidence: det.confidence,
                class: 'signature',
                grouped: det.grouped,
                group_count: det.group_count
              });
            });

            // Add QR codes
            result.qrs.forEach(det => {
              allDetections.push({
                coordinates: {
                  x1: det.bbox[0],
                  y1: det.bbox[1],
                  x2: det.bbox[2],
                  y2: det.bbox[3]
                },
                confidence: det.confidence,
                class: 'qr',
                grouped: det.grouped,
                group_count: det.group_count
              });
            });

            setCoordinates(allDetections);
            setCounts({
              stamp: result.summary.total_stamps,
              signature: result.summary.total_signatures,
              qr: result.summary.total_qrs
            });
            setImageSize(result.image_size);

            // Use total processing time if available, otherwise use inference time
            const processingTime = result.meta?.total_processing_time_ms || result.meta?.inference_time_ms || 0;
            setLatency(processingTime);

            // Calculate FPS from average inference time
            if (result.meta) {
              const avgFps = 1000 / result.meta.avg_inference_time_ms;
              setFps(avgFps);
            }

            // Log classification results for debugging
            if (result.classification) {
              console.log('[Classification]', {
                is_camera: result.classification.is_camera_photo,
                confidence: result.classification.confidence,
                reasons: result.classification.reasons
              });
            }
          }
        });

        wsService.onError((err) => {
          if (mounted) {
            console.error('[RealtimeScan] WebSocket error:', err);
            setError(err);
          }
        });

        // Connect to WebSocket
        await wsService.connect();

        // Start frame capture and sending
        startFrameCapture();

        if (mounted) {
          setIsLoading(false);
        }
      } catch (err) {
        if (mounted) {
          console.error('[RealtimeScan] Initialization error:', err);
          setError(err instanceof Error ? err.message : 'Failed to initialize scanner');
          setIsLoading(false);
        }
      }
    };

    initialize();

    // Cleanup on unmount
    return () => {
      mounted = false;
      stopFrameCapture();
      stopCamera();
      if (wsServiceRef.current) {
        wsServiceRef.current.destroy();
        wsServiceRef.current = null;
      }
    };
  }, [facingMode]);

  /**
   * Start camera with specified facing mode
   */
  const startCamera = async (mode: 'user' | 'environment') => {
    try {
      // Stop existing stream if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: mode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          aspectRatio: { ideal: 16 / 9 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error('[RealtimeScan] Camera error:', err);
      throw new Error('Failed to access camera. Please grant camera permissions.');
    }
  };

  /**
   * Stop camera
   */
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  /**
   * Start periodic frame capture and sending
   */
  const startFrameCapture = () => {
    const interval = calculateFrameInterval(TARGET_FPS);

    frameIntervalRef.current = window.setInterval(() => {
      if (!videoRef.current || !wsServiceRef.current?.isConnected()) {
        return;
      }

      // Capture frame from video
      const frameDataUrl = captureFrameFromVideo(videoRef.current, FRAME_QUALITY);
      if (frameDataUrl) {
        // Send frame via WebSocket
        wsServiceRef.current.sendFrame(frameDataUrl);
      }
    }, interval);
  };

  /**
   * Stop frame capture
   */
  const stopFrameCapture = () => {
    if (frameIntervalRef.current !== null) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  /**
   * Switch between front and rear camera
   */
  const switchCamera = async () => {
    const newMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(newMode);
  };

  /**
   * Capture current frame with detections
   */
  const captureFrame = async () => {
    if (!videoRef.current) return;

    try {
      // Stop real-time scanning
      stopFrameCapture();

      // Create canvas with current frame and detections
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Draw video frame
      ctx.drawImage(videoRef.current, 0, 0);

      // Convert to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => {
          if (b) resolve(b);
        }, 'image/jpeg', 0.95);
      });

      console.log('[RealtimeScan] Captured frame bytes:', blob.size);

     // Navigate to solution page (you could also upload via API)
      // For now, we'll just show a success message
      alert(`Captured! Found: ${counts.stamp} stamps, ${counts.signature} signatures, ${counts.qr} QR codes`);

      // Resume scanning
      startFrameCapture();
    } catch (err) {
      console.error('[RealtimeScan] Capture error:', err);
      alert('Failed to capture frame');
    }
  };

  /**
   * Exit scanner and return to home
   */
  const exitScanner = () => {
    navigate('/');
  };

  // Show loading state
  if (isLoading) {
    return (
      <div style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'black',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{ fontSize: '24px', marginBottom: '16px' }}>Initializing Scanner...</div>
          <div style={{ fontSize: '14px', opacity: 0.7 }}>Requesting camera access</div>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'black',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
        zIndex: 9999,
      }}>
        <div style={{ textAlign: 'center', color: 'white', maxWidth: '400px' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
          <div style={{ fontSize: '20px', marginBottom: '12px', fontWeight: 'bold' }}>Scanner Error</div>
          <div style={{ fontSize: '14px', opacity: 0.8, marginBottom: '24px' }}>{error}</div>
          <button
            onClick={exitScanner}
            style={{
              backgroundColor: 'rgb(0, 23, 255)',
              color: 'white',
              padding: '12px 32px',
              borderRadius: '24px',
              border: 'none',
              fontSize: '16px',
              fontWeight: '600',
              cursor: 'pointer',
            }}
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  const totalDetections = counts.stamp + counts.signature + counts.qr;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'black',
      overflow: 'hidden',
      zIndex: 9999,
    }}>
      {/* Video stream */}
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
        />
      </div>

      {/* Detection overlay */}
      <RealtimeDetectionOverlay
        videoElement={videoRef.current}
        coordinates={coordinates}
        counts={counts}
        imageSize={imageSize}
        fps={fps}
        latency={latency}
        showConfidence={true}
      />

      {/* Connection status indicator */}
      {wsState !== 'connected' && (
        <div style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          backgroundColor: wsState === 'connecting' ? 'rgba(255, 165, 0, 0.9)' : 'rgba(239, 68, 68, 0.9)',
          color: 'white',
          padding: '8px 16px',
          borderRadius: '8px',
          fontSize: '12px',
          fontWeight: 'bold',
          zIndex: 30,
        }}>
          {wsState === 'connecting' ? 'Connecting...' : 'Disconnected'}
        </div>
      )}

      {/* Bottom controls */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: '24px',
        background: 'linear-gradient(to top, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 0, 0) 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 30,
      }}>
        {/* Exit button */}
        <button
          onClick={exitScanner}
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(8px)',
            border: '2px solid white',
            color: 'white',
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '24px',
            cursor: 'pointer',
          }}
        >
          ✕
        </button>

        {/* Capture button */}
        <button
          onClick={captureFrame}
          disabled={totalDetections === 0}
          style={{
            backgroundColor: totalDetections > 0 ? 'rgb(0, 23, 255)' : 'rgba(100, 100, 100, 0.5)',
            border: '4px solid white',
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            cursor: totalDetections > 0 ? 'pointer' : 'not-allowed',
            opacity: totalDetections > 0 ? 1 : 0.5,
            transition: 'all 0.2s',
          }}
        />

        {/* Switch camera button */}
        <button
          onClick={switchCamera}
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(8px)',
            border: '2px solid white',
            color: 'white',
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '24px',
            cursor: 'pointer',
          }}
        >
          🔄
        </button>
      </div>
    </div>
  );
}
