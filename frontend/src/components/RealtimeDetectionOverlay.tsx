import { useEffect, useRef } from 'react';
import type { RealtimeDetection, DetectionCounts } from '../types/websocket.types';

interface RealtimeDetectionOverlayProps {
  videoElement: HTMLVideoElement | null;
  detections: RealtimeDetection[];
  counts: DetectionCounts;
  imageSize: { width: number; height: number };
  fps?: number;
  latency?: number;
  showConfidence?: boolean;
}

const COLORS = {
  stamp: { box: 'rgb(239, 68, 68)', label: 'rgba(239, 68, 68, 0.9)' },
  signature: { box: 'rgb(34, 197, 94)', label: 'rgba(34, 197, 94, 0.9)' },
  qr: { box: 'rgb(59, 130, 246)', label: 'rgba(59, 130, 246, 0.9)' },
};

export function RealtimeDetectionOverlay({
  videoElement,
  detections,
  counts,
  imageSize,
  fps = 0,
  latency = 0,
  showConfidence = true,
}: RealtimeDetectionOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoElement) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video display size
    const rect = videoElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factors
    const scaleX = canvas.width / imageSize.width;
    const scaleY = canvas.height / imageSize.height;

    // Draw detections
    detections.forEach((detection) => {
      const [x1, y1, x2, y2] = detection.bbox;
      const color = COLORS[detection.class];

      // Scale coordinates to canvas size
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;
      const width = scaledX2 - scaledX1;
      const height = scaledY2 - scaledY1;

      // Draw bounding box
      ctx.strokeStyle = color.box;
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX1, scaledY1, width, height);

      // Draw label background
      const label = `${detection.class}`;
      const confidence = showConfidence ? ` ${(detection.confidence * 100).toFixed(0)}%` : '';
      const labelText = label + confidence;

      ctx.font = 'bold 14px sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width + 8;
      const textHeight = 20;

      ctx.fillStyle = color.label;
      ctx.fillRect(scaledX1, scaledY1 - textHeight, textWidth, textHeight);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(labelText, scaledX1 + 4, scaledY1 - 6);
    });
  }, [videoElement, detections, imageSize, showConfidence]);

  // Calculate total detections
  const totalDetections = counts.stamp + counts.signature + counts.qr;

  return (
    <>
      {/* Canvas overlay for bounding boxes */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 10,
        }}
      />

      {/* Detection stats overlay */}
      <div
        style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          right: '16px',
          zIndex: 20,
          pointerEvents: 'none',
        }}
      >
        {/* Detection counts */}
        <div
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(8px)',
            borderRadius: '12px',
            padding: '12px 16px',
            marginBottom: '12px',
          }}
        >
          <div style={{ color: 'white', fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>
            {totalDetections === 0 ? 'Scanning...' : `${totalDetections} Detection${totalDetections !== 1 ? 's' : ''}`}
          </div>

          {totalDetections > 0 && (
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              {counts.stamp > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      backgroundColor: COLORS.stamp.box,
                    }}
                  />
                  <span style={{ color: 'white', fontSize: '14px' }}>
                    {counts.stamp} Stamp{counts.stamp !== 1 ? 's' : ''}
                  </span>
                </div>
              )}

              {counts.signature > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      backgroundColor: COLORS.signature.box,
                    }}
                  />
                  <span style={{ color: 'white', fontSize: '14px' }}>
                    {counts.signature} Signature{counts.signature !== 1 ? 's' : ''}
                  </span>
                </div>
              )}

              {counts.qr > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      backgroundColor: COLORS.qr.box,
                    }}
                  />
                  <span style={{ color: 'white', fontSize: '14px' }}>
                    {counts.qr} QR Code{counts.qr !== 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Performance stats */}
        {(fps > 0 || latency > 0) && (
          <div
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              backdropFilter: 'blur(8px)',
              borderRadius: '12px',
              padding: '8px 12px',
              display: 'flex',
              gap: '16px',
            }}
          >
            {fps > 0 && (
              <div style={{ color: 'rgba(255, 255, 255, 0.8)', fontSize: '12px' }}>
                FPS: <span style={{ fontWeight: 'bold', color: 'white' }}>{fps.toFixed(1)}</span>
              </div>
            )}
            {latency > 0 && (
              <div style={{ color: 'rgba(255, 255, 255, 0.8)', fontSize: '12px' }}>
                Latency: <span style={{ fontWeight: 'bold', color: 'white' }}>{latency.toFixed(0)}ms</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hint message when no detections */}
      {totalDetections === 0 && (
        <div
          style={{
            position: 'absolute',
            bottom: '80px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 20,
            pointerEvents: 'none',
          }}
        >
          <div
            style={{
              backgroundColor: 'rgba(0, 23, 255, 0.9)',
              backdropFilter: 'blur(8px)',
              borderRadius: '24px',
              padding: '12px 24px',
              color: 'white',
              fontSize: '14px',
              fontWeight: '500',
              whiteSpace: 'nowrap',
            }}
          >
            Point camera at documents with stamps, signatures, or QR codes
          </div>
        </div>
      )}
    </>
  );
}
