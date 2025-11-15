import { useEffect, useRef, useState } from 'react';
import type { Detection } from '../types/api.types';

interface ImageWithDetectionsProps {
  imageFile: File | null;
  stamps: Detection[];
  signatures: Detection[];
  qrs: Detection[];
  imageSize: {
    width_px: number;
    height_px: number;
  };
  base64Image?: string; // For PDF pages
}

const COLORS = {
  stamp: { box: 'rgb(239, 68, 68)', label: 'rgba(239, 68, 68, 0.9)' },
  signature: { box: 'rgb(34, 197, 94)', label: 'rgba(34, 197, 94, 0.9)' },
  qr: { box: 'rgb(59, 130, 246)', label: 'rgba(59, 130, 246, 0.9)' },
};

export function ImageWithDetections({
  imageFile,
  stamps,
  signatures,
  qrs,
  imageSize,
  base64Image,
}: ImageWithDetectionsProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    console.log('ImageWithDetections - Debug Info:', {
      hasBase64Image: !!base64Image,
      base64ImagePrefix: base64Image ? base64Image.substring(0, 50) : 'none',
      hasImageFile: !!imageFile,
      imageFileName: imageFile?.name || 'none'
    });

    // Load image
    const img = new Image();
    let blobUrl: string | null = null;
    
    // Use base64 image if available (PDF), otherwise use file object
    let imageSource: string | null = null;
    
    if (base64Image) {
      imageSource = base64Image;
      console.log('Using base64 image source');
    } else if (imageFile) {
      blobUrl = URL.createObjectURL(imageFile);
      imageSource = blobUrl;
      console.log('Using blob URL:', blobUrl);
    }
    
    if (!imageSource) {
      console.error('No image source available');
      return;
    }

    img.onload = () => {
      // Calculate display size (maintain aspect ratio, max width 800px)
      const maxWidth = 800;
      const scale = Math.min(maxWidth / imageSize.width_px, 1);
      const displayWidth = imageSize.width_px * scale;
      const displayHeight = imageSize.height_px * scale;

      setDisplaySize({ width: displayWidth, height: displayHeight });

      // Set canvas size to display size
      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Draw image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      // Draw detections
      const scaleX = displayWidth / imageSize.width_px;
      const scaleY = displayHeight / imageSize.height_px;

      // Helper function to draw a detection
      const drawDetection = (detection: Detection, color: { box: string; label: string }, type: string) => {
        const [x1, y1, x2, y2] = detection.bbox;
        
        // Scale coordinates
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sx2 = x2 * scaleX;
        const sy2 = y2 * scaleY;
        const width = sx2 - sx1;
        const height = sy2 - sy1;

        // Draw bounding box
        ctx.strokeStyle = color.box;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, width, height);

        // Draw label background
        const label = `${type} ${(detection.confidence * 100).toFixed(1)}%`;
        const groupInfo = (detection as any).grouped ? ` [G:${(detection as any).group_count}]` : '';
        const fullLabel = label + groupInfo;
        
        ctx.font = '14px Arial';
        const textMetrics = ctx.measureText(fullLabel);
        const textWidth = textMetrics.width;
        const textHeight = 20;

        // Position label above box, or below if too close to top
        const labelY = sy1 > textHeight + 5 ? sy1 - 5 : sy1 + height + textHeight;
        
        // Draw label background
        ctx.fillStyle = color.label;
        ctx.fillRect(sx1, labelY - textHeight, textWidth + 8, textHeight);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(fullLabel, sx1 + 4, labelY - 5);
      };

      // Draw all detections
      stamps.forEach(stamp => drawDetection(stamp, COLORS.stamp, 'Stamp'));
      signatures.forEach(sig => drawDetection(sig, COLORS.signature, 'Signature'));
      qrs.forEach(qr => drawDetection(qr, COLORS.qr, 'QR'));

      setImageLoaded(true);
    };

    img.onerror = () => {
      console.error('Failed to load image');
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };

    img.src = imageSource;

    // Cleanup: only revoke blob URL if we created one
    return () => {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [imageFile, base64Image, stamps, signatures, qrs, imageSize]);

  return (
    <div className="relative">
      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2" style={{ borderColor: 'rgba(0, 23, 255, 1)' }}></div>
        </div>
      )}
      <canvas
        ref={canvasRef}
        className="w-full h-auto rounded-lg"
        style={{
          maxWidth: '100%',
          opacity: imageLoaded ? 1 : 0,
          transition: 'opacity 0.3s ease-in-out',
        }}
      />
      {imageLoaded && (
        <div className="mt-2 text-xs text-center" style={{ color: 'rgba(153, 153, 153, 1)' }}>
          Original: {imageSize.width_px} × {imageSize.height_px} px
          {displaySize.width < imageSize.width_px && ' (scaled to fit)'}
        </div>
      )}
    </div>
  );
}
