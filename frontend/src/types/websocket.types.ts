/**
 * WebSocket types for real-time detection
 */

/**
 * Detection result from real-time detector
 */
export interface RealtimeDetection {
  bbox: [number, number, number, number]; // [x_min, y_min, x_max, y_max]
  confidence: number;
  class: 'stamp' | 'signature' | 'qr';
}

/**
 * Detection counts by type
 */
export interface DetectionCounts {
  stamp: number;
  signature: number;
  qr: number;
}

/**
 * Image size metadata
 */
export interface ImageSize {
  width: number;
  height: number;
}

/**
 * Performance metadata
 */
export interface PerformanceMetadata {
  frame_count: number;
  avg_inference_time_ms: number;
}

/**
 * Message sent from client to server
 */
export interface WebSocketFrameMessage {
  frame: string; // Base64-encoded JPEG image
}

/**
 * Successful detection response from server
 */
export interface WebSocketDetectionResponse {
  detections: RealtimeDetection[];
  counts: DetectionCounts;
  image_size: ImageSize;
  inference_time_ms: number;
  meta?: PerformanceMetadata;
}

/**
 * Error response from server
 */
export interface WebSocketErrorResponse {
  error: string;
  message: string;
}

/**
 * Union type for all possible server responses
 */
export type WebSocketResponse = WebSocketDetectionResponse | WebSocketErrorResponse;

/**
 * Type guard to check if response is an error
 */
export function isWebSocketError(response: WebSocketResponse): response is WebSocketErrorResponse {
  return 'error' in response;
}

/**
 * Type guard to check if response is a detection result
 */
export function isWebSocketDetection(response: WebSocketResponse): response is WebSocketDetectionResponse {
  return 'detections' in response;
}

/**
 * WebSocket connection state
 */
export const WEB_SOCKET_STATE = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error'
} as const;

export type WebSocketState = typeof WEB_SOCKET_STATE[keyof typeof WEB_SOCKET_STATE];

/**
 * Camera configuration
 */
export interface CameraConfig {
  facingMode?: 'user' | 'environment'; // Front or rear camera
  width?: number;
  height?: number;
  frameRate?: number;
}

/**
 * Detection statistics for display
 */
export interface DetectionStats {
  totalDetections: number;
  stamps: number;
  signatures: number;
  qrs: number;
  avgConfidence: number;
  fps: number;
  latency: number;
}
