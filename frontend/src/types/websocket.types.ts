/**
 * WebSocket types for real-time detection
 */

/**
 * Detection result from real-time detector
 */
export interface DetectionCoordinates {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RealtimeDetection {
  coordinates: DetectionCoordinates;
  normalized_coordinates?: DetectionCoordinates;
  confidence: number;
  class: 'stamp' | 'signature' | 'qr';
  grouped?: boolean;
  group_count?: number;
}

/**
 * Categorized detection (matching /detect format)
 */
export interface CategorizedDetection {
  bbox: number[]; // [x1, y1, x2, y2]
  confidence: number;
  class_name: 'stamp' | 'signature' | 'qr';
  grouped?: boolean;
  group_count?: number;
}

/**
 * Classification result for document photo detection
 */
export interface ClassificationResult {
  is_camera_photo: boolean;
  confidence: number;
  reasons: string[];
  exif_data: Record<string, any>;
  visual_features: Record<string, any>;
  classification_time_ms: number;
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
 * Detection summary
 */
export interface DetectionSummary {
  total_stamps: number;
  total_signatures: number;
  total_qrs: number;
  total_detections: number;
}

/**
 * Extended performance metadata
 */
export interface ExtendedPerformanceMetadata extends PerformanceMetadata {
  inference_time_ms: number;
  total_processing_time_ms: number;
}

/**
 * Successful detection response from server (new format matching /detect)
 */
export interface WebSocketDetectionResponse {
  image_size: ImageSize;
  stamps: CategorizedDetection[];
  signatures: CategorizedDetection[];
  qrs: CategorizedDetection[];
  summary: DetectionSummary;
  classification: ClassificationResult;
  meta?: ExtendedPerformanceMetadata;
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
  return 'stamps' in response && 'signatures' in response && 'qrs' in response;
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
