/**
 * API Type Definitions
 * TypeScript interfaces matching backend response schemas
 */

/**
 * Detection result for a single element (stamp, signature, or QR code)
 */
export interface Detection {
  id: string;
  bbox: [number, number, number, number]; // [x_min, y_min, x_max, y_max]
  confidence: number;
  class_name: 'stamp' | 'signature' | 'qr';
  class_id: number;
}

/**
 * Summary of all detections
 */
export interface DetectionSummary {
  total_stamps: number;
  total_signatures: number;
  total_qrs: number;
  total_detections: number;
}

/**
 * Metadata about the detection process
 */
export interface DetectionMeta {
  model_version: string;
  inference_time_ms: number;
  total_processing_time_ms: number;
  confidence_threshold: number;
}

/**
 * Image size information
 */
export interface ImageSize {
  width_px: number;
  height_px: number;
}

/**
 * Complete detection response from backend
 */
export interface DetectionResponse {
  image_size: ImageSize;
  stamps: Detection[];
  signatures: Detection[];
  qrs: Detection[];
  summary: DetectionSummary;
  meta: DetectionMeta;
}

/**
 * API Error response
 */
export interface APIError {
  detail: string;
  status?: number;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: 'healthy' | 'model_not_loaded';
  model_loaded: boolean;
  model_path: string | null;
  classes: Record<number, string> | null;
}

/**
 * Generic API response wrapper
 */
export interface APIResponse<T> {
  data?: T;
  error?: APIError;
  success: boolean;
}

/**
 * Document scan response
 */
export interface ScanDocumentResponse {
  success: boolean;
  transformed_image: string; // base64 data URL
  corners: [[number, number], [number, number], [number, number], [number, number]]; // 4 corner coordinates
}

/**
 * Batch detection response
 */
export interface BatchDetectionResponse {
  total_files: number;
  successful_detections: number;
  failed_detections: number;
  results: Array<DetectionResponse & {
    file_index: number;
    filename: string;
    success: boolean;
    error?: string;
  }>;
  summary: DetectionSummary;
  meta: {
    total_processing_time_ms: number;
    confidence_threshold: number;
  };
}
