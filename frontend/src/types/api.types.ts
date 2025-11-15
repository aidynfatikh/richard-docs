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
  grouped?: boolean;
  group_count?: number;
}

/**
 * Summary of all detections
 */
export interface DetectionSummary {
  total_stamps: number;
  total_signatures: number;
  total_qrs: number;
  total_detections: number;
  raw_detections?: number;
  grouped_detections?: number | null;
}

/**
 * Metadata about the detection process
 */
export interface DetectionMeta {
  model_version: string;
  inference_time_ms?: number;
  total_processing_time_ms: number;
  confidence_threshold: number;
  grouping_enabled?: boolean;
  group_iou_threshold?: number;
  page_number?: number;
  page_processing_time_ms?: number;
  is_pdf?: boolean;
}

/**
 * Image size information
 */
export interface ImageSize {
  width_px: number;
  height_px: number;
}

/**
 * Complete detection response from backend (single page)
 */
export interface DetectionResponse {
  image_size: ImageSize;
  stamps: Detection[];
  signatures: Detection[];
  qrs: Detection[];
  summary: DetectionSummary;
  meta: DetectionMeta;
  page_image?: string; // Base64 image data for PDF pages
}

/**
 * Multi-page PDF detection response from backend
 */
export interface MultiPageDetectionResponse {
  document_type: 'pdf';
  total_pages: number;
  pages: DetectionResponse[];
  summary: DetectionSummary;
  meta: {
    total_processing_time_ms: number;
    confidence_threshold: number;
    is_pdf: boolean;
  };
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
