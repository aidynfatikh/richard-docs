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
  results: Array<(DetectionResponse | MultiPageDetectionResponse) & {
    file_index: number;
    filename: string;
    success: boolean;
    error?: string;
    classification?: string;
    scan_applied?: boolean;
  }>;
  summary: DetectionSummary;
  meta: {
    total_processing_time_ms: number;
    avg_time_per_file_ms: number;
    confidence_threshold: number;
    parallel_workers: number;
  };
}

/**
 * Intelligent batch processing response (with classification + scanning metadata)
 */
export interface BatchProcessDocumentResultSuccess {
  file_index: number;
  filename: string;
  success: true;
  document_type: 'image' | 'pdf';
  processing: ProcessingMetadata;
  summary: DetectionSummary;
  transformed_image?: string;
  result?: DetectionResponse;
  pages?: DetectionResponse[];
}

export interface BatchProcessDocumentResultError {
  file_index: number;
  filename: string;
  success: false;
  error: string;
}

export type BatchProcessDocumentResult =
  | BatchProcessDocumentResultSuccess
  | BatchProcessDocumentResultError;

export interface BatchProcessDocumentResponse {
  total_files: number;
  successful_documents: number;
  failed_documents: number;
  results: BatchProcessDocumentResult[];
  summary: DetectionSummary;
  meta: {
    total_processing_time_ms: number;
    read_time_ms: number;
    preprocessing_time_ms: number;
    detection_time_ms: number;
    confidence_threshold: number;
    documents_ready_for_detection: number;
    pages_processed: number;
    max_workers: number;
  };
}

/**
 * Document classification indicators from EXIF
 */
export interface ExifIndicators {
  has_exif?: boolean;
  make?: string;
  model?: string;
  software?: string;
  is_phone_camera?: boolean;
  is_phone_model?: boolean;
  is_digital_software?: boolean;
  has_iso?: boolean;
  has_aperture?: boolean;
  has_focal_length?: boolean;
  has_exposure_time?: boolean;
  has_gps?: boolean;
  has_orientation?: boolean;
  has_datetime_original?: boolean;
  phone_aspect_ratio?: boolean;
  iso?: number;
  exif_error?: string;
}

/**
 * Document classification indicators from visual analysis
 */
export interface VisualIndicators {
  document_contour?: boolean;
  contour_area_ratio?: number;
  document_within_frame?: boolean;
  document_fills_frame?: boolean;
  edges_touch_boundaries?: boolean;
  edges_have_margin?: boolean;
  perspective_distortion?: boolean;
  visible_background?: boolean;
  blur_variance?: number;
  has_focus_variation?: boolean;
  non_uniform_lighting?: boolean;
  visual_error?: string;
}

/**
 * Document classification result
 */
export interface ClassificationResult {
  classification: 'camera_photo' | 'digital_document';
  confidence: number;
  scores: {
    exif: number;
    visual: number;
    final: number;
  };
  indicators: {
    exif: ExifIndicators;
    visual: VisualIndicators;
  };
  recommendation: 'apply_perspective_correction' | 'use_as_is';
  threshold?: number;
  reason?: string;
  error?: string;
}

/**
 * Scan metadata from DocScanner
 */
export interface ScanMetadata {
  applied: boolean;
  scan_success?: boolean;
  corners_detected?: [[number, number], [number, number], [number, number], [number, number]];
  error?: string;
  fallback_to_original?: boolean;
  reason?: string;
}

/**
 * Processing metadata for intelligent document processing
 */
export interface ProcessingMetadata {
  filename?: string;
  file_size_bytes?: number;
  format?: string;
  classification?: ClassificationResult;
  scan_reason?: string;
  scan?: ScanMetadata;
  page_count?: number;
  preprocessing_time_ms?: number;
}

/**
 * Intelligent document processing response
 * Extends DetectionResponse with classification and processing metadata
 */
export interface ProcessDocumentResponse extends DetectionResponse {
  processing: ProcessingMetadata;
  transformed_image?: string; // Base64-encoded scanned image (if perspective correction applied)
}

/**
 * Multi-page intelligent processing response
 */
export interface MultiPageProcessDocumentResponse extends MultiPageDetectionResponse {
  processing: ProcessingMetadata;
}

/**
 * Classification-only response
 */
export interface ClassifyDocumentResponse {
  filename?: string;
  file_size_bytes?: number;
  is_camera_photo: boolean;
  classification: 'camera_photo' | 'digital_document';
  confidence: number;
  recommendation: 'apply_perspective_correction' | 'use_as_is';
  details: ClassificationResult;
}
