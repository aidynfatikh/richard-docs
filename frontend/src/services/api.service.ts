/**
 * API Service
 * Centralized service for all backend API calls
 * Uses native fetch API with TypeScript for type safety
 */

import { API_CONFIG } from '../config/api.config';
import type {
  DetectionResponse,
  MultiPageDetectionResponse,
  HealthResponse,
  APIResponse,
  APIError,
  ScanDocumentResponse,
  BatchDetectionResponse,
  ProcessDocumentResponse,
  MultiPageProcessDocumentResponse,
  ClassifyDocumentResponse,
} from '../types/api.types';

/**
 * Custom error class for API errors
 */
export class APIException extends Error {
  status?: number;
  detail?: string;

  constructor(
    message: string,
    status?: number,
    detail?: string
  ) {
    super(message);
    this.name = 'APIException';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Base fetch wrapper with timeout and error handling
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = API_CONFIG.TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIException('Request timeout. Please try again.', 408);
    }
    throw error;
  }
}

/**
 * Handle API response and errors
 */
async function handleResponse<T>(response: Response): Promise<APIResponse<T>> {
  try {
    // Check if response is ok (status 200-299)
    if (!response.ok) {
      const errorData: APIError = await response.json().catch(() => ({
        detail: response.statusText || 'Unknown error occurred',
      }));

      return {
        success: false,
        error: {
          detail: errorData.detail,
          status: response.status,
        },
      };
    }

    // Parse successful response
    const data: T = await response.json();
    return {
      success: true,
      data,
    };
  } catch (error) {
    return {
      success: false,
      error: {
        detail: error instanceof Error ? error.message : 'Failed to parse response',
        status: response.status,
      },
    };
  }
}

/**
 * API Service Class
 */
class APIService {
  private baseURL: string;

  constructor(baseURL: string = API_CONFIG.BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Construct full URL from endpoint
   */
  private getURL(endpoint: string): string {
    return `${this.baseURL}${endpoint}`;
  }

  /**
   * Health check - verify backend is running
   */
  async healthCheck(): Promise<APIResponse<HealthResponse>> {
    try {
      const response = await fetchWithTimeout(
        this.getURL(API_CONFIG.ENDPOINTS.HEALTH),
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true', // Required for ngrok URLs
          },
        },
        5000 // Shorter timeout for health check
      );

      return handleResponse<HealthResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Health check failed',
        },
      };
    }
  }

  /**
   * Detect document elements (stamps, signatures, QR codes)
   * @param file - Image file to analyze
   * @param confidence - Optional confidence threshold (0-1)
   */
  async detectDocumentElements(
    file: File,
    confidence?: number
  ): Promise<APIResponse<DetectionResponse | MultiPageDetectionResponse>> {
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);

      // Add optional confidence parameter
      const url = new URL(this.getURL(API_CONFIG.ENDPOINTS.DETECT));
      if (confidence !== undefined) {
        url.searchParams.append('confidence', confidence.toString());
      }

      const response = await fetchWithTimeout(url.toString(), {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true', // Required for ngrok URLs
        },
        // Don't set Content-Type header - browser will set it with boundary for multipart/form-data
      });

      const result = await handleResponse<DetectionResponse | MultiPageDetectionResponse>(response);
      
      console.log('API Response received:', {
        success: result.success,
        hasData: !!result.data,
        isMultiPage: result.data && 'document_type' in result.data,
      });
      
      // Keep multi-page PDFs as-is, don't aggregate
      // The frontend will handle displaying all pages
      return result as APIResponse<DetectionResponse | MultiPageDetectionResponse>;
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Detection request failed',
        },
      };
    }
  }

  /**
   * Process multiple files sequentially
   * Returns array of results with file names
   */
  async detectMultipleDocuments(
    files: File[],
    confidence?: number,
    onProgress?: (current: number, total: number) => void
  ): Promise<Array<{ file: string; result: APIResponse<DetectionResponse | MultiPageDetectionResponse> }>> {
    const results: Array<{ file: string; result: APIResponse<DetectionResponse | MultiPageDetectionResponse> }> = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      // Call progress callback if provided
      if (onProgress) {
        onProgress(i + 1, files.length);
      }

      const result = await this.detectDocumentElements(file, confidence);
      results.push({
        file: file.name,
        result,
      });

      // Small delay between requests to avoid overwhelming the server
      if (i < files.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    return results;
  }

  /**
   * Scan a document image to detect boundaries and get perspective-corrected version
   * @param file - Image file to scan
   */
  async scanDocument(file: File): Promise<APIResponse<ScanDocumentResponse>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetchWithTimeout(
        this.getURL(API_CONFIG.ENDPOINTS.SCAN_DOCUMENT),
        {
          method: 'POST',
          body: formData,
        }
      );

      return handleResponse<ScanDocumentResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Document scan request failed',
        },
      };
    }
  }

  /**
   * Detect document elements in multiple images in a single batch request
   * OPTIMIZED for 1000+ files with chunking, progress tracking, and parallel processing
   * @param files - Array of image/PDF files
   * @param confidence - Optional confidence threshold (0-1)
   * @param onProgress - Optional progress callback (current, total)
   * @param chunkSize - Files per batch chunk (default 100)
   * @param maxWorkers - Parallel workers on backend (default 10)
   */
  async batchDetect(
    files: File[],
    confidence?: number,
    onProgress?: (current: number, total: number) => void,
    chunkSize: number = 100,
    maxWorkers: number = 10
  ): Promise<APIResponse<BatchDetectionResponse>> {
    try {
      if (files.length === 0) {
        return {
          success: false,
          error: { detail: 'No files provided' }
        };
      }

      // For large batches, process in chunks to avoid timeouts
      if (files.length > chunkSize) {
        return await this.batchDetectChunked(files, confidence, onProgress, chunkSize, maxWorkers);
      }

      // Single batch for smaller sets
      const formData = new FormData();

      // Append all files with the same key name
      files.forEach((file) => {
        formData.append('files', file);
      });

      // Add optional parameters
      const url = new URL(this.getURL(API_CONFIG.ENDPOINTS.BATCH_DETECT));
      if (confidence !== undefined) {
        url.searchParams.append('confidence', confidence.toString());
      }
      url.searchParams.append('max_workers', maxWorkers.toString());

      const response = await fetchWithTimeout(
        url.toString(),
        {
          method: 'POST',
          body: formData,
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        },
        120000 // 120 second timeout for batch processing
      );

      const result = await handleResponse<BatchDetectionResponse>(response);
      
      if (onProgress) {
        onProgress(files.length, files.length);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Batch detection request failed',
        },
      };
    }
  }

  /**
   * Process large batches in chunks with progress tracking
   * @private
   */
  private async batchDetectChunked(
    files: File[],
    confidence?: number,
    onProgress?: (current: number, total: number) => void,
    chunkSize: number = 100,
    maxWorkers: number = 10
  ): Promise<APIResponse<BatchDetectionResponse>> {
    try {
      const chunks: File[][] = [];
      for (let i = 0; i < files.length; i += chunkSize) {
        chunks.push(files.slice(i, i + chunkSize));
      }

      console.log(`Processing ${files.length} files in ${chunks.length} chunks of ${chunkSize}`);

      let allResults: BatchDetectionResponse['results'] = [];
      let totalProcessingTime = 0;
      let processedCount = 0;

      // Process chunks sequentially to avoid overwhelming server
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        
        const formData = new FormData();
        chunk.forEach((file) => {
          formData.append('files', file);
        });

        const url = new URL(this.getURL(API_CONFIG.ENDPOINTS.BATCH_DETECT));
        if (confidence !== undefined) {
          url.searchParams.append('confidence', confidence.toString());
        }
        url.searchParams.append('max_workers', maxWorkers.toString());

        const response = await fetchWithTimeout(
          url.toString(),
          {
            method: 'POST',
            body: formData,
            headers: {
              'ngrok-skip-browser-warning': 'true',
            },
          },
          120000
        );

        const chunkResult = await handleResponse<BatchDetectionResponse>(response);
        
        if (!chunkResult.success || !chunkResult.data) {
          throw new Error(`Chunk ${i + 1} failed: ${chunkResult.error?.detail}`);
        }

        // Adjust file indices to be global
        const adjustedResults = chunkResult.data.results.map(r => ({
          ...r,
          file_index: r.file_index + (i * chunkSize)
        }));

        allResults = [...allResults, ...adjustedResults];
        totalProcessingTime += chunkResult.data.meta.total_processing_time_ms;
        processedCount += chunk.length;

        // Update progress
        if (onProgress) {
          onProgress(processedCount, files.length);
        }

        console.log(`Chunk ${i + 1}/${chunks.length} completed (${processedCount}/${files.length} files)`);
      }

      // Aggregate results
      const successfulResults = allResults.filter(r => r.success);
      const failedResults = allResults.filter(r => !r.success);

      // Calculate aggregate summary
      const totalDetections = successfulResults.reduce((sum, r) => 
        sum + (r.summary?.total_detections || 0), 0
      );
      const totalStamps = successfulResults.reduce((sum, r) => 
        sum + (r.summary?.total_stamps || 0), 0
      );
      const totalSignatures = successfulResults.reduce((sum, r) => 
        sum + (r.summary?.total_signatures || 0), 0
      );
      const totalQrs = successfulResults.reduce((sum, r) => 
        sum + (r.summary?.total_qrs || 0), 0
      );

      const aggregatedResponse: BatchDetectionResponse = {
        total_files: files.length,
        successful_detections: successfulResults.length,
        failed_detections: failedResults.length,
        results: allResults,
        summary: {
          total_detections: totalDetections,
          total_stamps: totalStamps,
          total_signatures: totalSignatures,
          total_qrs: totalQrs,
        },
        meta: {
          total_processing_time_ms: totalProcessingTime,
          avg_time_per_file_ms: totalProcessingTime / files.length,
          confidence_threshold: confidence || 0.25,
          parallel_workers: maxWorkers
        }
      };

      return {
        success: true,
        data: aggregatedResponse
      };

    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Chunked batch processing failed',
        },
      };
    }
  }
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Batch detection request failed',
        },
      };
    }
  }

  /**
   * Intelligent document processing - auto-classifies and applies appropriate pipeline
   * @param file - Image file to process
   * @param confidence - Optional confidence threshold (0-1)
   * @param forceScan - Force perspective correction even if classified as digital
   * @param skipScan - Skip perspective correction even if classified as camera photo
   */
  async processDocument(
    file: File,
    confidence?: number,
    forceScan?: boolean,
    skipScan?: boolean
  ): Promise<APIResponse<ProcessDocumentResponse | MultiPageProcessDocumentResponse>> {
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);

      // Build URL with query parameters - use FormData approach like detectDocumentElements
      let urlString = this.getURL(API_CONFIG.ENDPOINTS.PROCESS_DOCUMENT);

      const params = new URLSearchParams();
      if (confidence !== undefined) {
        params.append('confidence', confidence.toString());
      }
      if (forceScan !== undefined) {
        params.append('force_scan', forceScan.toString());
      }
      if (skipScan !== undefined) {
        params.append('skip_scan', skipScan.toString());
      }

      if (params.toString()) {
        urlString += '?' + params.toString();
      }

      const response = await fetchWithTimeout(urlString, {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      });

      const result = await handleResponse<ProcessDocumentResponse | MultiPageProcessDocumentResponse>(response);

      console.log('Process Document Response:', {
        success: result.success,
        hasData: !!result.data,
        isMultiPage: result.data && 'document_type' in result.data,
        classification: result.data && 'processing' in result.data ? result.data.processing.classification : null,
      });

      return result;
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Process document request failed',
        },
      };
    }
  }

  /**
   * Classify document (camera photo vs digital document) without processing
   * @param file - Image file to classify
   */
  async classifyDocument(
    file: File
  ): Promise<APIResponse<ClassifyDocumentResponse>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetchWithTimeout(
        this.getURL(API_CONFIG.ENDPOINTS.CLASSIFY_DOCUMENT),
        {
          method: 'POST',
          body: formData,
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        }
      );

      return handleResponse<ClassifyDocumentResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: {
          detail: error instanceof Error ? error.message : 'Classify document request failed',
        },
      };
    }
  }

  /**
   * Convert a base64 data URL to a File object
   * @param dataUrl - Base64 data URL
   * @param filename - Filename for the created file
   */
  dataUrlToFile(dataUrl: string, filename: string): File {
    const arr = dataUrl.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  }
}

// Export singleton instance
export const apiService = new APIService();

// Export class for testing or custom instances
export default APIService;
