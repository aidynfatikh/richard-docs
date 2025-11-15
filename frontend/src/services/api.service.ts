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
  ): Promise<APIResponse<DetectionResponse>> {
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
        data: result.data
      });
      
      // Handle multi-page PDF response - convert to single page format
      if (result.success && result.data && 'document_type' in result.data) {
        const multiPageData = result.data as MultiPageDetectionResponse;
        
        console.log('Multi-page PDF detected:', {
          totalPages: multiPageData.total_pages,
          firstPageHasImage: !!multiPageData.pages[0]?.page_image,
          firstPageImagePrefix: multiPageData.pages[0]?.page_image?.substring(0, 50)
        });
        
        // Aggregate all pages into a single result
        const allStamps = multiPageData.pages.flatMap(page => page.stamps);
        const allSignatures = multiPageData.pages.flatMap(page => page.signatures);
        const allQRs = multiPageData.pages.flatMap(page => page.qrs);
        
        // Use the first page's image size (or could use the largest page)
        const imageSize = multiPageData.pages[0]?.image_size || { width_px: 0, height_px: 0 };
        
        // Get the first page's base64 image if available
        const pageImage = multiPageData.pages[0]?.page_image;
        
        const aggregatedResponse: DetectionResponse = {
          image_size: imageSize,
          stamps: allStamps,
          signatures: allSignatures,
          qrs: allQRs,
          summary: multiPageData.summary,
          meta: {
            model_version: multiPageData.pages[0]?.meta.model_version || 'unknown',
            total_processing_time_ms: multiPageData.meta.total_processing_time_ms,
            confidence_threshold: multiPageData.meta.confidence_threshold,
            is_pdf: true,
          },
          ...(pageImage && { page_image: pageImage }) // Include page_image if it exists
        };
        
        console.log('Aggregated response:', {
          hasPageImage: !!aggregatedResponse.page_image,
          pageImagePrefix: aggregatedResponse.page_image?.substring(0, 50)
        });
        
        return {
          success: true,
          data: aggregatedResponse
        };
      }
      
      return result as APIResponse<DetectionResponse>;
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
  ): Promise<Array<{ file: string; result: APIResponse<DetectionResponse> }>> {
    const results: Array<{ file: string; result: APIResponse<DetectionResponse> }> = [];

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
}

// Export singleton instance
export const apiService = new APIService();

// Export class for testing or custom instances
export default APIService;
