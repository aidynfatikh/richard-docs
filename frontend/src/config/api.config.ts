/**
 * API Configuration
 * Central configuration for all API endpoints and settings
 */

export const API_CONFIG = {
  // Base URL for the backend API
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  
  // API endpoints
  ENDPOINTS: {
    DETECT: '/detect',
    HEALTH: '/health',
    SCAN_DOCUMENT: '/scan-document',
    BATCH_DETECT: '/batch-detect',
  },
  
  // Request timeout in milliseconds
  TIMEOUT: 30000, // 30 seconds for file processing
  
  // Maximum file size in bytes (50MB)
  MAX_FILE_SIZE: 50 * 1024 * 1024,
  
  // Maximum number of files per request
  MAX_FILES_PER_REQUEST: 10,
} as const;
