/**
 * File Validation Utilities
 * Validates file types, sizes, and formats for document processing
 */

import { API_CONFIG } from '../config/api.config';

/**
 * Supported file types for document analysis
 * Backend currently processes images, but this is extensible for PDFs, DOCX, etc.
 */
export const SUPPORTED_FILE_TYPES = {
  // Images - Primary format for YOLO detection
  images: {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/webp': ['.webp'],
    'image/tiff': ['.tif', '.tiff'],
    'image/bmp': ['.bmp'],
  },
  
  // Documents - Future support (would need conversion to images)
  documents: {
    'application/pdf': ['.pdf'],
    // 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    // 'application/msword': ['.doc'],
  },
} as const;

/**
 * Get all accepted MIME types as a flat array
 */
export const getAllAcceptedMimeTypes = (): string[] => {
  return [
    ...Object.keys(SUPPORTED_FILE_TYPES.images),
    ...Object.keys(SUPPORTED_FILE_TYPES.documents),
  ];
};

/**
 * Get all accepted file extensions as a comma-separated string for input accept attribute
 */
export const getAcceptAttribute = (): string => {
  const extensions = [
    ...Object.values(SUPPORTED_FILE_TYPES.images).flat(),
    ...Object.values(SUPPORTED_FILE_TYPES.documents).flat(),
  ];
  return extensions.join(',');
};

/**
 * File validation result
 */
export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

/**
 * Validate a single file
 */
export const validateFile = (file: File): ValidationResult => {
  // Check if file exists
  if (!file) {
    return { isValid: false, error: 'No file provided' };
  }

  // Check file size
  if (file.size > API_CONFIG.MAX_FILE_SIZE) {
    const maxSizeMB = (API_CONFIG.MAX_FILE_SIZE / 1024 / 1024).toFixed(0);
    return {
      isValid: false,
      error: `File "${file.name}" exceeds maximum size of ${maxSizeMB}MB`,
    };
  }

  // Check if file is empty
  if (file.size === 0) {
    return {
      isValid: false,
      error: `File "${file.name}" is empty`,
    };
  }

  // Check file type
  const acceptedMimeTypes = getAllAcceptedMimeTypes();
  const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;
  
  const isTypeSupported = acceptedMimeTypes.includes(file.type) ||
    Object.values(SUPPORTED_FILE_TYPES.images).some(exts => (exts as readonly string[]).includes(fileExtension)) ||
    Object.values(SUPPORTED_FILE_TYPES.documents).some(exts => (exts as readonly string[]).includes(fileExtension));

  if (!isTypeSupported) {
    return {
      isValid: false,
      error: `File type not supported: ${file.name}. Please upload images (JPG, PNG, WEBP, TIFF, BMP) or PDFs.`,
    };
  }

  return { isValid: true };
};

/**
 * Validate multiple files
 */
export const validateFiles = (files: File[]): ValidationResult => {
  // Check if any files provided
  if (!files || files.length === 0) {
    return { isValid: false, error: 'No files selected' };
  }

  // Check maximum number of files
  if (files.length > API_CONFIG.MAX_FILES_PER_REQUEST) {
    return {
      isValid: false,
      error: `Maximum ${API_CONFIG.MAX_FILES_PER_REQUEST} files allowed per request`,
    };
  }

  // Validate each file
  for (const file of files) {
    const result = validateFile(file);
    if (!result.isValid) {
      return result;
    }
  }

  return { isValid: true };
};

/**
 * Format file size for display
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

/**
 * Get file type category (image or document)
 */
export const getFileCategory = (file: File): 'image' | 'document' | 'unknown' => {
  if (Object.keys(SUPPORTED_FILE_TYPES.images).includes(file.type)) {
    return 'image';
  }
  if (Object.keys(SUPPORTED_FILE_TYPES.documents).includes(file.type)) {
    return 'document';
  }
  return 'unknown';
};
