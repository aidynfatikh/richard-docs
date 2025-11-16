/**
 * API Configuration
 * Central configuration for all API endpoints and settings
 */

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0']);

function isBrowser(): boolean {
  return typeof window !== 'undefined';
}

function isLocalHost(hostname: string | null | undefined): boolean {
  if (!hostname) {
    return false;
  }
  return LOCAL_HOSTS.has(hostname);
}

function getHostname(url: string): string | null {
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

function getBrowserOrigin(): string | null {
  if (!isBrowser()) {
    return null;
  }
  const { protocol, hostname, port } = window.location;
  const portSegment = port ? `:${port}` : '';
  return `${protocol}//${hostname}${portSegment}`;
}

function resolveBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_URL?.trim();
  const browserOrigin = getBrowserOrigin();
  const browserHost = browserOrigin ? new URL(browserOrigin).hostname : null;

  if (envUrl) {
    const envHost = getHostname(envUrl);

    // Prevent attempts to call localhost APIs when the app is hosted remotely.
    if (browserOrigin && !isLocalHost(browserHost) && isLocalHost(envHost)) {
      console.warn('[API] VITE_API_URL points to a local address but the app is served remotely. Falling back to current origin.');
      return browserOrigin;
    }

    return envUrl.replace(/\/+$/, '');
  }

  if (browserOrigin) {
    return isLocalHost(browserHost) ? 'http://localhost:8000' : browserOrigin.replace(/\/+$/, '');
  }

  return 'http://localhost:8000';
}

export const API_CONFIG = {
  // Base URL for the backend API
  BASE_URL: resolveBaseUrl(),
  
  // API endpoints
  ENDPOINTS: {
    DETECT: '/detect',
    PROCESS_DOCUMENT: '/process-document', // NEW: Intelligent processing with auto-classification
    CLASSIFY_DOCUMENT: '/classify-document', // NEW: Classification only
    HEALTH: '/health',
    SCAN_DOCUMENT: '/scan-document',
    BATCH_DETECT: '/batch-detect',
    BATCH_PROCESS_DOCUMENT: '/batch-process-document',
    BATCH_DETECT_HQ: '/batch-detect-hq', // High-quality mode (slower but more accurate)
  },
  
  // Request timeout in milliseconds
  TIMEOUT: 30000, // 30 seconds for file processing
  
  // Maximum file size in bytes (50MB)
  MAX_FILE_SIZE: 50 * 1024 * 1024,
  
  // Maximum number of files per request
  MAX_FILES_PER_REQUEST: 10,
} as const;
