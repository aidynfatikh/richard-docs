/**
 * Device detection utilities for mobile-specific features
 */

export interface DeviceInfo {
  isMobile: boolean;
  isIOS: boolean;
  isAndroid: boolean;
  isTouchDevice: boolean;
}

/**
 * Detects if the current device is a mobile device
 */
export function detectDevice(): DeviceInfo {
  const userAgent = navigator.userAgent || navigator.vendor || (window as any).opera;

  // Check for mobile devices
  const isMobileUA = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(
    userAgent.toLowerCase()
  );

  // Additional check for tablets and mobile browsers
  const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

  // More reliable mobile detection using media queries
  const isMobileScreen = window.matchMedia('(max-width: 768px)').matches;

  // Combine checks
  const isMobile = isMobileUA || (isTouchDevice && isMobileScreen);

  // Detect iOS
  const isIOS = /iPad|iPhone|iPod/.test(userAgent) && !(window as any).MSStream;

  // Detect Android
  const isAndroid = /android/i.test(userAgent);

  return {
    isMobile,
    isIOS,
    isAndroid,
    isTouchDevice
  };
}

/**
 * Checks if the browser supports camera access
 */
export function supportsCameraAccess(): boolean {
  return !!(
    navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia &&
    window.MediaStreamTrack
  );
}

/**
 * Gets recommended camera constraints for document scanning
 */
export function getCameraConstraints(facingMode: 'user' | 'environment' = 'environment'): MediaStreamConstraints {
  return {
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 16 / 9 }
    },
    audio: false
  };
}

/**
 * Checks if device is in landscape mode
 */
export function isLandscape(): boolean {
  return window.matchMedia('(orientation: landscape)').matches;
}
