/**
 * WebSocket service for real-time detection
 */

import type {
  WebSocketFrameMessage,
  WebSocketResponse,
  WebSocketDetectionResponse,
  WebSocketState
} from '../types/websocket.types';
import { WEB_SOCKET_STATE } from '../types/websocket.types';

export class RealtimeDetectionService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private isManualClose = false;

  // Callbacks
  private onDetectionCallback?: (result: WebSocketDetectionResponse) => void;
  private onErrorCallback?: (error: string) => void;
  private onStateChangeCallback?: (state: WebSocketState) => void;

  constructor(apiUrl: string) {
    // Convert HTTP URL to WebSocket URL
    const wsProtocol = apiUrl.startsWith('https') ? 'wss' : 'ws';
    const urlWithoutProtocol = apiUrl.replace(/^https?:\/\//, '');
    this.url = `${wsProtocol}://${urlWithoutProtocol}/ws/detect`;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.isManualClose = false;
        this.ws = new WebSocket(this.url);

        this.notifyStateChange(WEB_SOCKET_STATE.CONNECTING);

        this.ws.onopen = () => {
          console.log('[WebSocket] Connected to real-time detection server');
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.notifyStateChange(WEB_SOCKET_STATE.CONNECTED);
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const response: WebSocketResponse = JSON.parse(event.data);

            if ('error' in response) {
              console.error('[WebSocket] Server error:', response.error, response.message);
              if (this.onErrorCallback) {
                this.onErrorCallback(response.message);
              }
            } else if ('detections' in response) {
              // Valid detection response
              if (this.onDetectionCallback) {
                this.onDetectionCallback(response);
              }
            }
          } catch (error) {
            console.error('[WebSocket] Failed to parse message:', error);
            if (this.onErrorCallback) {
              this.onErrorCallback('Failed to parse server response');
            }
          }
        };

        this.ws.onerror = (error) => {
          console.error('[WebSocket] Connection error:', error);
          this.notifyStateChange(WEB_SOCKET_STATE.ERROR);
          if (this.onErrorCallback) {
            this.onErrorCallback('WebSocket connection error');
          }
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.onclose = (event) => {
          console.log('[WebSocket] Connection closed:', event.code, event.reason);
          this.notifyStateChange(WEB_SOCKET_STATE.DISCONNECTED);
          this.ws = null;

          // Auto-reconnect if not manually closed
          if (!this.isManualClose && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[WebSocket] Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
              this.connect().catch(err => {
                console.error('[WebSocket] Reconnection failed:', err);
              });
            }, this.reconnectDelay);

            // Exponential backoff
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
          } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            if (this.onErrorCallback) {
              this.onErrorCallback('Failed to reconnect after multiple attempts');
            }
          }
        };

      } catch (error) {
        console.error('[WebSocket] Failed to create connection:', error);
        this.notifyStateChange(WEB_SOCKET_STATE.ERROR);
        reject(error);
      }
    });
  }

  /**
   * Send video frame for detection
   */
  sendFrame(frameDataUrl: string): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send frame: connection not open');
      return false;
    }

    try {
      const message: WebSocketFrameMessage = {
        frame: frameDataUrl
      };

      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('[WebSocket] Failed to send frame:', error);
      if (this.onErrorCallback) {
        this.onErrorCallback('Failed to send frame');
      }
      return false;
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.isManualClose = true;

    if (this.ws) {
      console.log('[WebSocket] Disconnecting...');
      this.ws.close(1000, 'Client requested disconnect');
      this.ws = null;
    }

    this.notifyStateChange(WEB_SOCKET_STATE.DISCONNECTED);
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get current connection state
   */
  getState(): WebSocketState {
    if (!this.ws) {
      return WEB_SOCKET_STATE.DISCONNECTED;
    }

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return WEB_SOCKET_STATE.CONNECTING;
      case WebSocket.OPEN:
        return WEB_SOCKET_STATE.CONNECTED;
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
        return WEB_SOCKET_STATE.DISCONNECTED;
      default:
        return WEB_SOCKET_STATE.ERROR;
    }
  }

  /**
   * Register callback for detection results
   */
  onDetection(callback: (result: WebSocketDetectionResponse) => void): void {
    this.onDetectionCallback = callback;
  }

  /**
   * Register callback for errors
   */
  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Register callback for state changes
   */
  onStateChange(callback: (state: WebSocketState) => void): void {
    this.onStateChangeCallback = callback;
  }

  /**
   * Notify state change to callback
   */
  private notifyStateChange(state: WebSocketState): void {
    if (this.onStateChangeCallback) {
      this.onStateChangeCallback(state);
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.disconnect();
    this.onDetectionCallback = undefined;
    this.onErrorCallback = undefined;
    this.onStateChangeCallback = undefined;
  }
}

/**
 * Helper function to capture frame from video element
 */
export function captureFrameFromVideo(
  videoElement: HTMLVideoElement,
  quality: number = 0.8
): string | null {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Failed to get canvas context');
      return null;
    }

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert to JPEG data URL with specified quality
    return canvas.toDataURL('image/jpeg', quality);
  } catch (error) {
    console.error('Failed to capture frame:', error);
    return null;
  }
}

/**
 * Helper function to calculate optimal frame capture interval
 * based on desired FPS and system capabilities
 */
export function calculateFrameInterval(targetFPS: number): number {
  // Cap at reasonable limits
  const cappedFPS = Math.max(1, Math.min(targetFPS, 30));
  return 1000 / cappedFPS; // Convert to milliseconds
}
