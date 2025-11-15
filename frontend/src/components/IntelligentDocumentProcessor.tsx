/**
 * Intelligent Document Processor Component
 * Showcases the AI-powered document classification and processing pipeline
 */

import { useState, useCallback } from 'react';
import { Upload, Camera, FileText, Zap, Check, X, AlertCircle, Sparkles } from 'lucide-react';
import { apiService } from '../services/api.service';
import type {
  ProcessDocumentResponse,
  MultiPageProcessDocumentResponse,
  ClassificationResult,
  ProcessingMetadata,
} from '../types/api.types';
import { ImageWithDetections } from './ImageWithDetections';

interface ProcessingState {
  isProcessing: boolean;
  currentStage: 'uploading' | 'classifying' | 'scanning' | 'detecting' | 'complete' | 'error';
  stageMessage: string;
}

export function IntelligentDocumentProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [processingState, setProcessingState] = useState<ProcessingState>({
    isProcessing: false,
    currentStage: 'uploading',
    stageMessage: '',
  });
  const [result, setResult] = useState<ProcessDocumentResponse | MultiPageProcessDocumentResponse | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [forceScan, setForceScan] = useState(false);
  const [skipScan, setSkipScan] = useState(false);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);

    // Create preview URL
    const reader = new FileReader();
    reader.onload = (event) => {
      setPreviewUrl(event.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  }, []);

  const processDocument = async () => {
    if (!file) return;

    setProcessingState({
      isProcessing: true,
      currentStage: 'uploading',
      stageMessage: 'Uploading document...',
    });

    try {
      // Stage 1: Classifying
      setProcessingState({
        isProcessing: true,
        currentStage: 'classifying',
        stageMessage: 'AI analyzing document type...',
      });

      // Call intelligent processing endpoint
      const response = await apiService.processDocument(file, 0.25, forceScan, skipScan);

      if (!response.success || !response.data) {
        throw new Error(response.error?.detail || 'Processing failed');
      }

      const data = response.data;

      // Stage 2: Scanning (if applied)
      const isSinglePage = !('document_type' in data);
      const processing = isSinglePage
        ? (data as ProcessDocumentResponse).processing
        : (data as MultiPageProcessDocumentResponse).processing;

      if (processing?.scan?.applied) {
        setProcessingState({
          isProcessing: true,
          currentStage: 'scanning',
          stageMessage: 'Applying perspective correction...',
        });
        await new Promise(resolve => setTimeout(resolve, 800));
      }

      // Stage 3: Detecting
      setProcessingState({
        isProcessing: true,
        currentStage: 'detecting',
        stageMessage: 'Detecting stamps, signatures, QR codes...',
      });
      await new Promise(resolve => setTimeout(resolve, 500));

      // Stage 4: Complete
      setProcessingState({
        isProcessing: true,
        currentStage: 'complete',
        stageMessage: 'Processing complete!',
      });

      setResult(data);

      // Auto-reset stage after showing complete
      setTimeout(() => {
        setProcessingState({
          isProcessing: false,
          currentStage: 'complete',
          stageMessage: '',
        });
      }, 1500);

    } catch (error) {
      console.error('Processing error:', error);
      setProcessingState({
        isProcessing: false,
        currentStage: 'error',
        stageMessage: error instanceof Error ? error.message : 'Processing failed',
      });
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setProcessingState({
      isProcessing: false,
      currentStage: 'uploading',
      stageMessage: '',
    });
    setForceScan(false);
    setSkipScan(false);
  };

  const renderClassificationBadge = (classification: ClassificationResult | undefined) => {
    if (!classification) return null;

    const isCameraPhoto = classification.classification === 'camera_photo';
    const confidence = classification.confidence;

    return (
      <div className="space-y-3">
        {/* Main classification badge */}
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${
          isCameraPhoto
            ? 'bg-blue-50 border-blue-500 text-blue-900'
            : 'bg-green-50 border-green-500 text-green-900'
        }`}>
          {isCameraPhoto ? <Camera className="w-5 h-5" /> : <FileText className="w-5 h-5" />}
          <div>
            <div className="font-bold">
              {isCameraPhoto ? '📷 Camera Photo' : '📄 Digital Document'}
            </div>
            <div className="text-sm opacity-75">
              Confidence: {(confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Scores breakdown */}
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="bg-white rounded p-2 border">
            <div className="text-gray-500 text-xs">EXIF</div>
            <div className="font-bold">{(classification.scores.exif * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-white rounded p-2 border">
            <div className="text-gray-500 text-xs">Visual</div>
            <div className="font-bold">{(classification.scores.visual * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-white rounded p-2 border">
            <div className="text-gray-500 text-xs">Final</div>
            <div className="font-bold">{(classification.scores.final * 100).toFixed(0)}%</div>
          </div>
        </div>

        {/* Indicators */}
        <details className="bg-white rounded-lg border p-3 text-sm">
          <summary className="cursor-pointer font-semibold text-gray-700 hover:text-gray-900">
            Detection Details
          </summary>
          <div className="mt-3 space-y-2">
            {/* EXIF Indicators */}
            {classification.indicators.exif.has_exif && (
              <div className="border-t pt-2">
                <div className="font-semibold text-xs text-gray-500 uppercase mb-2">EXIF Metadata</div>
                <div className="space-y-1 text-xs">
                  {classification.indicators.exif.make && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Camera: {classification.indicators.exif.make} {classification.indicators.exif.model}</span>
                    </div>
                  )}
                  {classification.indicators.exif.is_phone_camera && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Phone camera detected</span>
                    </div>
                  )}
                  {classification.indicators.exif.has_gps && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>GPS data present</span>
                    </div>
                  )}
                  {classification.indicators.exif.software && (
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-3 h-3 text-amber-600" />
                      <span>Software: {classification.indicators.exif.software}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Visual Indicators */}
            {Object.keys(classification.indicators.visual).length > 0 && (
              <div className="border-t pt-2">
                <div className="font-semibold text-xs text-gray-500 uppercase mb-2">Visual Analysis</div>
                <div className="space-y-1 text-xs">
                  {classification.indicators.visual.document_contour && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Document contour detected ({(classification.indicators.visual.contour_area_ratio! * 100).toFixed(0)}% of frame)</span>
                    </div>
                  )}
                  {classification.indicators.visual.perspective_distortion && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Perspective distortion found</span>
                    </div>
                  )}
                  {classification.indicators.visual.visible_background && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Background visible</span>
                    </div>
                  )}
                  {classification.indicators.visual.has_focus_variation && (
                    <div className="flex items-center gap-2">
                      <Check className="w-3 h-3 text-green-600" />
                      <span>Focus variation detected</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </details>
      </div>
    );
  };

  const renderProcessingPipeline = (processing: ProcessingMetadata | undefined) => {
    if (!processing) return null;

    const scanApplied = processing.scan?.applied;
    const scanSuccess = processing.scan?.scan_success;

    return (
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4 border-2 border-purple-200">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-5 h-5 text-purple-600" />
          <h3 className="font-bold text-purple-900">Processing Pipeline</h3>
        </div>

        <div className="space-y-2">
          {/* Step 1: Classification */}
          <div className="flex items-center gap-3 bg-white rounded-lg p-3 border">
            <div className="flex-shrink-0 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold">
              <Check className="w-5 h-5" />
            </div>
            <div className="flex-1">
              <div className="font-semibold text-sm">Document Classified</div>
              <div className="text-xs text-gray-600">
                {processing.classification?.classification === 'camera_photo' ? 'Camera photo detected' : 'Digital document detected'}
              </div>
            </div>
          </div>

          {/* Step 2: Perspective Correction */}
          <div className={`flex items-center gap-3 bg-white rounded-lg p-3 border ${
            scanApplied ? 'border-green-300' : 'border-gray-200'
          }`}>
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
              scanApplied
                ? scanSuccess ? 'bg-green-500' : 'bg-amber-500'
                : 'bg-gray-300'
            }`}>
              {scanApplied ? (scanSuccess ? <Check className="w-5 h-5" /> : '!') : <X className="w-5 h-5" />}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-sm">Perspective Correction</div>
              <div className="text-xs text-gray-600">
                {scanApplied
                  ? scanSuccess
                    ? 'Applied successfully'
                    : 'Failed, using original'
                  : 'Skipped (not needed)'
                }
              </div>
            </div>
          </div>

          {/* Step 3: Object Detection */}
          <div className="flex items-center gap-3 bg-white rounded-lg p-3 border border-green-300">
            <div className="flex-shrink-0 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold">
              <Check className="w-5 h-5" />
            </div>
            <div className="flex-1">
              <div className="font-semibold text-sm">Object Detection</div>
              <div className="text-xs text-gray-600">
                Stamps, signatures, and QR codes detected
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <Zap className="w-8 h-8 text-yellow-500" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            Intelligent Document Processor
          </h1>
        </div>
        <p className="text-gray-600">
          AI-powered classification • Automatic perspective correction • Smart object detection
        </p>
      </div>

      {/* Upload Area */}
      {!file && (
        <div className="border-4 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-purple-400 transition-colors">
          <input
            type="file"
            onChange={handleFileSelect}
            accept="image/*,.pdf"
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <p className="text-xl font-semibold text-gray-700 mb-2">
              Upload Document
            </p>
            <p className="text-gray-500">
              Images (JPG, PNG) or PDFs • Phone photos or digital documents
            </p>
          </label>
        </div>
      )}

      {/* File Preview & Controls */}
      {file && !result && (
        <div className="space-y-4">
          {/* Preview */}
          {previewUrl && (
            <div className="bg-white rounded-lg border-2 p-4">
              <img
                src={previewUrl}
                alt="Document preview"
                className="max-h-96 mx-auto rounded"
              />
            </div>
          )}

          {/* Advanced Options */}
          <div className="bg-gray-50 rounded-lg p-4 border">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-semibold text-gray-700 hover:text-gray-900"
            >
              <span>Advanced Options</span>
              <span className="text-xs">{showAdvanced ? '▼' : '▶'}</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={forceScan}
                    onChange={(e) => {
                      setForceScan(e.target.checked);
                      if (e.target.checked) setSkipScan(false);
                    }}
                    className="rounded"
                  />
                  <span>Force perspective correction (even if digital document)</span>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={skipScan}
                    onChange={(e) => {
                      setSkipScan(e.target.checked);
                      if (e.target.checked) setForceScan(false);
                    }}
                    className="rounded"
                  />
                  <span>Skip perspective correction (even if camera photo)</span>
                </label>
              </div>
            )}
          </div>

          {/* Process Button */}
          <div className="flex gap-3">
            <button
              onClick={processDocument}
              disabled={processingState.isProcessing}
              className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {processingState.isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>{processingState.stageMessage}</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Process Document</span>
                </>
              )}
            </button>
            <button
              onClick={reset}
              className="px-6 py-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50"
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Processing Info */}
          {(() => {
            const isSinglePage = !('document_type' in result);
            const processing = isSinglePage
              ? (result as ProcessDocumentResponse).processing
              : (result as MultiPageProcessDocumentResponse).processing;

            return (
              <div className="grid md:grid-cols-2 gap-6">
                {/* Classification */}
                <div>
                  <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                    <Camera className="w-5 h-5" />
                    AI Classification
                  </h3>
                  {renderClassificationBadge(processing?.classification)}
                </div>

                {/* Pipeline */}
                <div>
                  <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    Processing Steps
                  </h3>
                  {renderProcessingPipeline(processing)}
                </div>
              </div>
            );
          })()}

          {/* Detection Results */}
          {(() => {
            const isSinglePage = !('document_type' in result);

            if (isSinglePage) {
              const singleResult = result as ProcessDocumentResponse;
              return (
                <div>
                  <h3 className="font-bold text-lg mb-3">Detection Results</h3>
                  <ImageWithDetections
                    imageUrl={previewUrl || ''}
                    detectionResult={singleResult}
                  />
                  <div className="mt-4 bg-white rounded-lg border p-4">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-2xl font-bold text-red-600">{singleResult.summary.total_stamps}</div>
                        <div className="text-sm text-gray-600">Stamps</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-green-600">{singleResult.summary.total_signatures}</div>
                        <div className="text-sm text-gray-600">Signatures</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-blue-600">{singleResult.summary.total_qrs}</div>
                        <div className="text-sm text-gray-600">QR Codes</div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            } else {
              const multiResult = result as MultiPageProcessDocumentResponse;
              return (
                <div>
                  <h3 className="font-bold text-lg mb-3">
                    Multi-Page Results ({multiResult.total_pages} pages)
                  </h3>
                  <div className="space-y-4">
                    {multiResult.pages.map((page, idx) => (
                      <div key={idx} className="bg-white rounded-lg border-2 p-4">
                        <div className="font-semibold mb-3">Page {idx + 1}</div>
                        {page.page_image && (
                          <ImageWithDetections
                            imageUrl={page.page_image}
                            detectionResult={page}
                          />
                        )}
                      </div>
                    ))}
                    <div className="bg-white rounded-lg border p-4">
                      <div className="font-semibold mb-2">Total across all pages</div>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className="text-2xl font-bold text-red-600">{multiResult.summary.total_stamps}</div>
                          <div className="text-sm text-gray-600">Stamps</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-green-600">{multiResult.summary.total_signatures}</div>
                          <div className="text-sm text-gray-600">Signatures</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-blue-600">{multiResult.summary.total_qrs}</div>
                          <div className="text-sm text-gray-600">QR Codes</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            }
          })()}

          {/* New Upload Button */}
          <div className="text-center">
            <button
              onClick={reset}
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-8 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700"
            >
              Process Another Document
            </button>
          </div>
        </div>
      )}

      {/* Error Display */}
      {processingState.currentStage === 'error' && (
        <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-red-900">Processing Error</div>
            <div className="text-red-700 text-sm">{processingState.stageMessage}</div>
          </div>
        </div>
      )}
    </div>
  );
}
