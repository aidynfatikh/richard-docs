import { useLocation, useNavigate } from 'react-router-dom';
import type { DetectionResponse, BatchDetectionResponse, Detection } from '../types/api.types';
import { HackathonHeader } from '../components/HackathonHeader';
import { HackathonFooter } from '../components/HackathonFooter';
import { ImageWithDetections } from '../components/ImageWithDetections';
import { detectDevice } from '../utils/deviceDetection';

interface SolutionPageState {
  results?: Array<{ fileName: string; fileObject: File; data: DetectionResponse }>;
  batchResults?: BatchDetectionResponse;
}

export function SolutionPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as SolutionPageState | null;
  const isMobile = detectDevice().isMobile;

  // Handle batch results from mobile scanner
  const isBatchMode = state?.batchResults !== undefined;
  const results = isBatchMode
    ? state.batchResults!.results.filter(r => r.success)
    : state?.results || [];

  // Redirect if no results
  if (!state || (results.length === 0)) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4" style={{ color: 'rgba(247, 247, 248, 1)' }}>
            No Results Found
          </h1>
          <p className="text-lg mb-6" style={{ color: 'rgba(153, 153, 153, 1)' }}>
            Please analyze documents first to see results.
          </p>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:brightness-90"
            style={{
              backgroundColor: 'rgba(0, 23, 255, 1)',
              color: 'rgba(255, 255, 255, 1)'
            }}
          >
            Go Back to Analyzer
          </button>
        </div>
      </main>
    );
  }

  // Calculate overall statistics
  const totalDetections = isBatchMode
    ? state.batchResults!.summary.total_detections
    : results.reduce((sum, r) => sum + (r as any).data.summary.total_detections, 0);

  const totalStamps = isBatchMode
    ? state.batchResults!.summary.total_stamps
    : results.reduce((sum, r) => sum + (r as any).data.summary.total_stamps, 0);

  const totalSignatures = isBatchMode
    ? state.batchResults!.summary.total_signatures
    : results.reduce((sum, r) => sum + (r as any).data.summary.total_signatures, 0);

  const totalQRs = isBatchMode
    ? state.batchResults!.summary.total_qrs
    : results.reduce((sum, r) => sum + (r as any).data.summary.total_qrs, 0);

  const avgProcessingTime = isBatchMode
    ? state.batchResults!.meta.total_processing_time_ms / results.length
    : results.reduce((sum, r) => sum + (r as any).data.meta.total_processing_time_ms, 0) / results.length;

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      {!isMobile && <HackathonHeader />}
      <main className="pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-12">

            <h1 className="text-4xl sm:text-5xl font-bold mb-4" style={{ color: 'rgba(0, 23, 255, 1)' }}>
              Analysis Results
            </h1>
            <p className="text-lg" style={{ color: 'rgba(153, 153, 153, 1)' }}>
              Detailed detection report for {results.length} document{results.length > 1 ? 's' : ''}
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="mb-6 px-6 py-2 rounded-xl font-semibold transition-all duration-200 hover:brightness-90"
            style={{
              backgroundColor: 'rgba(17, 17, 17, 1)',
              color: 'rgba(247, 247, 248, 1)',
              border: '1px solid rgba(153, 153, 153, 0.3)'
            }}
          >
            ← Analyze More Documents
          </button>

          {/* Overall Statistics */}
          <div className="mb-12 p-6 rounded-2xl" style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}>
            <h2 className="text-2xl font-bold mb-6 text-center" style={{ color: 'rgba(247, 247, 248, 1)' }}>
              Overall Statistics
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6">
              <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                <div className="text-4xl sm:text-5xl font-bold mb-2" style={{ color: 'rgba(0, 23, 255, 1)' }}>
                  {totalDetections}
                </div>
                <div className="text-sm font-medium" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                  Total Detections
                </div>
              </div>
              <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                <div className="text-4xl sm:text-5xl font-bold mb-2" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                  {totalStamps}
                </div>
                <div className="text-sm font-medium" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                  Stamps
                </div>
              </div>
              <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                <div className="text-4xl sm:text-5xl font-bold mb-2" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                  {totalSignatures}
                </div>
                <div className="text-sm font-medium" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                  Signatures
                </div>
              </div>
              <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                <div className="text-4xl sm:text-5xl font-bold mb-2" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                  {totalQRs}
                </div>
                <div className="text-sm font-medium" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                  QR Codes
                </div>
              </div>
            </div>
            <div className="mt-6 text-center text-sm" style={{ color: 'rgba(153, 153, 153, 1)' }}>
              Average processing time: <span className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                {avgProcessingTime.toFixed(0)}ms
              </span> per document
            </div>
          </div>

          {/* Individual Document Results */}
          <div>
            <h2 className="text-2xl font-bold mb-6" style={{ color: 'rgba(247, 247, 248, 1)' }}>
              Document Details
            </h2>
            <div className="space-y-6">
              {results.map((result: any, index) => {
                const fileName = isBatchMode ? result.filename : result.fileName;
                const fileObject = isBatchMode ? null : result.fileObject;
                const data = isBatchMode ? result : result.data;

                return (
                <div
                  key={index}
                  className="p-6 rounded-2xl"
                  style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}
                >
                  {/* Document Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V9L13 2Z" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M13 2V9H20" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      <h3 className="font-semibold text-lg truncate" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                        {fileName}
                      </h3>
                    </div>
                    <span className="text-xs px-3 py-1 rounded-full font-medium" style={{ 
                      backgroundColor: 'rgba(0, 23, 255, 0.1)', 
                      color: 'rgba(0, 23, 255, 1)',
                      border: '1px solid rgba(0, 23, 255, 0.3)'
                    }}>
                      Document #{index + 1}
                    </span>
                  </div>

                  {/* Image Visualization */}
                  {fileObject && (
                    <div className="mb-6 p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                      <ImageWithDetections
                        imageFile={fileObject}
                        stamps={data.stamps}
                        signatures={data.signatures}
                        qrs={data.qrs}
                        imageSize={data.image_size}
                      />
                    </div>
                  )}
                  {!fileObject && isBatchMode && (
                    <div className="mb-6 p-4 rounded-xl text-center" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)', color: 'rgba(153, 153, 153, 1)' }}>
                      <p className="text-sm">Scanned document #{index + 1}</p>
                      <p className="text-xs mt-2">Image visualization not available in batch mode</p>
                    </div>
                  )}

                  {/* Detection Summary */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                      <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(0, 23, 255, 1)' }}>
                        {data.summary.total_detections}
                      </div>
                      <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Total</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                      <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                        {data.summary.total_stamps}
                      </div>
                      <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Stamps</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                      <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                        {data.summary.total_signatures}
                      </div>
                      <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Signatures</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                      <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                        {data.summary.total_qrs}
                      </div>
                      <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>QR Codes</div>
                    </div>
                  </div>

                  {/* Detailed Detections */}
                  {(data.stamps.length > 0 || data.signatures.length > 0 || data.qrs.length > 0) && (
                    <div className="space-y-4">
                      {/* Stamps */}
                      {data.stamps.length > 0 && (
                        <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                          <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(239, 68, 68, 1)' }}></span>
                            Stamps ({data.stamps.length})
                          </h4>
                          <div className="space-y-2">
                            {data.stamps.map((stamp: Detection, i: number) => (
                              <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                  Position: ({stamp.bbox[0].toFixed(0)}, {stamp.bbox[1].toFixed(0)}) - ({stamp.bbox[2].toFixed(0)}, {stamp.bbox[3].toFixed(0)})
                                </span>
                                <span className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                                  {(stamp.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Signatures */}
                      {data.signatures.length > 0 && (
                        <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                          <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(34, 197, 94, 1)' }}></span>
                            Signatures ({data.signatures.length})
                          </h4>
                          <div className="space-y-2">
                            {data.signatures.map((sig: Detection, i: number) => (
                              <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                  Position: ({sig.bbox[0].toFixed(0)}, {sig.bbox[1].toFixed(0)}) - ({sig.bbox[2].toFixed(0)}, {sig.bbox[3].toFixed(0)})
                                </span>
                                <span className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                                  {(sig.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* QR Codes */}
                      {data.qrs.length > 0 && (
                        <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                          <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(59, 130, 246, 1)' }}></span>
                            QR Codes ({data.qrs.length})
                          </h4>
                          <div className="space-y-2">
                            {data.qrs.map((qr: Detection, i: number) => (
                              <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                  Position: ({qr.bbox[0].toFixed(0)}, {qr.bbox[1].toFixed(0)}) - ({qr.bbox[2].toFixed(0)}, {qr.bbox[3].toFixed(0)})
                                </span>
                                <span className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                                  {(qr.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="mt-4 pt-4 border-t grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs" style={{ borderColor: 'rgba(153, 153, 153, 0.2)' }}>
                    <div>
                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>Processing Time:</span>
                      <div className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                        {data.meta.total_processing_time_ms.toFixed(0)}ms
                      </div>
                    </div>
                    <div>
                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>Confidence:</span>
                      <div className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                        {(data.meta.confidence_threshold * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>Model:</span>
                      <div className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                        {data.meta.model_version}
                      </div>
                    </div>
                  </div>
                </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>

      {!isMobile && <HackathonFooter />}
    </div>
  );
}
