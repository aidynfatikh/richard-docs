import { useLocation, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import type { DetectionResponse, BatchDetectionResponse, Detection } from '../types/api.types';
import { ImageWithDetections } from '../components/ImageWithDetections';

interface SolutionPageState {
  results?: Array<{ 
    fileName: string; 
    fileObject: File | null; 
    data: DetectionResponse;
    pages?: DetectionResponse[]; // For multi-page PDFs
  }>;
  batchResults?: BatchDetectionResponse;
}

export function SolutionPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as SolutionPageState | null;
  const [selectedDocIndex, setSelectedDocIndex] = useState(0);

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

  const selectedResult = !isBatchMode && results.length > 0 ? results[selectedDocIndex] as any : null;

  return (
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

          {/* Documents Selector */}
          {!isBatchMode && results.length > 1 && (
            <div className="mb-12">
              <h2 className="text-2xl font-bold mb-6" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                Documents
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {results.map((result: any, index) => (
                  <button
                    key={index}
                    onClick={() => setSelectedDocIndex(index)}
                    className="p-4 rounded-xl text-left transition-all duration-200 hover:brightness-110"
                    style={{
                      backgroundColor: selectedDocIndex === index ? 'rgba(0, 23, 255, 0.2)' : 'rgba(17, 17, 17, 0.5)',
                      border: selectedDocIndex === index ? '2px solid rgba(0, 23, 255, 1)' : '1px solid rgba(153, 153, 153, 0.2)',
                    }}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V9L13 2Z" stroke={selectedDocIndex === index ? 'rgba(0, 23, 255, 1)' : 'rgba(153, 153, 153, 1)'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M13 2V9H20" stroke={selectedDocIndex === index ? 'rgba(0, 23, 255, 1)' : 'rgba(153, 153, 153, 1)'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      <span className="font-semibold text-sm truncate" style={{ color: selectedDocIndex === index ? 'rgba(0, 23, 255, 1)' : 'rgba(247, 247, 248, 1)' }}>
                        {result.fileName}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                      <span>{result.data.summary.total_detections} detections</span>
                      <span>•</span>
                      <span>{(result.data.meta?.total_processing_time_ms || 0).toFixed(0)}ms</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected Document Details */}
          {!isBatchMode && selectedResult && (
            <div>
              <h2 className="text-2xl font-bold mb-6" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                Document Details
              </h2>
              <div>
                {(() => {
                const result = selectedResult;
                const fileName = result.fileName;
                const pages = result.pages; // Pages array for PDFs
                const isMulitPagePDF = pages && pages.length > 0;

                return (
                <div className="space-y-6">
                  {/* Document Header */}
                  <div className="p-6 rounded-2xl" style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}>
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
                        Document #{selectedDocIndex + 1}
                      </span>
                    </div>

                    {/* Overall Document Summary */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
                      <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                        <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(0, 23, 255, 1)' }}>
                          {result.data.summary.total_detections}
                        </div>
                        <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Total</div>
                      </div>
                      <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                        <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                          {result.data.summary.total_stamps}
                        </div>
                        <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Stamps</div>
                      </div>
                      <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                        <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                          {result.data.summary.total_signatures}
                        </div>
                        <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Signatures</div>
                      </div>
                      <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                        <div className="text-3xl font-bold mb-1" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                          {result.data.summary.total_qrs}
                        </div>
                        <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>QR Codes</div>
                      </div>
                    </div>

                    {isMulitPagePDF && (
                      <div className="text-xs text-center" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                        {pages.length} page{pages.length > 1 ? 's' : ''}
                      </div>
                    )}
                  </div>

                  {/* Pages Display */}
                  {isMulitPagePDF ? (
                    // Multi-page PDF - show all pages one after another
                    pages.map((pageData: DetectionResponse, pageIndex: number) => (
                      <div key={pageIndex} className="p-6 rounded-2xl mb-6" style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}>
                        <h4 className="text-lg font-semibold mb-4" style={{ color: 'rgba(0, 23, 255, 1)' }}>
                          Page {pageIndex + 1}
                        </h4>
                        
                        {/* Page Image */}
                        {pageData.page_image && (
                          <div className="mb-6 p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                            <ImageWithDetections
                              imageFile={null}
                              stamps={pageData.stamps}
                              signatures={pageData.signatures}
                              qrs={pageData.qrs}
                              imageSize={pageData.image_size}
                              base64Image={pageData.page_image}
                            />
                          </div>
                        )}

                        {/* Page Summary */}
                        <div className="grid grid-cols-4 gap-3 mb-4">
                          <div className="text-center p-2 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                            <div className="text-2xl font-bold mb-1" style={{ color: 'rgba(0, 23, 255, 1)' }}>
                              {pageData.summary.total_detections}
                            </div>
                            <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Total</div>
                          </div>
                          <div className="text-center p-2 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                            <div className="text-2xl font-bold mb-1" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                              {pageData.summary.total_stamps}
                            </div>
                            <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Stamps</div>
                          </div>
                          <div className="text-center p-2 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                            <div className="text-2xl font-bold mb-1" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                              {pageData.summary.total_signatures}
                            </div>
                            <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>Signatures</div>
                          </div>
                          <div className="text-center p-2 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                            <div className="text-2xl font-bold mb-1" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                              {pageData.summary.total_qrs}
                            </div>
                            <div className="text-xs" style={{ color: 'rgba(153, 153, 153, 1)' }}>QR Codes</div>
                          </div>
                        </div>

                        {/* Page Detailed Detections */}
                        {(pageData.stamps.length > 0 || pageData.signatures.length > 0 || pageData.qrs.length > 0) && (
                          <div className="space-y-3 mb-4">
                            {/* Stamps */}
                            {pageData.stamps.length > 0 && (
                              <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <h5 className="font-semibold mb-2 flex items-center gap-2 text-sm" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(239, 68, 68, 1)' }}></span>
                                  Stamps ({pageData.stamps.length})
                                </h5>
                                <div className="space-y-1">
                                  {pageData.stamps.map((stamp, i: number) => (
                                    <div key={i} className="flex items-center justify-between text-xs p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                        ({stamp.bbox[0].toFixed(0)}, {stamp.bbox[1].toFixed(0)}) - ({stamp.bbox[2].toFixed(0)}, {stamp.bbox[3].toFixed(0)})
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
                            {pageData.signatures.length > 0 && (
                              <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <h5 className="font-semibold mb-2 flex items-center gap-2 text-sm" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(34, 197, 94, 1)' }}></span>
                                  Signatures ({pageData.signatures.length})
                                </h5>
                                <div className="space-y-1">
                                  {pageData.signatures.map((sig, i: number) => (
                                    <div key={i} className="flex items-center justify-between text-xs p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                        ({sig.bbox[0].toFixed(0)}, {sig.bbox[1].toFixed(0)}) - ({sig.bbox[2].toFixed(0)}, {sig.bbox[3].toFixed(0)})
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
                            {pageData.qrs.length > 0 && (
                              <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                <h5 className="font-semibold mb-2 flex items-center gap-2 text-sm" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(59, 130, 246, 1)' }}></span>
                                  QR Codes ({pageData.qrs.length})
                                </h5>
                                <div className="space-y-1">
                                  {pageData.qrs.map((qr, i: number) => (
                                    <div key={i} className="flex items-center justify-between text-xs p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                      <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                        ({qr.bbox[0].toFixed(0)}, {qr.bbox[1].toFixed(0)}) - ({qr.bbox[2].toFixed(0)}, {qr.bbox[3].toFixed(0)})
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

                        {/* Page Processing Time */}
                        <div className="text-xs text-center" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                          Processing Time: {(pageData.meta.page_processing_time_ms || 0).toFixed(0)}ms
                        </div>
                      </div>
                    ))
                  ) : (
                    // Single image file
                    <div className="p-6 rounded-2xl" style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}>
                      {/* Image Visualization */}
                      {(result.fileObject || result.data.page_image || (result.data as any).transformed_image) && (
                        <div className="mb-6 p-4 rounded-xl" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                          <ImageWithDetections
                            imageFile={result.fileObject}
                            stamps={result.data.stamps}
                            signatures={result.data.signatures}
                            qrs={result.data.qrs}
                            imageSize={result.data.image_size}
                            base64Image={(result.data as any).transformed_image || result.data.page_image}
                          />
                        </div>
                      )}

                      {/* Detailed Detections for Single Image */}
                      {(result.data.stamps.length > 0 || result.data.signatures.length > 0 || result.data.qrs.length > 0) && (
                        <div className="space-y-4">
                          {/* Stamps */}
                          {result.data.stamps.length > 0 && (
                            <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                              <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(239, 68, 68, 1)' }}></span>
                                Stamps ({result.data.stamps.length})
                              </h4>
                              <div className="space-y-2">
                                {result.data.stamps.map((stamp: Detection, i: number) => (
                                  <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                    <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                      ({stamp.bbox[0].toFixed(0)}, {stamp.bbox[1].toFixed(0)}) - ({stamp.bbox[2].toFixed(0)}, {stamp.bbox[3].toFixed(0)})
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
                          {result.data.signatures.length > 0 && (
                            <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                              <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(34, 197, 94, 1)' }}>
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(34, 197, 94, 1)' }}></span>
                                Signatures ({result.data.signatures.length})
                              </h4>
                              <div className="space-y-2">
                                {result.data.signatures.map((sig: Detection, i: number) => (
                                  <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                    <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                      ({sig.bbox[0].toFixed(0)}, {sig.bbox[1].toFixed(0)}) - ({sig.bbox[2].toFixed(0)}, {sig.bbox[3].toFixed(0)})
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
                          {result.data.qrs.length > 0 && (
                            <div className="p-4 rounded-lg" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                              <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(59, 130, 246, 1)' }}>
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'rgba(59, 130, 246, 1)' }}></span>
                                QR Codes ({result.data.qrs.length})
                              </h4>
                              <div className="space-y-2">
                                {result.data.qrs.map((qr: Detection, i: number) => (
                                  <div key={i} className="flex items-center justify-between text-sm p-2 rounded" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                                    <span style={{ color: 'rgba(153, 153, 153, 1)' }}>
                                      ({qr.bbox[0].toFixed(0)}, {qr.bbox[1].toFixed(0)}) - ({qr.bbox[2].toFixed(0)}, {qr.bbox[3].toFixed(0)})
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
                      <div className="mt-4 pt-4 border-t text-xs text-center" style={{ borderColor: 'rgba(153, 153, 153, 0.2)', color: 'rgba(153, 153, 153, 1)' }}>
                        Processing Time: {(result.data.meta.total_processing_time_ms || result.data.meta.page_processing_time_ms || 0).toFixed(0)}ms
                      </div>
                    </div>
                  )}
                </div>
                );
                })()}
              </div>
            </div>
          )}

          {/* Batch Mode Results */}
          {isBatchMode && (
            <div>
              <h2 className="text-2xl font-bold mb-6" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                Document Details
              </h2>
              <div className="space-y-6">
                {results.map((result: any, index) => {
                  const fileName = result.filename;
                  const data = result;

                  return (
                  <div
                    key={index}
                    className="p-6 rounded-2xl"
                    style={{ backgroundColor: 'rgba(17, 17, 17, 0.5)', border: '1px solid rgba(153, 153, 153, 0.2)' }}
                  >
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

                    <div className="mb-6 p-4 rounded-xl text-center" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)', color: 'rgba(153, 153, 153, 1)' }}>
                      <p className="text-sm">Scanned document #{index + 1}</p>
                      <p className="text-xs mt-2">Image visualization not available in batch mode</p>
                    </div>

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

                    <div className="mt-4 pt-4 border-t text-xs" style={{ borderColor: 'rgba(153, 153, 153, 0.2)' }}>
                      <div className="font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                        Processing Time: {(data.meta.total_processing_time_ms || data.meta.page_processing_time_ms || 0).toFixed(0)}ms
                      </div>
                    </div>
                  </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </main>
  );
}
