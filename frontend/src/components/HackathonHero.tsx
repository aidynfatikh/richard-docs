import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { validateFile, formatFileSize, getAcceptAttribute } from '../utils/fileValidation';
import { apiService } from '../services/api.service';
import type { DetectionResponse } from '../types/api.types';
import { detectDevice } from '../utils/deviceDetection';

export function HackathonHero() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = detectDevice().isMobile;

  // Auto-clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError('');
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleFileSelect = (selectedFiles: FileList | null) => {
    setError('');
    
    if (!selectedFiles || selectedFiles.length === 0) {
      return;
    }

    const fileArray = Array.from(selectedFiles);
    
    // Validate each file before adding
    const validFiles: File[] = [];
    for (const file of fileArray) {
      const validation = validateFile(file);
      if (!validation.isValid) {
        setError(validation.error || 'Invalid file');
        return;
      }
      validFiles.push(file);
    }
    
    setFiles(prev => [...prev, ...validFiles]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearAllFiles = () => {
    setFiles([]);
  };

  const handleInspect = async () => {
    setError('');
    
    if (files.length === 0) {
      setError('Please upload at least one document for inspection');
      return;
    }

    setIsProcessing(true);
    setProgress({ current: 0, total: files.length });

    try {
      // Process files with progress tracking
      const responses = await apiService.detectMultipleDocuments(
        files,
        0.25, // confidence threshold
        (current, total) => {
          setProgress({ current, total });
        }
      );

      // Filter successful results and expand multi-page PDFs
      const successfulResults: Array<{ 
        fileName: string; 
        fileObject: File | null; 
        data: DetectionResponse;
        pages?: DetectionResponse[]; // For multi-page PDFs
      }> = [];
      const errors: string[] = [];

      responses.forEach(({ file, result }, index) => {
        if (result.success && result.data) {
          // Check if it's a multi-page PDF
          if ('document_type' in result.data) {
            const multiPageData = result.data as any;
            console.log(`Multi-page PDF detected for ${file}: ${multiPageData.total_pages} pages`);
            
            // Calculate aggregate summary for the whole PDF
            const aggregateSummary = {
              total_detections: multiPageData.summary.total_detections,
              total_stamps: multiPageData.summary.total_stamps,
              total_signatures: multiPageData.summary.total_signatures,
              total_qrs: multiPageData.summary.total_qrs,
            };
            
            // Store all pages data
            successfulResults.push({
              fileName: file,
              fileObject: null,
              data: {
                ...multiPageData.pages[0], // Use first page's structure
                summary: aggregateSummary,
                meta: multiPageData.meta
              },
              pages: multiPageData.pages // Store all pages
            });
          } else {
            // Single image file
            console.log(`Processing single image for ${file}`);
            successfulResults.push({ 
              fileName: file, 
              fileObject: files[index], 
              data: result.data 
            });
          }
        } else {
          errors.push(`${file}: ${result.error?.detail || 'Unknown error'}`);
        }
      });

      if (errors.length > 0) {
        setError(`Some files failed: ${errors.join(', ')}`);
      } else if (successfulResults.length === 0) {
        setError('All files failed to process');
      } else {
        // Navigate to solution page with results
        navigate('/solution', { state: { results: successfulResults } });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process documents');
    } finally {
      setIsProcessing(false);
      setProgress({ current: 0, total: 0 });
    }
  };

  return (
    <section className="min-h-screen flex items-center pt-16 pb-8 sm:pt-20 sm:pb-12 md:py-20 lg:py-24 xl:py-28" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div className="text-center">
          {/* Main Heading */}
          <h1 className="text-4xl sm:text-5xl md:text-5xl lg:text-6xl xl:text-6xl font-bold mb-6 sm:mb-8 md:mb-10 leading-tight cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(0, 23, 255, 1)' }}
          >
            <span className="inline-block">
              Digital Inspector
            </span><br />
            <span className="inline-block">
              AI Document Analyzer
            </span>
          </h1>

          {/* Subheading */}
          <h2 className="text-lg sm:text-xl md:text-xl lg:text-2xl xl:text-2xl font-semibold mb-12 sm:mb-6 md:mb-8 lg:mb-10 cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(247, 247, 248, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(0, 23, 255, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
          >
            Automated detection of signatures, stamps & QR codes
          </h2>

          {/* Mobile Camera Scan Button */}
          {isMobile && (
            <div className="mb-6">
              <button
                onClick={() => navigate('/scan')}
                className="px-8 py-4 rounded-xl text-lg font-bold transition-all duration-200 hover:brightness-90 shadow-lg flex items-center justify-center gap-3"
                style={{
                  backgroundColor: 'rgba(0, 23, 255, 1)',
                  color: 'rgba(255, 255, 255, 1)'
                }}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Scan with Camera
              </button>
            </div>
          )}

          {/* Or Divider for mobile */}
          {isMobile && (
            <div className="flex items-center gap-4 mb-6 w-full max-w-md">
              <div className="flex-1 h-px" style={{ backgroundColor: 'rgba(153, 153, 153, 0.3)' }}></div>
              <span className="text-sm" style={{ color: 'rgba(153, 153, 153, 1)' }}>or upload files</span>
              <div className="flex-1 h-px" style={{ backgroundColor: 'rgba(153, 153, 153, 0.3)' }}></div>
            </div>
          )}

          {/* File Upload Area */}
          <div className="flex flex-col items-center max-w-4xl mx-auto w-full">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={getAcceptAttribute()}
              onChange={handleFileInputChange}
              className="hidden"
              disabled={isProcessing}
            />
            
            {/* Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className="w-full border-2 border-dashed rounded-2xl p-6 sm:p-12 text-center transition-all duration-200 cursor-pointer"
              style={{
                borderColor: isDragging ? 'rgba(0, 23, 255, 1)' : error ? 'rgba(239, 68, 68, 1)' : 'rgba(153, 153, 153, 0.3)',
                backgroundColor: isDragging ? 'rgba(31, 107, 255, 0.05)' : 'rgba(17, 17, 17, 0.5)'
              }}
              onClick={handleButtonClick}
            >
              <div className="flex flex-col items-center gap-4">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M17 8L12 3L7 8" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 3V15" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <div>
                  <p className="text-lg font-semibold mb-2" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                    {isProcessing ? 'Processing...' : isDragging ? 'Drop files here' : 'Drag & drop files here'}
                  </p>
                  <p className="text-sm" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                    {isProcessing 
                      ? `Analyzing ${progress.current} of ${progress.total} files...`
                      : 'Images (JPG, PNG, WEBP, TIFF, BMP) and PDFs supported • Max 50MB per file'
                    }
                  </p>
                </div>
              </div>
            </div>

            {/* Selected Files and Analyze Button */}
            {files.length > 0 && (
              <div className="w-full mt-6">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3">
                  <div className="flex items-center gap-3">
                    <p className="text-sm font-semibold" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                      Selected files ({files.length})
                    </p>
                    <button
                      onClick={clearAllFiles}
                      className="text-xs font-medium px-3 py-1 rounded-lg transition-all duration-200 hover:brightness-90"
                      style={{
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        color: 'rgba(239, 68, 68, 1)',
                        border: '1px solid rgba(239, 68, 68, 0.3)'
                      }}
                    >
                      Clear All
                    </button>
                  </div>
                  <button
                    onClick={handleInspect}
                    disabled={isProcessing}
                    className="w-full sm:w-auto px-6 py-2 rounded-xl text-sm font-semibold transition-all duration-200 hover:brightness-90 shadow-lg flex items-center justify-center gap-2 mr-3 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      backgroundColor: 'rgba(0, 23, 255, 1)',
                      color: 'rgba(255, 255, 255, 1)'
                    }}
                  >
                    {isProcessing ? (
                      <>
                        <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </>
                    ) : (
                      'Analyze Documents'
                    )}
                  </button>
                </div>
                <div className="max-h-40 overflow-y-auto space-y-2" style={{
                  scrollbarWidth: 'thin',
                  scrollbarColor: 'rgba(0, 0, 0, 0.5) transparent'
                }}>
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-lg gap-2"
                      style={{ backgroundColor: 'rgba(17, 17, 17, 1)' }}
                    >
                      <div className="flex items-center gap-2 sm:gap-3 flex-1 min-w-0">
                        <svg className="flex-shrink-0" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M13 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V9L13 2Z" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M13 2V9H20" stroke="rgba(0, 23, 255, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span className="text-xs sm:text-sm truncate" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                          {file.name}
                        </span>
                        <span className="text-xs flex-shrink-0 hidden sm:inline" style={{ color: 'rgba(153, 153, 153, 1)' }}>
                          ({formatFileSize(file.size)})
                        </span>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(index);
                        }}
                        className="flex-shrink-0 p-1 rounded hover:bg-red-500/20 transition-colors"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M18 6L6 18M6 6L18 18" stroke="rgba(239, 68, 68, 1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-4 w-full">
                <p className="text-sm font-medium text-center" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                  {error}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
