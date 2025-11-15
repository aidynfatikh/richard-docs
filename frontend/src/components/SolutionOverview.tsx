export function SolutionOverview() {
  return (
    <section id="solution" className="py-12 sm:py-16 md:py-20 lg:py-24" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 items-start">
          {/* Left Content */}
          <div className="animate-in slide-in-from-left duration-800">
            <h2 id="approach" className="text-3xl sm:text-4xl md:text-5xl font-bold mb-6 sm:mb-8 leading-tight cursor-pointer" 
              style={{ color: 'rgba(31, 107, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(51, 127, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
            >
              Our Approach
            </h2>
            
            <div className="space-y-4 sm:space-y-6 text-base sm:text-lg">
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                We developed an end-to-end Computer Vision pipeline that analyzes construction documents
                and automatically identifies critical compliance markers with precision and speed.
              </p>
              
              <p className="hidden md:block cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Our system uses advanced object detection models trained on architectural documents
                to ensure nothing slips through the cracks during compliance reviews.
              </p>
              
              <p className="hidden md:block cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                From upload to verification — powered by state-of-the-art AI that transforms manual inspection into automated intelligence.
              </p>
            </div>
          </div>

          {/* Right Content - Key Features */}
          <div className="space-y-4 sm:space-y-6">
            <div className="p-4 sm:p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-200">
              <h3 className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 transition-all duration-200"
                style={{ color: 'rgba(31, 107, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(51, 127, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
              >
                Signature Detection
              </h3>
              <p className="text-sm sm:text-base cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Automatically identifies handwritten and digital signatures on technical drawings and official documents
              </p>
            </div>

            <div className="p-4 sm:p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-400">
              <h3 className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 transition-all duration-200"
                style={{ color: 'rgba(31, 107, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(51, 127, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
              >
                Stamp Recognition
              </h3>
              <p className="text-sm sm:text-base cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Detects official stamps and seals from regulatory bodies ensuring document authenticity
              </p>
            </div>

            <div className="p-4 sm:p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-600">
              <h3 className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 transition-all duration-200"
                style={{ color: 'rgba(31, 107, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(51, 127, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
              >
                QR Code Extraction
              </h3>
              <p className="text-sm sm:text-base cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(31, 107, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Locates and decodes QR codes embedded in plans for instant access to digital verification systems
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
