export function HackathonFooter() {
  return (
    <footer className="py-6 sm:py-8 lg:py-10" style={{ backgroundColor: 'rgba(0, 23, 255, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {/* Brand */}
          <div>
            <h3 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 cursor-pointer" style={{ color: 'rgba(255, 255, 255, 1)' }}>
              Team <span style={{ color: 'rgba(255, 255, 255, 1)' }}>Richards</span>
            </h3>
            <p className="text-sm sm:text-base" style={{ color: 'rgba(255, 255, 255, 1)', opacity: 0.8 }}>
              Revolutionizing construction document compliance with Computer Vision AI
            </p>
          </div>

          {/* Project Info */}
          <div>
            <h4 className="text-base sm:text-lg mb-3 sm:mb-4 transition-colors duration-200 cursor-pointer" 
              style={{ color: 'rgba(200, 220, 255, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
            >
              Key Features
            </h4>
            <ul className="space-y-1.5 sm:space-y-2 text-sm sm:text-base">
              <li className="transition-colors duration-200 cursor-pointer" 
                style={{ color: 'rgba(255, 255, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              >
                Signature Detection
              </li>
              <li className="transition-colors duration-200 cursor-pointer" 
                style={{ color: 'rgba(255, 255, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              >
                Stamp Recognition
              </li>
              <li className="transition-colors duration-200 cursor-pointer" 
                style={{ color: 'rgba(255, 255, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              >
                QR Code Extraction
              </li>
              <li className="transition-colors duration-200 cursor-pointer" 
                style={{ color: 'rgba(255, 255, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              >
                Automated Compliance
              </li>
            </ul>
          </div>

          {/* Event Info */}
          <div>
            <h4 className="text-base sm:text-lg mb-3 sm:mb-4 transition-colors duration-200 cursor-pointer" 
              style={{ color: 'rgba(200, 220, 255, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
            >
              Hackathon
            </h4>
            <div className="text-sm sm:text-base">
              <p className="mb-2 cursor-pointer transition-colors duration-200" 
                style={{ color: 'rgba(255, 255, 255, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              >
                Armeta AI 2025
              </p>
            </div>
          </div>
        </div>

        <div className="mt-6 sm:mt-8 pt-6 sm:pt-8 border-t text-center transition-all duration-300 hover:border-opacity-80" style={{ borderColor: 'rgba(255, 255, 255, 1)' }}>
          <p className="text-xs sm:text-sm transition-opacity duration-300 hover:opacity-90" style={{ color: 'rgba(255, 255, 255, 1)', opacity: 0.6 }}>
            Team Richards • <a href="https://github.com/aidynfatikh/richard-docs" target="_blank" rel="noopener noreferrer" className="hover:underline transition-all duration-300 font-semibold">View on GitHub - RichardDocs</a>
          </p>
        </div>
      </div>
    </footer>
  );
}
