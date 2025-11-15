export function HackathonFooter() {
  return (
    <footer className="py-8 animate-in fade-in duration-800" style={{ backgroundColor: 'rgba(31, 107, 255, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="animate-in slide-in-from-bottom duration-800 delay-200">
            <h3 className="text-2xl font-bold mb-4 cursor-pointer" style={{ color: 'rgba(255, 255, 255, 1)' }}>
              Team <span style={{ color: 'rgba(255, 255, 255, 1)' }}>Richards</span>
            </h3>
            <p style={{ color: 'rgba(255, 255, 255, 1)', opacity: 0.8 }}>
              Revolutionizing construction document compliance with Computer Vision AI
            </p>
          </div>

          {/* Project Info */}
          <div className="animate-in slide-in-from-bottom duration-800 delay-400">
            <h4 className="mb-4 transition-colors duration-200 cursor-pointer" 
              style={{ color: 'rgba(200, 220, 255, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
            >
              Key Features
            </h4>
            <ul className="space-y-2">
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
          <div className="animate-in slide-in-from-bottom duration-800 delay-600">
            <h4 className="mb-4 transition-colors duration-200 cursor-pointer" 
              style={{ color: 'rgba(200, 220, 255, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(200, 220, 255, 1)'}
            >
              Hackathon
            </h4>
            <div>
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

        <div className="mt-8 pt-8 border-t text-center transition-all duration-300 hover:border-opacity-80" style={{ borderColor: 'rgba(255, 255, 255, 1)' }}>
          <p className="transition-opacity duration-300 hover:opacity-90" style={{ color: 'rgba(255, 255, 255, 1)', opacity: 0.6 }}>
            Built with 🖤 Team Richards • <a href="https://github.com/aidynfatikh/richard-docs" target="_blank" rel="noopener noreferrer" className="hover:underline transition-all duration-300 font-semibold">View on GitHub - RichardDocs</a>
          </p>
        </div>
      </div>
    </footer>
  );
}
