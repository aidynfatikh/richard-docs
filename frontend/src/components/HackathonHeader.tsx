import { useState, useEffect } from 'react';

export function HackathonHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleNavigation = (section: string) => {
    setIsMobileMenuOpen(false);
    const element = document.getElementById(section);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleLogoClick = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    let ticking = false;
    
    const handleScroll = () => {
      const isScrolled = window.scrollY > 100;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    const throttledScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', throttledScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', throttledScroll);
    };
  }, [scrolled]);

  return (
    <div className="sticky top-0 z-[1100] w-full" style={{
      paddingTop: window.innerWidth >= 768 && scrolled ? '0.5rem' : '0',
      paddingBottom: window.innerWidth >= 768 && scrolled ? '0.5rem' : '0',
      transition: 'padding 300ms'
    }}>
      <div className="w-full" style={{ 
        borderBottom: window.innerWidth >= 768 && scrolled ? 'none' : '1px solid rgba(153, 153, 153, 0.2)',
        transition: 'border 300ms'
      }}>
        <div className="w-full grid place-items-center">
          <div className="w-full px-4 sm:px-6 lg:px-8" style={{
            maxWidth: window.innerWidth >= 768 && scrolled ? '56rem' : '80rem',
            transition: 'max-width 300ms'
          }}>
            <header 
              className="backdrop-blur-md"
              style={{ 
                padding: window.innerWidth >= 768 && scrolled ? '0.5rem 1.5rem' : '1rem 1.5rem',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                borderRadius: window.innerWidth >= 768 && scrolled ? '9999px' : '0',
                boxShadow: window.innerWidth >= 768 && scrolled ? '0 10px 15px -3px rgb(0 0 0 / 0.1)' : 'none',
                border: window.innerWidth >= 768 && scrolled ? '1px solid rgba(153, 153, 153, 0.2)' : '1px solid transparent',
                transition: 'all 300ms ease-out'
              }}
            >
            <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex-shrink-0">
            <button onClick={handleLogoClick} className="flex items-center gap-3 group cursor-pointer bg-transparent border-none p-0">
              <img 
                src="/icon.png" 
                alt="SexyAldarKose" 
                className="object-contain h-10 w-10"
                style={{
                  height: window.innerWidth >= 768 && scrolled ? '2rem' : '2.5rem',
                  width: window.innerWidth >= 768 && scrolled ? '2rem' : '2.5rem',
                  transition: 'all 300ms'
                }}
              />
              <h1 className="font-bold text-2xl"
                style={{ 
                  color: 'rgba(247, 247, 248, 1)',
                  fontSize: window.innerWidth >= 768 && scrolled ? '1.25rem' : '1.5rem',
                  transition: 'font-size 300ms'
                }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(0, 23, 255, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                RichardDocs
              </h1>
            </button>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center" style={{
            gap: window.innerWidth >= 768 && scrolled ? '0.25rem' : '0.5rem',
            transition: 'gap 300ms'
          }}>
            <button 
              onClick={() => handleNavigation('approach')}
              className="font-medium rounded-lg cursor-pointer"
              style={{ 
                color: 'rgba(153, 153, 153, 1)',
                padding: window.innerWidth >= 768 && scrolled ? '0.5rem 1rem' : '0.75rem 1.5rem',
                fontSize: window.innerWidth >= 768 && scrolled ? '0.875rem' : '1rem',
                transition: 'all 300ms'
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(0, 23, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Approach
            </button>
            <button 
              onClick={() => handleNavigation('team')}
              className="font-medium rounded-lg cursor-pointer"
              style={{ 
                color: 'rgba(153, 153, 153, 1)',
                padding: window.innerWidth >= 768 && scrolled ? '0.5rem 1rem' : '0.75rem 1.5rem',
                fontSize: window.innerWidth >= 768 && scrolled ? '0.875rem' : '1rem',
                transition: 'all 300ms'
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(0, 23, 255, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Team
            </button>
          </nav>

          {/* Desktop CTA */}
          <div className="hidden md:flex">
            <a 
              href="https://armeta.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="rounded-lg font-semibold hover:brightness-90 cursor-pointer"
              style={{ 
                backgroundColor: 'rgba(0, 23, 255, 1)', 
                color: 'rgba(255, 255, 255, 1)',
                padding: window.innerWidth >= 768 && scrolled ? '0.625rem 1.25rem' : '0.75rem 1.5rem',
                fontSize: window.innerWidth >= 768 && scrolled ? '0.875rem' : '1rem',
                transition: 'all 300ms'
              }}
            >
              Armeta AI
            </a>
          </div>

          {/* Mobile hamburger button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-md"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              aria-label="Toggle mobile menu"
            >
              <svg
                className="h-6 w-6 transition-transform duration-300"
                style={{
                  transform: isMobileMenuOpen ? 'rotate(90deg)' : 'rotate(0deg)'
                }}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
            </div>
          </header>

          {/* Mobile Navigation Menu */}
          <div 
            className={`md:hidden fixed left-0 right-0 shadow-lg z-[1050] transition-all duration-300 ease-in-out transform overflow-hidden ${
              isMobileMenuOpen 
                ? 'opacity-100 translate-y-0 scale-y-100' 
                : 'opacity-0 -translate-y-2 scale-y-95 pointer-events-none'
            }`}
            style={{ 
              top: '4rem',
              transformOrigin: 'top',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
              borderBottom: '1px solid rgba(153, 153, 153, 0.2)'
            }}
          >
            <div className="px-6 py-4 space-y-1">
              <button
                onClick={() => handleNavigation('approach')}
                className="block w-full text-left px-4 py-3 text-base font-medium transition-all duration-200 rounded-lg"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(31, 107, 255, 0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                Approach
              </button>
              <button
                onClick={() => handleNavigation('team')}
                className="block w-full text-left px-4 py-3 text-base font-medium transition-all duration-200 rounded-lg"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(31, 107, 255, 0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                Team
              </button>
              <a 
                href="https://armeta.ai/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="block px-4 py-3 rounded-lg text-base font-medium transition-all duration-200 hover:brightness-95 cursor-pointer" 
                style={{ backgroundColor: 'rgba(0, 23, 255, 1)', color: 'rgba(255, 255, 255, 1)' }}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Armeta AI
              </a>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
