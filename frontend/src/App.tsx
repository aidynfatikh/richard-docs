import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { HackathonHeader } from './components/HackathonHeader';
import { HackathonFooter } from './components/HackathonFooter';
import { HomePage } from './pages/HomePage';
import { SolutionPage } from './pages/SolutionPage';
import { RealtimeScanPage } from './pages/RealtimeScanPage';

function AppContent() {
  const location = useLocation();
  const isFullScreenRoute = location.pathname === '/scan';

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      {!isFullScreenRoute && <HackathonHeader />}

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/solution" element={<SolutionPage />} />
        <Route path="/scan" element={<RealtimeScanPage />} />
      </Routes>

      {!isFullScreenRoute && <HackathonFooter />}
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App