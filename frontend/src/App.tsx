import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HackathonHeader } from './components/HackathonHeader';
import { HackathonFooter } from './components/HackathonFooter';
import { HomePage } from './pages/HomePage';
import { SolutionPage } from './pages/SolutionPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
        <HackathonHeader />
        
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/solution" element={<SolutionPage />} />
        </Routes>
        
        <HackathonFooter />
      </div>
    </BrowserRouter>
  );
}

export default App