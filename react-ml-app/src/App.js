import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import PricePredictionForm from './components/PricePredictionForm';
import ModelComparison from './components/ModelComparison';
import Analytics from './components/Analytics';
import About from './components/About';
import LoadingScreen from './components/LoadingScreen';
import Footer from './components/Footer';
import mlModelService from './services/mlModelService';

// Create custom theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
      light: '#8fa4f3',
      dark: '#4c63d2',
    },
    secondary: {
      main: '#764ba2',
      light: '#9c6fd4',
      dark: '#5a3a7d',
    },
    background: {
      default: 'transparent',
      paper: 'rgba(255, 255, 255, 0.1)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.8)',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3.5rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2.5rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.5rem',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        },
      },
    },
  },
});

function App() {
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState('dashboard');
  
  // Mock data for model results (will be replaced with real API data)
  const [modelData, setModelData] = useState({
    algorithms: [
      { name: 'LightGBM', accuracy: 45.68, color: '#667eea' },
      { name: 'XGBoost', accuracy: 45.61, color: '#764ba2' },
      { name: 'CatBoost', accuracy: 44.22, color: '#f093fb' },
    ],
    trainingResults: {
      totalModels: 30,
      trainingTime: 21.34,
      bestAccuracy: 45.68,
      ensembleAccuracy: 46.12,
    },
    predictionHistory: [
      { product: 'Gaming Laptop', predictedPrice: 1299.99, actualPrice: 1350.00, accuracy: 96.3 },
      { product: 'Wireless Headphones', predictedPrice: 89.99, actualPrice: 85.00, accuracy: 94.1 },
      { product: 'Smartphone', predictedPrice: 699.99, actualPrice: 729.99, accuracy: 95.9 },
      { product: 'Smart Watch', predictedPrice: 249.99, actualPrice: 259.99, accuracy: 96.2 },
    ],
  });

  useEffect(() => {
    // Load model data from API
    const loadModelData = async () => {
      try {
        const response = await mlModelService.getModelStatus();
        if (response.success) {
          setModelData(response.data);
        }
      } catch (error) {
        console.log('Using fallback data:', error);
      }
    };

    loadModelData();

    // Simulate loading time
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ 
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        }}>
          <Navbar currentPage={currentPage} setCurrentPage={setCurrentPage} />
          
          <Box sx={{ flex: 1 }}>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentPage}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Routes>
                  <Route 
                    path="/" 
                    element={<Dashboard modelData={modelData} />} 
                  />
                  <Route 
                    path="/predict" 
                    element={<PricePredictionForm modelData={modelData} />} 
                  />
                  <Route 
                    path="/models" 
                    element={<ModelComparison modelData={modelData} />} 
                  />
                  <Route 
                    path="/analytics" 
                    element={<Analytics modelData={modelData} />} 
                  />
                  <Route 
                    path="/about" 
                    element={<About />} 
                  />
                </Routes>
              </motion.div>
            </AnimatePresence>
          </Box>
          
          <Footer />
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;