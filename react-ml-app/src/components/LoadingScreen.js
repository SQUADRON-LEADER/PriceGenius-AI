import React from 'react';
import { motion } from 'framer-motion';
import { Box, Typography, CircularProgress } from '@mui/material';
import { SmartToy, Psychology, Analytics } from '@mui/icons-material';

const LoadingScreen = () => {
  const iconVariants = {
    initial: { scale: 0.8, opacity: 0.5 },
    animate: { 
      scale: [0.8, 1.2, 0.8], 
      opacity: [0.5, 1, 0.5],
      transition: { 
        duration: 2, 
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography
            variant="h2"
            sx={{
              background: 'linear-gradient(45deg, #ffffff, #f0f0f0)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontWeight: 'bold',
              mb: 2,
            }}
          >
            Amazon ML Predictor
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, mb: 4 }}>
            <motion.div variants={iconVariants} initial="initial" animate="animate">
              <SmartToy sx={{ fontSize: 40, color: '#ffffff' }} />
            </motion.div>
            <motion.div 
              variants={iconVariants} 
              initial="initial" 
              animate="animate"
              style={{ animationDelay: '0.5s' }}
            >
              <Psychology sx={{ fontSize: 40, color: '#ffffff' }} />
            </motion.div>
            <motion.div 
              variants={iconVariants} 
              initial="initial" 
              animate="animate"
              style={{ animationDelay: '1s' }}
            >
              <Analytics sx={{ fontSize: 40, color: '#ffffff' }} />
            </motion.div>
          </Box>
          
          <CircularProgress 
            size={60} 
            thickness={4}
            sx={{ 
              color: '#ffffff',
              mb: 2,
            }}
          />
          
          <Typography
            variant="h6"
            sx={{
              color: 'rgba(255, 255, 255, 0.9)',
              fontWeight: 300,
            }}
          >
            Loading AI Models...
          </Typography>
        </Box>
      </motion.div>
    </Box>
  );
};

export default LoadingScreen;