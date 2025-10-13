import React, { useState } from 'react';
import {
  Container,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Paper,
  Divider,
} from '@mui/material';
import {
  PredictionsOutlined,
  ShoppingCart,
  TrendingUp,
  Analytics,
  MonetizationOn,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const PricePredictionForm = ({ modelData }) => {
  const [formData, setFormData] = useState({
    productName: '',
    category: '',
    description: '',
    brand: '',
    features: '',
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const generatePrediction = () => {
    setLoading(true);
    
    // Simulate API call to ML model
    setTimeout(() => {
      // Mock prediction based on form data
      const basePrice = Math.random() * 1000 + 50; // Random price between $50-$1050
      const categoryMultiplier = getCategoryMultiplier(formData.category);
      const brandMultiplier = getBrandMultiplier(formData.brand);
      
      const predictedPrice = basePrice * categoryMultiplier * brandMultiplier;
      const confidenceScore = Math.random() * 20 + 80; // 80-100% confidence
      
      setPrediction({
        price: predictedPrice.toFixed(2),
        algorithms: {
          lightgbm: (predictedPrice * (0.98 + Math.random() * 0.04)).toFixed(2),
          xgboost: (predictedPrice * (0.97 + Math.random() * 0.06)).toFixed(2),
          catboost: (predictedPrice * (0.96 + Math.random() * 0.08)).toFixed(2),
        },
        ensemble: predictedPrice.toFixed(2),
      });
      setConfidence(confidenceScore.toFixed(1));
      setLoading(false);
    }, 2000);
  };

  const getCategoryMultiplier = (category) => {
    const multipliers = {
      'electronics': 1.5,
      'gaming': 2.0,
      'phones': 1.8,
      'laptops': 2.5,
      'accessories': 0.8,
      'home': 1.2,
    };
    return multipliers[category.toLowerCase()] || 1.0;
  };

  const getBrandMultiplier = (brand) => {
    const multipliers = {
      'apple': 1.8,
      'samsung': 1.5,
      'google': 1.4,
      'sony': 1.3,
      'microsoft': 1.6,
      'generic': 0.7,
    };
    return multipliers[brand.toLowerCase()] || 1.0;
  };

  const isFormValid = formData.productName && formData.category && formData.description;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography
            variant="h3"
            sx={{
              background: 'linear-gradient(45deg, #ffffff, #f0f0f0)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontWeight: 'bold',
              mb: 2,
            }}
          >
            🎯 Price Prediction
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            Enter product details to get AI-powered price predictions
          </Typography>
        </Box>
      </motion.div>

      <Grid container spacing={4}>
        {/* Input Form */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ShoppingCart />
                  Product Information
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Product Name"
                      name="productName"
                      value={formData.productName}
                      onChange={handleInputChange}
                      placeholder="e.g., iPhone 15 Pro Max 256GB"
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Category"
                      name="category"
                      value={formData.category}
                      onChange={handleInputChange}
                      placeholder="e.g., Electronics, Gaming, Phones"
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Brand"
                      name="brand"
                      value={formData.brand}
                      onChange={handleInputChange}
                      placeholder="e.g., Apple, Samsung, Sony"
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      label="Product Description"
                      name="description"
                      value={formData.description}
                      onChange={handleInputChange}
                      placeholder="Detailed product description with specifications..."
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      multiline
                      rows={3}
                      label="Key Features"
                      name="features"
                      value={formData.features}
                      onChange={handleInputChange}
                      placeholder="List key features separated by commas..."
                      variant="outlined"
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Button
                      fullWidth
                      variant="contained"
                      size="large"
                      onClick={generatePrediction}
                      disabled={!isFormValid || loading}
                      startIcon={loading ? <CircularProgress size={20} /> : <PredictionsOutlined />}
                      sx={{
                        py: 1.5,
                        background: 'linear-gradient(45deg, #667eea, #764ba2)',
                        '&:hover': {
                          background: 'linear-gradient(45deg, #5a6fd8, #6a4190)',
                        },
                      }}
                    >
                      {loading ? 'Predicting...' : 'Predict Price'}
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Prediction Results */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingUp />
                  Prediction Results
                </Typography>
                
                {!prediction && !loading && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Analytics sx={{ fontSize: 80, color: 'rgba(255, 255, 255, 0.3)', mb: 2 }} />
                    <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                      Fill out the form to get predictions
                    </Typography>
                  </Box>
                )}
                
                {loading && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CircularProgress size={60} sx={{ mb: 2 }} />
                    <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                      AI models are analyzing...
                    </Typography>
                  </Box>
                )}
                
                {prediction && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    {/* Ensemble Prediction */}
                    <Paper
                      sx={{
                        p: 3,
                        mb: 3,
                        background: 'linear-gradient(45deg, #667eea, #764ba2)',
                        textAlign: 'center',
                      }}
                    >
                      <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'white' }}>
                        ${prediction.ensemble}
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                        Ensemble Prediction
                      </Typography>
                      <Chip
                        label={`${confidence}% Confidence`}
                        sx={{
                          mt: 1,
                          backgroundColor: 'rgba(255, 255, 255, 0.2)',
                          color: 'white',
                        }}
                      />
                    </Paper>
                    
                    {/* Individual Model Predictions */}
                    <Typography variant="h6" sx={{ mb: 2 }}>
                      Individual Model Predictions:
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#667eea' }} />
                            <Typography>LightGBM</Typography>
                          </Box>
                          <Typography variant="h6">${prediction.algorithms.lightgbm}</Typography>
                        </Box>
                        <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)' }} />
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#764ba2' }} />
                            <Typography>XGBoost</Typography>
                          </Box>
                          <Typography variant="h6">${prediction.algorithms.xgboost}</Typography>
                        </Box>
                        <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)' }} />
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#f093fb' }} />
                            <Typography>CatBoost</Typography>
                          </Box>
                          <Typography variant="h6">${prediction.algorithms.catboost}</Typography>
                        </Box>
                      </Grid>
                    </Grid>
                    
                    <Alert 
                      severity="info" 
                      sx={{ 
                        mt: 3,
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        color: 'white',
                        '& .MuiAlert-icon': {
                          color: '#2196f3',
                        },
                      }}
                    >
                      This prediction is based on {modelData.trainingResults.totalModels} trained models with {modelData.trainingResults.bestAccuracy}% accuracy.
                    </Alert>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      {/* Quick Examples */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <Card sx={{ mt: 4 }}>
          <CardContent>
            <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
              <MonetizationOn />
              Quick Examples
            </Typography>
            <Grid container spacing={2}>
              {[
                { name: 'Gaming Laptop', category: 'Gaming', price: '$1,299' },
                { name: 'Wireless Headphones', category: 'Electronics', price: '$89' },
                { name: 'Smartphone', category: 'Phones', price: '$699' },
                { name: 'Smart Watch', category: 'Accessories', price: '$249' },
              ].map((example, index) => (
                <Grid item xs={12} sm={6} md={3} key={index}>
                  <Paper
                    sx={{
                      p: 2,
                      textAlign: 'center',
                      background: 'rgba(255, 255, 255, 0.05)',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        background: 'rgba(255, 255, 255, 0.1)',
                        transform: 'translateY(-4px)',
                      },
                    }}
                    onClick={() => {
                      setFormData({
                        productName: example.name,
                        category: example.category,
                        description: `High-quality ${example.name.toLowerCase()} with premium features`,
                        brand: 'Premium Brand',
                        features: 'Premium, High-quality, Latest technology',
                      });
                    }}
                  >
                    <Typography variant="h6">{example.name}</Typography>
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      {example.category}
                    </Typography>
                    <Typography variant="h6" sx={{ color: '#667eea', mt: 1 }}>
                      ~{example.price}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </motion.div>
    </Container>
  );
};

export default PricePredictionForm;