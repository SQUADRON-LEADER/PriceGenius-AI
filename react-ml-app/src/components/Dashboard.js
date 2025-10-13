import React, { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Button,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  Speed,
  Psychology,
  Timeline,
  MonetizationOn,
  AssessmentOutlined as Accuracy,
  ModelTraining,
  Refresh,
  CheckCircle,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import mlModelService from '../services/mlModelService';

const Dashboard = ({ modelData = {} }) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  // Default data structure to prevent undefined errors
  const defaultData = {
    algorithms: [
      { name: 'LightGBM', accuracy: 45.68 },
      { name: 'XGBoost', accuracy: 45.61 },
      { name: 'CatBoost', accuracy: 44.22 }
    ],
    trainingResults: {
      bestAccuracy: 45.68,
      totalModels: 3,
      trainingTime: 7.2,
      ensembleAccuracy: 46.12
    },
    predictionHistory: [
      { id: 1, price: 299.99, confidence: 0.92, product: 'Electronics' },
      { id: 2, price: 149.50, confidence: 0.88, product: 'Books' },
      { id: 3, price: 89.99, confidence: 0.95, product: 'Home & Garden' }
    ]
  };

  // Use provided modelData or fallback to defaultData
  const data = { ...defaultData, ...modelData };

  const accuracyData = (data.algorithms || []).map((algo, index) => ({
    name: algo.name,
    accuracy: algo.accuracy,
    epoch: index + 1,
  }));

  const trainingProgressData = [
    { epoch: 1, lightgbm: 42.1, xgboost: 41.8, catboost: 40.5 },
    { epoch: 3, lightgbm: 44.2, xgboost: 44.1, catboost: 42.3 },
    { epoch: 6, lightgbm: 45.1, xgboost: 44.8, catboost: 43.8 },
    { epoch: 9, lightgbm: 45.6, xgboost: 45.3, catboost: 44.1 },
    { epoch: 10, lightgbm: 45.68, xgboost: 45.61, catboost: 44.22 },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
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
            🚀 AI-Powered Price Prediction
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)', maxWidth: 600, mx: 'auto' }}>
            Advanced machine learning models trained on Amazon marketplace data to predict product prices with high accuracy
          </Typography>
        </Box>
      </motion.div>

      <Grid container spacing={3}>
        {/* Key Metrics Cards */}
        <Grid item xs={12} md={3}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: '#667eea', mr: 2 }}>
                    <Accuracy />
                  </Avatar>
                  <Typography variant="h6">Best Accuracy</Typography>
                </Box>
                <Typography variant="h3" sx={{ color: '#667eea', fontWeight: 'bold' }}>
                  {data.trainingResults.bestAccuracy}%
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  LightGBM Algorithm
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: '#764ba2', mr: 2 }}>
                    <ModelTraining />
                  </Avatar>
                  <Typography variant="h6">Models Trained</Typography>
                </Box>
                <Typography variant="h3" sx={{ color: '#f093fb', fontWeight: 'bold' }}>
                  {data.trainingResults.totalModels}
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  3 Algorithms × 10 Epochs
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: '#f093fb', mr: 2 }}>
                    <Speed />
                  </Avatar>
                  <Typography variant="h6">Training Time</Typography>
                </Box>
                <Typography variant="h3" sx={{ color: '#4facfe', fontWeight: 'bold' }}>
                  {data.trainingResults.trainingTime}m
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Optimized Performance
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={3}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: '#4ecdc4', mr: 2 }}>
                    <TrendingUp />
                  </Avatar>
                  <Typography variant="h6">Ensemble</Typography>
                </Box>
                <Typography variant="h3" sx={{ color: '#43e97b', fontWeight: 'bold' }}>
                  {data.trainingResults.ensembleAccuracy}%
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Combined Models
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Model Performance Chart */}
        <Grid item xs={12} md={8}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Timeline />
                  Training Progress
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingProgressData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="epoch" stroke="rgba(255, 255, 255, 0.7)" />
                    <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                      }}
                    />
                    <Line type="monotone" dataKey="lightgbm" stroke="#667eea" strokeWidth={3} name="LightGBM" />
                    <Line type="monotone" dataKey="xgboost" stroke="#764ba2" strokeWidth={3} name="XGBoost" />
                    <Line type="monotone" dataKey="catboost" stroke="#f093fb" strokeWidth={3} name="CatBoost" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Algorithm Rankings */}
        <Grid item xs={12} md={4}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Psychology />
                  Model Rankings
                </Typography>
                <List>
                  {(data.algorithms || [])
                    .sort((a, b) => b.accuracy - a.accuracy)
                    .map((algo, index) => (
                      <ListItem key={algo.name} sx={{ px: 0 }}>
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: algo.color }}>
                            {index + 1}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={algo.name}
                          secondary={
                            <Box sx={{ mt: 1 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="body2">Accuracy</Typography>
                                <Typography variant="body2">{algo.accuracy}%</Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={algo.accuracy}
                                sx={{
                                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                  '& .MuiLinearProgress-bar': {
                                    backgroundColor: algo.color,
                                  },
                                }}
                              />
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Recent Predictions */}
        <Grid item xs={12}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MonetizationOn />
                  Recent Predictions
                </Typography>
                <Grid container spacing={2}>
                  {(data.predictionHistory || []).map((prediction, index) => (
                    <Grid item xs={12} sm={6} md={3} key={index}>
                      <Box
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          background: 'rgba(255, 255, 255, 0.05)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        <Typography variant="h6" sx={{ mb: 1 }}>
                          {prediction.product}
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1 }}>
                          Predicted: ${prediction.predictedPrice}
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
                          Actual: ${prediction.actualPrice}
                        </Typography>
                        <Chip
                          label={`${prediction.accuracy}% accurate`}
                          size="small"
                          sx={{
                            backgroundColor: prediction.accuracy > 95 ? '#4caf50' : '#ff9800',
                            color: 'white',
                          }}
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;