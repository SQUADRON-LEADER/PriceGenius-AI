import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  Avatar,
} from '@mui/material';
import {
  ModelTraining,
  Speed,
  Accuracy,
  Psychology,
  Timeline,
  TrendingUp,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const ModelComparison = ({ modelData }) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const performanceData = modelData.algorithms.map(algo => ({
    name: algo.name.replace('_', ' '),
    accuracy: algo.accuracy,
    speed: Math.random() * 30 + 70, // Mock speed score
    memory: Math.random() * 40 + 60, // Mock memory usage
  }));

  const radarData = [
    { metric: 'Accuracy', LightGBM: 91, XGBoost: 90, CatBoost: 88 },
    { metric: 'Speed', LightGBM: 88, XGBoost: 85, CatBoost: 82 },
    { metric: 'Memory Efficiency', LightGBM: 85, XGBoost: 80, CatBoost: 90 },
    { metric: 'Stability', LightGBM: 92, XGBoost: 89, CatBoost: 94 },
    { metric: 'Scalability', LightGBM: 95, XGBoost: 92, CatBoost: 87 },
  ];

  const hyperparameters = [
    {
      algorithm: 'LightGBM',
      params: {
        'Learning Rate': '0.044',
        'Num Leaves': '127',
        'Max Depth': '12',
        'Iterations': '1440',
        'Feature Fraction': '0.9',
      }
    },
    {
      algorithm: 'XGBoost',
      params: {
        'Learning Rate': '0.051',
        'Max Depth': '9',
        'Subsample': '0.84',
        'Iterations': '1190',
        'Colsample Bytree': '0.9',
      }
    },
    {
      algorithm: 'CatBoost',
      params: {
        'Learning Rate': '0.030',
        'Depth': '9',
        'L2 Leaf Reg': '3',
        'Iterations': '1400',
        'Random Seed': '42',
      }
    },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
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
            🤖 Model Comparison
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            Detailed analysis and comparison of ML algorithms
          </Typography>
        </Box>
      </motion.div>

      <Grid container spacing={3}>
        {/* Algorithm Performance Chart */}
        <Grid item xs={12} lg={8}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingUp />
                  Performance Comparison
                </Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255, 255, 255, 0.7)" />
                    <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="accuracy" fill="#667eea" name="Accuracy %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Model Rankings */}
        <Grid item xs={12} lg={4}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Psychology />
                  Rankings
                </Typography>
                {modelData.algorithms
                  .sort((a, b) => b.accuracy - a.accuracy)
                  .map((algo, index) => (
                    <Box key={algo.name} sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(255, 255, 255, 0.05)' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar sx={{ bgcolor: algo.color, mr: 2, width: 32, height: 32 }}>
                          {index + 1}
                        </Avatar>
                        <Box>
                          <Typography variant="h6">{algo.name}</Typography>
                          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                            {algo.accuracy}% Accuracy
                          </Typography>
                        </Box>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={(algo.accuracy / modelData.trainingResults.bestAccuracy) * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: algo.color,
                            borderRadius: 4,
                          },
                        }}
                      />
                      <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                        <Chip 
                          label={index === 0 ? "Best" : index === 1 ? "Good" : "Fair"} 
                          size="small"
                          sx={{ backgroundColor: index === 0 ? '#4caf50' : index === 1 ? '#ff9800' : '#f44336' }}
                        />
                      </Box>
                    </Box>
                  ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Radar Chart */}
        <Grid item xs={12} lg={6}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Accuracy />
                  Multi-Metric Analysis
                </Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255, 255, 255, 0.2)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }} />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 100]} 
                      tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 10 }}
                    />
                    <Radar name="LightGBM" dataKey="LightGBM" stroke="#667eea" fill="#667eea" fillOpacity={0.2} strokeWidth={2} />
                    <Radar name="XGBoost" dataKey="XGBoost" stroke="#764ba2" fill="#764ba2" fillOpacity={0.2} strokeWidth={2} />
                    <Radar name="CatBoost" dataKey="CatBoost" stroke="#f093fb" fill="#f093fb" fillOpacity={0.2} strokeWidth={2} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Training Statistics */}
        <Grid item xs={12} lg={6}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Timeline />
                  Training Statistics
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(102, 126, 234, 0.1)' }}>
                      <ModelTraining sx={{ fontSize: 40, color: '#667eea', mb: 1 }} />
                      <Typography variant="h4" sx={{ color: '#667eea', fontWeight: 'bold' }}>
                        {modelData.trainingResults.totalModels}
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Total Models
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(118, 75, 162, 0.1)' }}>
                      <Speed sx={{ fontSize: 40, color: '#764ba2', mb: 1 }} />
                      <Typography variant="h4" sx={{ color: '#764ba2', fontWeight: 'bold' }}>
                        {modelData.trainingResults.trainingTime}m
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Training Time
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12}>
                    <Box sx={{ p: 2, borderRadius: 2, bgcolor: 'rgba(240, 147, 251, 0.1)' }}>
                      <Typography variant="h6" sx={{ color: '#f093fb', mb: 2 }}>
                        Ensemble Performance
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography>Accuracy</Typography>
                        <Typography>{modelData.trainingResults.ensembleAccuracy}%</Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={modelData.trainingResults.ensembleAccuracy}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: '#f093fb',
                            borderRadius: 4,
                          },
                        }}
                      />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Hyperparameters Table */}
        <Grid item xs={12}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ModelTraining />
                  Best Hyperparameters
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Algorithm</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Learning Rate</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Depth/Leaves</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Iterations</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Other Parameters</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {hyperparameters.map((row) => (
                        <TableRow key={row.algorithm}>
                          <TableCell sx={{ color: 'white' }}>
                            <Chip 
                              label={row.algorithm}
                              sx={{
                                backgroundColor: 
                                  row.algorithm === 'LightGBM' ? '#667eea' :
                                  row.algorithm === 'XGBoost' ? '#764ba2' : '#f093fb',
                                color: 'white',
                              }}
                            />
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {row.params['Learning Rate']}
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {row.params['Max Depth'] || row.params['Num Leaves'] || row.params['Depth']}
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {row.params['Iterations']}
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {Object.entries(row.params)
                              .filter(([key]) => !['Learning Rate', 'Max Depth', 'Num Leaves', 'Depth', 'Iterations'].includes(key))
                              .map(([key, value]) => `${key}: ${value}`)
                              .join(', ')}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ModelComparison;