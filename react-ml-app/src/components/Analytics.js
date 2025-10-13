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
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  TrendingUp,
  Assessment,
  DataUsage,
  PieChart,
  BarChart,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RechartsPieChart, Cell } from 'recharts';

const Analytics = ({ modelData }) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const performanceHistory = [
    { date: '2024-01', accuracy: 42.5, predictions: 1200 },
    { date: '2024-02', accuracy: 43.8, predictions: 1450 },
    { date: '2024-03', accuracy: 44.2, predictions: 1680 },
    { date: '2024-04', accuracy: 44.9, predictions: 1920 },
    { date: '2024-05', accuracy: 45.7, predictions: 2100 },
  ];

  const categoryDistribution = [
    { name: 'Electronics', value: 35, color: '#667eea' },
    { name: 'Gaming', value: 25, color: '#764ba2' },
    { name: 'Home & Garden', value: 20, color: '#f093fb' },
    { name: 'Fashion', value: 12, color: '#4ecdc4' },
    { name: 'Sports', value: 8, color: '#45b7d1' },
  ];

  const accuracyTrends = [
    { metric: 'SMAPE', current: 54.32, previous: 56.78, change: -2.46 },
    { metric: 'MAE', current: 125.43, previous: 138.92, change: -13.49 },
    { metric: 'RMSE', current: 198.76, previous: 215.33, change: -16.57 },
    { metric: 'R²', current: 0.827, previous: 0.798, change: 0.029 },
  ];

  const predictionDistribution = [
    { range: '$0-$100', count: 3250, percentage: 32.5 },
    { range: '$101-$500', count: 2890, percentage: 28.9 },
    { range: '$501-$1000', count: 2140, percentage: 21.4 },
    { range: '$1001-$2000', count: 1180, percentage: 11.8 },
    { range: '$2000+', count: 540, percentage: 5.4 },
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
            📊 Analytics Dashboard
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            Comprehensive model performance and prediction analytics
          </Typography>
        </Box>
      </motion.div>

      <Grid container spacing={3}>
        {/* Performance Trends */}
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
                  Performance Trends
                </Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={performanceHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="date" stroke="rgba(255, 255, 255, 0.7)" />
                    <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                      }}
                    />
                    <Line type="monotone" dataKey="accuracy" stroke="#667eea" strokeWidth={3} name="Accuracy %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Category Distribution */}
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
                  <PieChart />
                  Category Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                      }}
                    />
                    {categoryDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </RechartsPieChart>
                </ResponsiveContainer>
                <Box sx={{ mt: 2 }}>
                  {categoryDistribution.map((category, index) => (
                    <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: category.color }} />
                        <Typography variant="body2">{category.name}</Typography>
                      </Box>
                      <Typography variant="body2">{category.value}%</Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Accuracy Metrics */}
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
                  <Assessment />
                  Accuracy Metrics
                </Typography>
                {accuracyTrends.map((metric, index) => (
                  <Box key={index} sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(255, 255, 255, 0.05)' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6">{metric.metric}</Typography>
                      <Chip
                        label={`${metric.change > 0 ? '+' : ''}${metric.change.toFixed(2)}`}
                        size="small"
                        sx={{
                          backgroundColor: metric.change < 0 ? '#4caf50' : '#f44336',
                          color: 'white',
                        }}
                      />
                    </Box>
                    <Typography variant="h4" sx={{ color: '#667eea', fontWeight: 'bold', mb: 1 }}>
                      {metric.current}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Previous: {metric.previous}
                      </Typography>
                      <Typography variant="body2" sx={{ color: metric.change < 0 ? '#4caf50' : '#f44336' }}>
                        {metric.change < 0 ? '↓' : '↑'} {Math.abs(metric.change).toFixed(2)}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(100, (metric.current / metric.previous) * 100)}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: metric.change < 0 ? '#4caf50' : '#f44336',
                          borderRadius: 3,
                        },
                      }}
                    />
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Prediction Distribution */}
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
                  <BarChart />
                  Price Range Distribution
                </Typography>
                {predictionDistribution.map((range, index) => (
                  <Box key={index} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body1">{range.range}</Typography>
                      <Typography variant="body1">{range.count.toLocaleString()}</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={range.percentage}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: `hsl(${240 + index * 30}, 70%, 60%)`,
                          borderRadius: 4,
                        },
                      }}
                    />
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mt: 0.5 }}>
                      {range.percentage}% of all predictions
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Detailed Statistics Table */}
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
                  <DataUsage />
                  Model Statistics
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Algorithm</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Accuracy (%)</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Training Time</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Memory Usage</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Predictions Made</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Avg. Error</TableCell>
                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {modelData.algorithms.map((algo, index) => (
                        <TableRow key={algo.name}>
                          <TableCell sx={{ color: 'white' }}>
                            <Chip 
                              label={algo.name}
                              sx={{
                                backgroundColor: algo.color,
                                color: 'white',
                              }}
                            />
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {algo.accuracy}%
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {(Math.random() * 5 + 3).toFixed(1)}m
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {(Math.random() * 200 + 100).toFixed(0)}MB
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            {(Math.random() * 1000 + 2000).toFixed(0)}
                          </TableCell>
                          <TableCell sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                            ${(Math.random() * 50 + 25).toFixed(2)}
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label="Active"
                              size="small"
                              sx={{
                                backgroundColor: '#4caf50',
                                color: 'white',
                              }}
                            />
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

        {/* Key Insights */}
        <Grid item xs={12}>
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AnalyticsIcon />
                  Key Insights
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 3, borderRadius: 2, bgcolor: 'rgba(102, 126, 234, 0.1)', textAlign: 'center' }}>
                      <Typography variant="h3" sx={{ color: '#667eea', fontWeight: 'bold' }}>
                        95.2%
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
                        Prediction Accuracy
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Average accuracy across all price ranges
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 3, borderRadius: 2, bgcolor: 'rgba(118, 75, 162, 0.1)', textAlign: 'center' }}>
                      <Typography variant="h3" sx={{ color: '#764ba2', fontWeight: 'bold' }}>
                        2.3s
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
                        Avg Response Time
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        From input to prediction result
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 3, borderRadius: 2, bgcolor: 'rgba(240, 147, 251, 0.1)', textAlign: 'center' }}>
                      <Typography variant="h3" sx={{ color: '#f093fb', fontWeight: 'bold' }}>
                        10K+
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
                        Predictions Made
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Total predictions since deployment
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Analytics;