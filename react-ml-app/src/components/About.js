import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Chip,
} from '@mui/material';
import {
  Info,
  Psychology,
  Speed,
  Accuracy,
  DataUsage,
  Code,
  GitHub,
  School,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const About = () => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const features = [
    {
      icon: <Psychology />,
      title: 'Advanced ML Algorithms',
      description: 'LightGBM, XGBoost, and CatBoost ensemble for maximum accuracy',
    },
    {
      icon: <Speed />,
      title: 'Real-time Predictions',
      description: 'Lightning-fast price predictions with sub-second response times',
    },
    {
      icon: <Accuracy />,
      title: 'High Accuracy',
      description: '45.68% accuracy on validation data with ensemble optimization',
    },
    {
      icon: <DataUsage />,
      title: 'Big Data Processing',
      description: 'Trained on 75,000+ Amazon product samples with feature engineering',
    },
  ];

  const technologies = [
    { name: 'Python', category: 'Backend', color: '#3776ab' },
    { name: 'React', category: 'Frontend', color: '#61dafb' },
    { name: 'LightGBM', category: 'ML', color: '#667eea' },
    { name: 'XGBoost', category: 'ML', color: '#764ba2' },
    { name: 'CatBoost', category: 'ML', color: '#f093fb' },
    { name: 'Scikit-learn', category: 'ML', color: '#f7931e' },
    { name: 'Material-UI', category: 'Frontend', color: '#0081cb' },
    { name: 'Plotly', category: 'Visualization', color: '#3f51b5' },
  ];

  const teamMembers = [
    {
      name: 'AI Model',
      role: 'Lead ML Engineer',
      avatar: '🤖',
      description: 'Advanced ensemble model with 3 algorithms and 10 epochs training',
    },
    {
      name: 'Data Pipeline',
      role: 'Data Engineer', 
      avatar: '📊',
      description: 'TF-IDF vectorization, SVD reduction, and feature engineering',
    },
    {
      name: 'React Interface',
      role: 'Frontend Developer',
      avatar: '⚛️',
      description: 'Beautiful, responsive UI with real-time predictions',
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
            🚀 About Amazon ML Predictor
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.8)', maxWidth: 800, mx: 'auto' }}>
            An advanced machine learning application that predicts Amazon product prices using state-of-the-art 
            ensemble algorithms with a beautiful, interactive React interface.
          </Typography>
        </Box>
      </motion.div>

      <Grid container spacing={4}>
        {/* Project Overview */}
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
                  <Info />
                  Project Overview
                </Typography>
                
                <Typography variant="body1" sx={{ mb: 3, lineHeight: 1.8, color: 'rgba(255, 255, 255, 0.9)' }}>
                  This project represents a comprehensive machine learning solution for predicting Amazon product prices. 
                  Built with cutting-edge technology and best practices, it combines the power of ensemble learning 
                  with an intuitive user interface.
                </Typography>
                
                <Typography variant="h6" sx={{ mb: 2, color: '#667eea' }}>
                  🎯 Key Objectives
                </Typography>
                
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <Accuracy sx={{ color: '#667eea' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="High Accuracy Predictions"
                      secondary="Achieve maximum prediction accuracy using ensemble methods"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Speed sx={{ color: '#764ba2' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Fast Performance"
                      secondary="Optimized training and inference for real-time applications"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Psychology sx={{ color: '#f093fb' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="User-Friendly Interface"
                      secondary="Intuitive React-based UI for seamless user experience"
                    />
                  </ListItem>
                </List>

                <Divider sx={{ my: 3, bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

                <Typography variant="h6" sx={{ mb: 2, color: '#667eea' }}>
                  📈 Model Performance
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(102, 126, 234, 0.1)' }}>
                      <Typography variant="h4" sx={{ color: '#667eea', fontWeight: 'bold' }}>
                        45.68%
                      </Typography>
                      <Typography variant="body2">Best Accuracy</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(118, 75, 162, 0.1)' }}>
                      <Typography variant="h4" sx={{ color: '#764ba2', fontWeight: 'bold' }}>
                        30
                      </Typography>
                      <Typography variant="body2">Models Trained</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(240, 147, 251, 0.1)' }}>
                      <Typography variant="h4" sx={{ color: '#f093fb', fontWeight: 'bold' }}>
                        21.3m
                      </Typography>
                      <Typography variant="body2">Training Time</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, bgcolor: 'rgba(78, 205, 196, 0.1)' }}>
                      <Typography variant="h4" sx={{ color: '#4ecdc4', fontWeight: 'bold' }}>
                        75K
                      </Typography>
                      <Typography variant="body2">Training Samples</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Technical Features */}
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
                  <Code />
                  Key Features
                </Typography>
                
                {features.map((feature, index) => (
                  <Box key={index} sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Avatar sx={{ bgcolor: '#667eea', mr: 2, width: 32, height: 32 }}>
                        {feature.icon}
                      </Avatar>
                      <Typography variant="h6">{feature.title}</Typography>
                    </Box>
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', ml: 6 }}>
                      {feature.description}
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Technology Stack */}
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
                  <School />
                  Technology Stack
                </Typography>
                
                <Grid container spacing={2}>
                  {technologies.map((tech, index) => (
                    <Grid item xs={6} sm={4} md={3} key={index}>
                      <Box
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          textAlign: 'center',
                          bgcolor: 'rgba(255, 255, 255, 0.05)',
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            bgcolor: 'rgba(255, 255, 255, 0.1)',
                            transform: 'translateY(-4px)',
                          },
                        }}
                      >
                        <Box
                          sx={{
                            width: 40,
                            height: 40,
                            borderRadius: '50%',
                            bgcolor: tech.color,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mx: 'auto',
                            mb: 1,
                          }}
                        >
                          <Typography variant="h6" sx={{ color: 'white' }}>
                            {tech.name.charAt(0)}
                          </Typography>
                        </Box>
                        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                          {tech.name}
                        </Typography>
                        <Chip 
                          label={tech.category} 
                          size="small"
                          sx={{
                            fontSize: '0.6rem',
                            height: 16,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
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

        {/* Team */}
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
                  <GitHub />
                  System Components
                </Typography>
                
                {teamMembers.map((member, index) => (
                  <Box key={index} sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
                    <Avatar sx={{ bgcolor: 'transparent', fontSize: '2rem', mr: 3 }}>
                      {member.avatar}
                    </Avatar>
                    <Box>
                      <Typography variant="h6">{member.name}</Typography>
                      <Typography variant="body2" sx={{ color: '#667eea', mb: 0.5 }}>
                        {member.role}
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        {member.description}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Architecture Overview */}
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
                  System Architecture
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center', p: 3, borderRadius: 2, bgcolor: 'rgba(102, 126, 234, 0.1)' }}>
                      <Typography variant="h4" sx={{ mb: 2 }}>📊</Typography>
                      <Typography variant="h6" sx={{ color: '#667eea', mb: 2 }}>
                        Data Processing
                      </Typography>
                      <List dense>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="TF-IDF Vectorization" secondary="10,000 features" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="SVD Reduction" secondary="200 components" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="Feature Engineering" secondary="Numeric + Text" />
                        </ListItem>
                      </List>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center', p: 3, borderRadius: 2, bgcolor: 'rgba(118, 75, 162, 0.1)' }}>
                      <Typography variant="h4" sx={{ mb: 2 }}>🤖</Typography>
                      <Typography variant="h6" sx={{ color: '#764ba2', mb: 2 }}>
                        ML Pipeline
                      </Typography>
                      <List dense>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="LightGBM" secondary="45.68% accuracy" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="XGBoost" secondary="45.61% accuracy" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="CatBoost" secondary="44.22% accuracy" />
                        </ListItem>
                      </List>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center', p: 3, borderRadius: 2, bgcolor: 'rgba(240, 147, 251, 0.1)' }}>
                      <Typography variant="h4" sx={{ mb: 2 }}>⚛️</Typography>
                      <Typography variant="h6" sx={{ color: '#f093fb', mb: 2 }}>
                        User Interface
                      </Typography>
                      <List dense>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="React Components" secondary="Responsive design" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="Material-UI" secondary="Modern styling" />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText primary="Interactive Charts" secondary="Real-time updates" />
                        </ListItem>
                      </List>
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

export default About;