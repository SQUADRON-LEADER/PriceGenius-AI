import React from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  Link,
  IconButton,
  Divider,
  Chip,
} from '@mui/material';
import {
  GitHub,
  LinkedIn,
  Twitter,
  Email,
  Favorite,
  TrendingUp,
  Psychology,
  Speed,
  Code,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const stats = [
    { label: 'Models Trained', value: '30+', icon: <Psychology /> },
    { label: 'Predictions Made', value: '10K+', icon: <TrendingUp /> },
    { label: 'Accuracy Rate', value: '45.68%', icon: <Speed /> },
    { label: 'Code Lines', value: '5K+', icon: <Code /> },
  ];

  const links = {
    product: [
      { name: 'Dashboard', href: '/' },
      { name: 'Predictions', href: '/predict' },
      { name: 'Models', href: '/models' },
      { name: 'Analytics', href: '/analytics' },
    ],
    resources: [
      { name: 'Documentation', href: '#' },
      { name: 'API Reference', href: '#' },
      { name: 'Tutorials', href: '#' },
      { name: 'Support', href: '#' },
    ],
    company: [
      { name: 'About', href: '/about' },
      { name: 'Blog', href: '#' },
      { name: 'Careers', href: '#' },
      { name: 'Contact', href: '#' },
    ],
  };

  return (
    <Box
      component="footer"
      sx={{
        background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        mt: 8,
        py: 6,
      }}
    >
      <Container maxWidth="xl">
        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <Grid container spacing={3} sx={{ mb: 6 }}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Box
                    sx={{
                      textAlign: 'center',
                      p: 3,
                      borderRadius: 3,
                      background: 'rgba(255, 255, 255, 0.05)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        background: 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    <Box sx={{ color: '#667eea', mb: 1 }}>{stat.icon}</Box>
                    <Typography variant="h4" sx={{ color: '#667eea', fontWeight: 'bold', mb: 1 }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                      {stat.label}
                    </Typography>
                  </Box>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </motion.div>

        <Divider sx={{ mb: 6, bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Main Footer Content */}
        <Grid container spacing={4}>
          {/* Brand Section */}
          <Grid item xs={12} md={4}>
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <Typography
                variant="h4"
                sx={{
                  background: 'linear-gradient(45deg, #667eea, #764ba2)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontWeight: 'bold',
                  mb: 2,
                }}
              >
                🤖 ML Predictor
              </Typography>
              <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 3, lineHeight: 1.6 }}>
                Advanced AI-powered price prediction using state-of-the-art machine learning algorithms. 
                Built with React and powered by ensemble models for maximum accuracy.
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
                <Chip
                  label="LightGBM"
                  size="small"
                  sx={{ backgroundColor: '#667eea', color: 'white' }}
                />
                <Chip
                  label="XGBoost"
                  size="small"
                  sx={{ backgroundColor: '#764ba2', color: 'white' }}
                />
                <Chip
                  label="CatBoost"
                  size="small"
                  sx={{ backgroundColor: '#f093fb', color: 'white' }}
                />
              </Box>

              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton
                  sx={{
                    color: 'rgba(255, 255, 255, 0.8)',
                    '&:hover': { color: '#667eea', backgroundColor: 'rgba(102, 126, 234, 0.1)' },
                  }}
                >
                  <GitHub />
                </IconButton>
                <IconButton
                  sx={{
                    color: 'rgba(255, 255, 255, 0.8)',
                    '&:hover': { color: '#0077b5', backgroundColor: 'rgba(0, 119, 181, 0.1)' },
                  }}
                >
                  <LinkedIn />
                </IconButton>
                <IconButton
                  sx={{
                    color: 'rgba(255, 255, 255, 0.8)',
                    '&:hover': { color: '#1da1f2', backgroundColor: 'rgba(29, 161, 242, 0.1)' },
                  }}
                >
                  <Twitter />
                </IconButton>
                <IconButton
                  sx={{
                    color: 'rgba(255, 255, 255, 0.8)',
                    '&:hover': { color: '#ea4335', backgroundColor: 'rgba(234, 67, 53, 0.1)' },
                  }}
                >
                  <Email />
                </IconButton>
              </Box>
            </motion.div>
          </Grid>

          {/* Links Sections */}
          <Grid item xs={12} md={8}>
            <Grid container spacing={4}>
              <Grid item xs={12} sm={4}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                  viewport={{ once: true }}
                >
                  <Typography variant="h6" sx={{ color: 'white', mb: 2, fontWeight: 'bold' }}>
                    Product
                  </Typography>
                  {links.product.map((link, index) => (
                    <Link
                      key={index}
                      href={link.href}
                      sx={{
                        display: 'block',
                        color: 'rgba(255, 255, 255, 0.7)',
                        textDecoration: 'none',
                        mb: 1,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: '#667eea',
                          transform: 'translateX(4px)',
                        },
                      }}
                    >
                      {link.name}
                    </Link>
                  ))}
                </motion.div>
              </Grid>

              <Grid item xs={12} sm={4}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  viewport={{ once: true }}
                >
                  <Typography variant="h6" sx={{ color: 'white', mb: 2, fontWeight: 'bold' }}>
                    Resources
                  </Typography>
                  {links.resources.map((link, index) => (
                    <Link
                      key={index}
                      href={link.href}
                      sx={{
                        display: 'block',
                        color: 'rgba(255, 255, 255, 0.7)',
                        textDecoration: 'none',
                        mb: 1,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: '#764ba2',
                          transform: 'translateX(4px)',
                        },
                      }}
                    >
                      {link.name}
                    </Link>
                  ))}
                </motion.div>
              </Grid>

              <Grid item xs={12} sm={4}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.3 }}
                  viewport={{ once: true }}
                >
                  <Typography variant="h6" sx={{ color: 'white', mb: 2, fontWeight: 'bold' }}>
                    Company
                  </Typography>
                  {links.company.map((link, index) => (
                    <Link
                      key={index}
                      href={link.href}
                      sx={{
                        display: 'block',
                        color: 'rgba(255, 255, 255, 0.7)',
                        textDecoration: 'none',
                        mb: 1,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: '#f093fb',
                          transform: 'translateX(4px)',
                        },
                      }}
                    >
                      {link.name}
                    </Link>
                  ))}
                </motion.div>
              </Grid>
            </Grid>
          </Grid>
        </Grid>

        <Divider sx={{ my: 4, bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Bottom Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 2,
            }}
          >
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
              © {currentYear} Amazon ML Predictor. All rights reserved.
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                Made with
              </Typography>
              <Favorite sx={{ color: '#e91e63', fontSize: 16 }} />
              <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                using React & AI
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.6)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  '&:hover': { color: '#667eea' },
                }}
              >
                Privacy Policy
              </Link>
              <Link
                href="#"
                sx={{
                  color: 'rgba(255, 255, 255, 0.6)',
                  textDecoration: 'none',
                  fontSize: '0.875rem',
                  '&:hover': { color: '#667eea' },
                }}
              >
                Terms of Service
              </Link>
            </Box>
          </Box>
        </motion.div>
      </Container>
    </Box>
  );
};

export default Footer;