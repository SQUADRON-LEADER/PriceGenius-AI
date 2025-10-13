# 🚀 Amazon ML Price Predictor

A beautiful, modern React web application for AI-powered Amazon product price prediction using advanced machine learning algorithms.

![Amazon ML Predictor](https://img.shields.io/badge/Amazon-ML%20Predictor-FF9900?style=for-the-badge&logo=amazon&logoColor=white)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Material-UI](https://img.shields.io/badge/Material--UI-5.14.17-0081CB?style=for-the-badge&logo=material-ui&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Features

### 🤖 Advanced ML Algorithms
- **LightGBM**: 45.68% accuracy with optimized hyperparameters
- **XGBoost**: 45.61% accuracy with gradient boosting
- **CatBoost**: 44.22% accuracy with categorical features handling
- **Ensemble Model**: 46.12% accuracy combining all algorithms

### 🎨 Beautiful User Interface
- **Modern Design**: Glass morphism effects with gradient backgrounds
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Interactive Charts**: Real-time visualization with Recharts
- **Smooth Animations**: Framer Motion powered transitions
- **Dark Theme**: Eye-friendly interface with Material-UI

### ⚡ Real-time Predictions
- **Fast Response**: Sub-second prediction times
- **Live Updates**: Interactive forms with instant feedback
- **Multiple Models**: Compare predictions across algorithms
- **Confidence Scores**: Reliability indicators for each prediction

### 📊 Comprehensive Analytics
- **Performance Tracking**: Monitor model accuracy over time
- **Detailed Metrics**: SMAPE, MAE, RMSE, and R² scores
- **Category Analysis**: Price distribution across product categories
- **Historical Trends**: Track prediction performance evolution

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+ (for ML backend)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/amazon-ml-predictor.git
   cd amazon-ml-predictor
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

## 🏗️ Project Structure

```
react-ml-app/
├── public/
│   ├── index.html          # Main HTML template
│   └── manifest.json       # PWA configuration
├── src/
│   ├── components/         # React components
│   │   ├── Dashboard.js    # Main dashboard with metrics
│   │   ├── PricePredictionForm.js  # Prediction interface
│   │   ├── ModelComparison.js      # Algorithm comparison
│   │   ├── Analytics.js    # Performance analytics
│   │   ├── About.js        # Project information
│   │   ├── Navbar.js       # Navigation component
│   │   └── LoadingScreen.js # Loading animation
│   ├── App.js              # Main application component
│   ├── index.js            # Application entry point
│   └── index.css           # Global styles
└── package.json            # Dependencies and scripts
```

## 🎯 Key Components

### Dashboard
- **Real-time Metrics**: Live performance indicators
- **Algorithm Rankings**: Compare model performances
- **Training Progress**: Visual training history
- **Recent Predictions**: Latest prediction results

### Price Prediction
- **Smart Forms**: Intuitive product information input
- **Multiple Predictions**: Compare results across algorithms
- **Confidence Indicators**: Reliability scores for predictions
- **Quick Examples**: Pre-filled sample products

### Model Comparison
- **Performance Charts**: Visual algorithm comparison
- **Hyperparameters**: Detailed model configurations
- **Radar Analysis**: Multi-dimensional performance view
- **Training Statistics**: Comprehensive model metrics

### Analytics
- **Trend Analysis**: Historical performance tracking
- **Category Distribution**: Price range analytics
- **Accuracy Metrics**: Detailed error analysis
- **Key Insights**: Performance summaries

## 🛠️ Technology Stack

### Frontend
- **React 18.2.0**: Modern component-based architecture
- **Material-UI 5.14.17**: Google's Material Design components
- **Framer Motion**: Smooth animations and transitions
- **Recharts**: Interactive data visualization
- **React Router**: Client-side routing

### Backend (Python ML Models)
- **LightGBM**: Microsoft's gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Yandex's gradient boosting library
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation and analysis

### Styling & UI
- **CSS3**: Modern styling with gradients and animations
- **Glass Morphism**: Translucent UI elements
- **Responsive Design**: Mobile-first approach
- **Custom Theme**: Dark mode with brand colors

## 📈 Model Performance

| Algorithm | Accuracy | Training Time | Features |
|-----------|----------|---------------|----------|
| LightGBM  | 45.68%   | 7.2 minutes   | Gradient boosting, leaf-wise tree growth |
| XGBoost   | 45.61%   | 6.8 minutes   | Extreme gradient boosting, level-wise |
| CatBoost  | 44.22%   | 7.4 minutes   | Categorical features, symmetric trees |
| Ensemble  | 46.12%   | -             | Weighted combination of all models |

## 🎨 Design Features

### Visual Elements
- **Gradient Backgrounds**: Beautiful color transitions
- **Glass Morphism**: Frosted glass effect cards
- **Smooth Animations**: Page transitions and hover effects
- **Interactive Charts**: Hover and click interactions
- **Progress Indicators**: Visual feedback for all actions

### User Experience
- **Intuitive Navigation**: Clear menu structure
- **Fast Loading**: Optimized components and assets
- **Error Handling**: Graceful error states
- **Accessibility**: ARIA labels and keyboard navigation
- **Mobile Optimized**: Touch-friendly interface

## 🔧 Customization

### Theme Configuration
Edit `src/App.js` to customize the theme:
```javascript
const theme = createTheme({
  palette: {
    primary: { main: '#667eea' },
    secondary: { main: '#764ba2' },
    // Add your custom colors
  }
});
```

### Component Styling
Each component uses Material-UI's `sx` prop for styling:
```javascript
<Card sx={{
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  // Add your custom styles
}}>
```

## 📱 Responsive Design

The application is fully responsive and tested on:
- **Desktop**: 1920x1080 and above
- **Laptop**: 1366x768 to 1920x1080
- **Tablet**: 768x1024 (iPad)
- **Mobile**: 375x667 (iPhone) and larger

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Netlify
1. Build the project: `npm run build`
2. Upload the `build` folder to Netlify
3. Configure redirects for React Router

### Deploy to Vercel
```bash
npm install -g vercel
vercel
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon ML Challenge**: Dataset and inspiration
- **Material-UI Team**: Beautiful React components
- **Recharts**: Interactive chart library
- **Framer Motion**: Smooth animation library
- **Open Source Community**: Various libraries and tools

## 🔗 Links

- **Live Demo**: [Amazon ML Predictor](https://your-demo-link.com)
- **Documentation**: [API Docs](https://your-docs-link.com)
- **GitHub**: [Source Code](https://github.com/your-username/amazon-ml-predictor)

---

**Made with ❤️ using React and AI**