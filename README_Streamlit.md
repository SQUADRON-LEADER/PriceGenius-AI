# Amazon ML Price Predictor - Streamlit Dashboard 🛒

A comprehensive, dynamic web application built with Streamlit for predicting Amazon product prices using advanced machine learning algorithms.

## 🌟 Features

### 🎯 Price Prediction
- **Multi-Input Methods**: Manual entry, catalog content parsing, and quick examples
- **Real-time Predictions**: Get instant price predictions using ensemble ML models
- **Algorithm Comparison**: View predictions from LightGBM, XGBoost, and CatBoost
- **Confidence Scoring**: Each prediction comes with a confidence percentage
- **Visual Analysis**: Interactive charts showing algorithm performance

### 📊 Data Analytics Dashboard
- **Price Distribution Analysis**: Histogram and statistics of training data
- **Category Breakdown**: Pie charts and analysis of product categories
- **Prediction History**: Track and analyze your prediction patterns
- **Algorithm Performance**: Box plots comparing model outputs

### 🤖 Model Performance Tracking
- **Accuracy Metrics**: Real-time display of model SMAPE scores
- **Training Statistics**: Performance metrics and hyperparameters
- **Evolution Tracking**: See how model accuracy changes over time
- **Detailed Configuration**: View complete model hyperparameters

### 📈 Market Insights & Trends
- **Market Trend Analysis**: Simulated market price trends for different categories
- **Category Performance**: Growth analysis across product segments
- **Key Insights**: Market intelligence and growth indicators
- **Accuracy Evolution**: Track prediction accuracy improvements

### ⚙️ Settings & Configuration
- **Ensemble Weights**: Adjust model weights in the ensemble
- **Confidence Thresholds**: Set minimum confidence levels
- **Display Preferences**: Customize charts and themes
- **Data Management**: Export data, clear history, reset performance
- **API Configuration**: Connect to backend services

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit plotly pandas numpy scikit-learn requests
```

### Running the Application

#### Method 1: Direct Command
```bash
python -m streamlit run streamlit_app.py
```

#### Method 2: Using Batch File
Double-click `start-streamlit.bat` or run:
```bash
start-streamlit.bat
```

#### Method 3: With Backend (Full Features)
1. Start the Flask backend:
```bash
cd python-backend
python app.py
```

2. Start Streamlit (in main directory):
```bash
python -m streamlit run streamlit_app.py
```

## 🎮 How to Use

### Making Predictions

1. **Navigate to Price Prediction**: Select "🎯 Price Prediction" from sidebar
2. **Choose Input Method**:
   - **Manual Entry**: Fill in product name, brand, category, and description
   - **Catalog Content**: Paste Amazon catalog content for automatic parsing
   - **Quick Examples**: Choose from pre-loaded product examples
3. **Click "🔮 Predict Price"**: Get instant predictions with confidence scores
4. **View Results**: See ensemble prediction and individual algorithm outputs

### Example Inputs

#### Manual Entry Example:
- **Product Name**: Apple MacBook Pro 16-inch
- **Brand**: Apple
- **Category**: Gaming Laptop
- **Description**: M1 Pro chip, 16GB RAM, 512GB SSD, professional laptop

#### Catalog Content Example:
```
Item Name: Apple iPhone 14 Pro Max, Brand: Apple, Features: A16 Bionic chip, Pro camera system, 128GB storage, 6.7-inch display, premium smartphone with advanced features
```

#### Quick Examples Available:
- Apple iPhone 14 Pro Max
- Samsung Gaming Monitor 4K
- Sony WH-1000XM4 Headphones
- Dell XPS 13 Laptop
- iPad Air with Apple Pencil

### Exploring Analytics

1. **Data Analytics**: View training data distributions and category analysis
2. **Model Performance**: Compare algorithm accuracies and view configurations
3. **Market Insights**: Explore market trends and category performance
4. **Settings**: Customize the application to your preferences

## 🛠️ Technical Architecture

### Machine Learning Models
- **LightGBM**: 45.68% accuracy (SMAPE) - Primary model
- **XGBoost**: 45.61% accuracy (SMAPE) - Secondary model
- **CatBoost**: 44.22% accuracy (SMAPE) - Tertiary model
- **Ensemble**: Weighted average based on individual model performance

### Feature Engineering
- **TF-IDF Vectorization**: Text feature extraction from product descriptions
- **SVD Dimensionality Reduction**: 200-component feature space
- **Robust Scaling**: Feature normalization for model input
- **212-Dimensional Feature Space**: Comprehensive product representation

### Data Processing Pipeline
1. **Content Parsing**: Extract product information from catalog content
2. **Feature Extraction**: Convert text to numerical features
3. **Preprocessing**: Scale and normalize features
4. **Prediction**: Generate ensemble prediction with confidence
5. **Visualization**: Create interactive charts and displays

## 📱 User Interface Components

### Navigation Sidebar
- **Logo**: Amazon branding
- **Page Selection**: Navigate between different sections
- **Quick Stats**: Session statistics and metrics

### Main Dashboard Areas
- **Header**: Dynamic title with gradient background
- **Input Forms**: Responsive forms for product information
- **Prediction Results**: Highlighted prediction boxes with confidence
- **Charts**: Interactive Plotly visualizations
- **Data Tables**: Sortable and filterable data displays

### Interactive Elements
- **Sliders**: Adjust ensemble weights and thresholds
- **Buttons**: Trigger predictions and actions
- **Expandable Sections**: Detailed model parameters and history
- **Download Features**: Export prediction history as CSV

## 🎨 Styling & Design

### Color Scheme
- **Primary**: Amazon Orange (#ff9500, #ff7b00)
- **Algorithm Colors**: 
  - LightGBM: #ff6b6b (Red)
  - XGBoost: #4ecdc4 (Teal)
  - CatBoost: #45b7d1 (Blue)
- **Background**: Clean white with subtle gradients
- **Cards**: Soft shadows and rounded corners

### Responsive Design
- **Two-column Layouts**: Optimal use of screen space
- **Adaptive Charts**: Charts resize based on container width
- **Mobile Friendly**: Streamlit's responsive components

## 🔧 Configuration Options

### Model Ensemble Weights
```python
ensemble_weights = {
    'lightgbm': 0.45,  # Best performing model
    'xgboost': 0.35,   # Second best
    'catboost': 0.20   # Third best
}
```

### API Configuration
- **Backend URL**: http://localhost:5000 (Flask backend)
- **Timeout**: 5 seconds for API calls
- **Fallback**: Local prediction if backend unavailable

### Data Sources
- **Training Data**: 75,000 Amazon product samples
- **Features**: Catalog content, product names, descriptions
- **Target**: Product prices in USD

## 📊 Performance Metrics

### Model Accuracy (SMAPE)
- **LightGBM**: 45.68% (Best)
- **XGBoost**: 45.61% (Very Good)
- **CatBoost**: 44.22% (Good)
- **Ensemble**: Weighted average performance

### Processing Time
- **Feature Extraction**: ~0.1-0.5 seconds
- **Model Prediction**: ~0.5-2.0 seconds
- **Total Response**: ~1-3 seconds per prediction

## 🚨 Troubleshooting

### Common Issues

1. **Streamlit not found**:
   ```bash
   pip install streamlit
   ```

2. **Port already in use**:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

3. **Backend connection failed**:
   - Check if Flask backend is running on port 5000
   - Application will use local fallback prediction

4. **Missing dependencies**:
   ```bash
   pip install plotly pandas numpy scikit-learn requests
   ```

### Performance Optimization
- Sample data loading (1000 samples vs 75K for performance)
- Cached predictor initialization
- Efficient chart rendering with Plotly
- Background API calls with fallback

## 🎯 Use Cases

### Business Applications
- **E-commerce Pricing**: Dynamic pricing strategies
- **Market Analysis**: Competitive pricing intelligence
- **Product Valuation**: Automated product assessment
- **Price Optimization**: Data-driven pricing decisions

### Research & Development
- **Algorithm Comparison**: Evaluate different ML approaches
- **Feature Engineering**: Test different feature combinations
- **Model Performance**: Track accuracy improvements
- **Market Research**: Analyze pricing trends and patterns

## 🔮 Future Enhancements

### Planned Features
- **Real-time Data Integration**: Live Amazon product feeds
- **Advanced Filtering**: Category and brand-specific models
- **Batch Processing**: Upload CSV files for bulk predictions
- **A/B Testing**: Compare different model configurations
- **Export Reports**: Generate detailed PDF reports
- **User Authentication**: Multi-user support with saved preferences

### Model Improvements
- **Deep Learning Models**: Neural network implementations
- **Image Processing**: Product image analysis for pricing
- **Sentiment Analysis**: Review sentiment impact on pricing
- **Market Integration**: Real-time market data incorporation

## 📞 Support & Contact

### Application Features
- **Built-in Help**: Tooltips and explanations throughout the app
- **Example Data**: Pre-loaded examples for testing
- **Error Handling**: Graceful fallbacks and error messages
- **Performance Monitoring**: Real-time metrics and statistics

### Technical Details
- **Framework**: Streamlit 1.44.1+
- **Visualization**: Plotly Express & Graph Objects
- **ML Libraries**: scikit-learn, pandas, numpy
- **Backend**: Flask (optional, with fallback)

---

**🎉 Enjoy exploring the Amazon ML Price Predictor Dashboard!**

Navigate through the different sections to discover powerful machine learning insights and make accurate price predictions for Amazon products.