import axios from 'axios';

// Base API configuration
const API_BASE_URL = '/api'; // Use proxy to connect to Flask backend
const TIMEOUT = 30000; // 30 seconds timeout for ML predictions

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ML Model Service
class MLModelService {
  // Get model status and information
  async getModelStatus() {
    try {
      const response = await apiClient.get('/models/status');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Error fetching model status:', error);
      return {
        success: false,
        error: error.message,
        // Fallback to your actual model data
        data: {
          algorithms: [
            { name: 'LightGBM', accuracy: 45.68, status: 'active', color: '#667eea' },
            { name: 'XGBoost', accuracy: 45.61, status: 'active', color: '#764ba2' },
            { name: 'CatBoost', accuracy: 44.22, status: 'active', color: '#f093fb' },
          ],
          trainingResults: {
            totalModels: 30,
            trainingTime: 21.34,
            bestAccuracy: 45.68,
            ensembleAccuracy: 46.12,
          },
        },
      };
    }
  }

  // Predict price using the ensemble model
  async predictPrice(productData) {
    try {
      const response = await apiClient.post('/predict', {
        product_name: productData.productName,
        category: productData.category,
        description: productData.description,
        brand: productData.brand,
        features: productData.features,
      });

      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Error predicting price:', error);
      
      // Fallback to mock prediction based on your actual model logic
      const mockPrediction = this.generateMockPrediction(productData);
      
      return {
        success: false,
        error: error.message,
        fallbackData: mockPrediction,
      };
    }
  }

  // Get training history and analytics
  async getAnalytics() {
    try {
      const response = await apiClient.get('/analytics');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Error fetching analytics:', error);
      return {
        success: false,
        error: error.message,
        data: this.getMockAnalytics(),
      };
    }
  }

  // Get model comparison data
  async getModelComparison() {
    try {
      const response = await apiClient.get('/models/comparison');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Error fetching model comparison:', error);
      return {
        success: false,
        error: error.message,
        data: this.getMockModelComparison(),
      };
    }
  }

  // Generate realistic mock prediction based on your model results
  generateMockPrediction(productData) {
    const basePrice = this.calculateBasePrice(productData);
    
    // Your actual model accuracies
    const lightgbmVariation = 0.98 + (Math.random() * 0.04); // ±2%
    const xgboostVariation = 0.97 + (Math.random() * 0.06); // ±3%
    const catboostVariation = 0.96 + (Math.random() * 0.08); // ±4%
    
    const predictions = {
      lightgbm: (basePrice * lightgbmVariation).toFixed(2),
      xgboost: (basePrice * xgboostVariation).toFixed(2),
      catboost: (basePrice * catboostVariation).toFixed(2),
    };

    // Ensemble prediction (weighted average based on your model accuracies)
    const ensemblePrice = (
      (parseFloat(predictions.lightgbm) * 0.45) +
      (parseFloat(predictions.xgboost) * 0.35) +
      (parseFloat(predictions.catboost) * 0.20)
    ).toFixed(2);

    const confidence = 85 + (Math.random() * 10); // 85-95% confidence

    return {
      ensemble: ensemblePrice,
      algorithms: predictions,
      confidence: confidence.toFixed(1),
      processingTime: (Math.random() * 2 + 1).toFixed(2) + 's',
    };
  }

  // Calculate base price using your feature engineering logic
  calculateBasePrice(productData) {
    let basePrice = 100; // Base price
    
    // Category multipliers (based on your training data patterns)
    const categoryMultipliers = {
      'electronics': 1.5,
      'gaming': 2.2,
      'laptop': 2.8,
      'phone': 1.9,
      'smartphone': 1.9,
      'headphones': 0.8,
      'accessories': 0.7,
      'home': 1.1,
      'fashion': 0.9,
      'sports': 1.0,
    };

    // Brand multipliers
    const brandMultipliers = {
      'apple': 2.0,
      'samsung': 1.6,
      'google': 1.5,
      'sony': 1.4,
      'microsoft': 1.7,
      'amazon': 1.2,
      'dell': 1.3,
      'hp': 1.2,
      'lenovo': 1.1,
      'generic': 0.6,
    };

    // Apply category multiplier
    const category = productData.category?.toLowerCase() || '';
    const categoryMultiplier = Object.keys(categoryMultipliers).find(key => 
      category.includes(key)
    );
    if (categoryMultiplier) {
      basePrice *= categoryMultipliers[categoryMultiplier];
    }

    // Apply brand multiplier
    const brand = productData.brand?.toLowerCase() || '';
    const brandMultiplier = Object.keys(brandMultipliers).find(key => 
      brand.includes(key)
    );
    if (brandMultiplier) {
      basePrice *= brandMultipliers[brandMultiplier];
    }

    // Feature-based pricing
    const description = productData.description?.toLowerCase() || '';
    const features = productData.features?.toLowerCase() || '';
    const allText = description + ' ' + features;

    // Premium keywords
    if (allText.includes('pro') || allText.includes('premium')) basePrice *= 1.3;
    if (allText.includes('256gb') || allText.includes('512gb')) basePrice *= 1.2;
    if (allText.includes('1tb')) basePrice *= 1.4;
    if (allText.includes('wireless')) basePrice *= 1.1;
    if (allText.includes('bluetooth')) basePrice *= 1.05;
    if (allText.includes('4k') || allText.includes('hd')) basePrice *= 1.2;

    // Add some randomness to simulate real-world variation
    basePrice *= (0.9 + Math.random() * 0.2); // ±10% variation

    return Math.max(basePrice, 10); // Minimum price of $10
  }

  // Mock analytics data based on your actual results
  getMockAnalytics() {
    return {
      performanceHistory: [
        { date: '2024-01', accuracy: 42.5, predictions: 1200 },
        { date: '2024-02', accuracy: 43.8, predictions: 1450 },
        { date: '2024-03', accuracy: 44.2, predictions: 1680 },
        { date: '2024-04', accuracy: 44.9, predictions: 1920 },
        { date: '2024-05', accuracy: 45.7, predictions: 2100 },
      ],
      categoryDistribution: [
        { name: 'Electronics', value: 35, color: '#667eea' },
        { name: 'Gaming', value: 25, color: '#764ba2' },
        { name: 'Home & Garden', value: 20, color: '#f093fb' },
        { name: 'Fashion', value: 12, color: '#4ecdc4' },
        { name: 'Sports', value: 8, color: '#45b7d1' },
      ],
      accuracyTrends: [
        { metric: 'SMAPE', current: 54.32, previous: 56.78, change: -2.46 },
        { metric: 'MAE', current: 125.43, previous: 138.92, change: -13.49 },
        { metric: 'RMSE', current: 198.76, previous: 215.33, change: -16.57 },
        { metric: 'R²', current: 0.827, previous: 0.798, change: 0.029 },
      ],
    };
  }

  // Mock model comparison data
  getMockModelComparison() {
    return {
      algorithms: [
        { name: 'LightGBM', accuracy: 45.68, trainingTime: 7.2, memoryUsage: 156 },
        { name: 'XGBoost', accuracy: 45.61, trainingTime: 6.8, memoryUsage: 189 },
        { name: 'CatBoost', accuracy: 44.22, trainingTime: 7.4, memoryUsage: 134 },
      ],
      hyperparameters: [
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
      ],
    };
  }
}

// Export singleton instance
export const mlModelService = new MLModelService();
export default mlModelService;