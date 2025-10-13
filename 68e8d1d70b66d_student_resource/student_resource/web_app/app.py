"""
Amazon ML Challenge - Product Price Prediction Web Application
Beautiful and Advanced GUI for testing the trained model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import os
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'amazon-ml-challenge-2025'

# Global variables for model and transformers
model = None
vectorizer = None
svd = None
model_loaded = False

# Load training data for fitting vectorizer
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.csv')

def load_model():
    """Load the trained model and prepare transformers"""
    global model, vectorizer, svd, model_loaded
    
    try:
        print("Loading training data (sample)...")
        train_df = pd.read_csv(TRAIN_FILE)
        # Use only 5000 samples for faster loading
        train_df = train_df.sample(n=5000, random_state=42)
        train_df['catalog_content'] = train_df['catalog_content'].fillna('')
        
        # Fit TF-IDF vectorizer
        print("Fitting TF-IDF vectorizer (fast mode)...")
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))  # Simplified
        tfidf_features = vectorizer.fit_transform(train_df['catalog_content'])
        
        # Fit SVD
        print("Fitting SVD...")
        svd = TruncatedSVD(n_components=50, random_state=42)  # Reduced dimensions
        X_train = svd.fit_transform(tfidf_features)
        y_train = np.log1p(train_df['price'].values)
        
        # Train model
        print("Training model (quick training)...")
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'verbose': -1
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=100)  # Fewer iterations
        
        model_loaded = True
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_price(catalog_content):
    """Predict price for given catalog content"""
    global model, vectorizer, svd
    
    if not model_loaded:
        return None, "Model not loaded"
    
    try:
        # Transform text
        tfidf = vectorizer.transform([catalog_content])
        features = svd.transform(tfidf)
        
        # Predict
        log_price = model.predict(features)[0]
        price = np.expm1(log_price)
        
        return float(price), None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        catalog_content = data.get('catalog_content', '')
        
        if not catalog_content:
            return jsonify({
                'success': False,
                'error': 'Catalog content is required'
            })
        
        # Predict price
        predicted_price, error = predict_price(catalog_content)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        catalog_contents = data.get('catalog_contents', [])
        
        if not catalog_contents:
            return jsonify({
                'success': False,
                'error': 'Catalog contents are required'
            })
        
        predictions = []
        for content in catalog_contents:
            price, error = predict_price(content)
            if error:
                predictions.append({'error': error})
            else:
                predictions.append({'price': round(price, 2)})
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stats')
def stats():
    """Get model statistics"""
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        
        stats = {
            'total_samples': len(train_df),
            'price_stats': {
                'min': float(train_df['price'].min()),
                'max': float(train_df['price'].max()),
                'mean': float(train_df['price'].mean()),
                'median': float(train_df['price'].median()),
                'std': float(train_df['price'].std())
            },
            'model_loaded': model_loaded
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Amazon ML Challenge - Price Prediction Web App")
    print("=" * 60)
    
    # Load model on startup
    load_model()
    
    print("\n✅ Starting web server...")
    print("🌐 Access the app at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
