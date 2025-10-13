"""
Multi-Modal Training Script - Text + Images
Train model with both catalog content (text) and product images to achieve 99% accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import cv2
from PIL import Image
import requests
from io import BytesIO
import re
import joblib
import os
import warnings
from datetime import datetime
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = '../dataset'
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.csv')
TEST_FILE = os.path.join(DATASET_PATH, 'test.csv')
OUTPUT_FILE = os.path.join(DATASET_PATH, 'test_out.csv')
MODELS_PATH = '../models'
IMAGES_PATH = '../images'

# Create directories
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Constants
RANDOM_SEED = 42
SAMPLE_SIZE = 10000  # Use subset for faster image processing
IMAGE_SIZE = (224, 224)
MAX_IMAGES = 5000  # Limit images to download

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def download_image(url, timeout=5):
    """Download image from URL with error handling"""
    try:
        if pd.isna(url) or not url.strip():
            return None
        
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img.convert('RGB')
    except:
        pass
    return None

def extract_image_features(img):
    """Extract features from image using simple computer vision"""
    if img is None:
        return np.zeros(20)  # Return zeros if no image
    
    try:
        # Resize image
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized)
        
        # Basic color features
        mean_rgb = np.mean(img_array, axis=(0, 1))  # Average RGB
        std_rgb = np.std(img_array, axis=(0, 1))    # RGB standard deviation
        
        # Convert to HSV for more features
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        mean_hsv = np.mean(img_cv, axis=(0, 1))
        std_hsv = np.std(img_cv, axis=(0, 1))
        
        # Brightness and contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge detection (complexity measure)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Combine all features
        features = np.concatenate([
            mean_rgb,      # 3 features
            std_rgb,       # 3 features
            mean_hsv,      # 3 features
            std_hsv,       # 3 features
            [brightness, contrast, edge_density],  # 3 features
            [img_array.shape[0], img_array.shape[1]],  # 2 features (dimensions)
            [np.mean(img_array), np.std(img_array), np.min(img_array), np.max(img_array)]  # 4 features
        ])
        
        return features
    except:
        return np.zeros(20)

def extract_text_features(text):
    """Extract advanced text features"""
    features = {}
    
    # Basic text stats
    text_str = str(text).lower()
    features['text_length'] = len(text_str)
    features['word_count'] = len(text_str.split())
    features['unique_words'] = len(set(text_str.split()))
    features['avg_word_length'] = features['text_length'] / max(features['word_count'], 1)
    
    # Extract numbers
    numbers = re.findall(r'\d+', text_str)
    features['number_count'] = len(numbers)
    features['max_number'] = max([int(n) for n in numbers], default=0)
    features['avg_number'] = np.mean([int(n) for n in numbers]) if numbers else 0
    
    # High-value keywords
    high_value_keywords = ['premium', 'pro', 'ultra', 'max', 'gaming', 'professional', 
                          'flagship', 'advanced', 'high-end', 'luxury', 'elite', 'deluxe']
    features['premium_score'] = sum(1 for kw in high_value_keywords if kw in text_str)
    
    # Brand indicators
    brands = ['apple', 'samsung', 'sony', 'dell', 'hp', 'lenovo', 'asus', 'msi', 'razer', 
              'nvidia', 'intel', 'amd', 'microsoft', 'google', 'amazon', 'lg', 'canon', 'nikon']
    features['brand_score'] = sum(1 for brand in brands if brand in text_str)
    
    # Technical specifications
    tech_keywords = ['gb', 'tb', 'ghz', 'mhz', 'ram', 'ssd', 'hdd', 'nvme', 'core', 'processor', 
                    'graphics', 'gpu', 'cpu', 'display', 'screen', 'monitor', 'camera', 'lens',
                    'battery', 'mah', 'wireless', 'bluetooth', '4k', '8k', 'hd', 'fhd']
    features['tech_score'] = sum(1 for kw in tech_keywords if kw in text_str)
    
    # Storage capacity
    storage_matches = re.findall(r'(\d+)\s*(gb|tb)', text_str)
    max_storage = 0
    for size, unit in storage_matches:
        size = int(size)
        if unit == 'tb':
            size *= 1024
        max_storage = max(max_storage, size)
    features['max_storage_gb'] = max_storage
    
    # RAM capacity
    ram_matches = re.findall(r'(\d+)\s*gb\s*ram', text_str)
    features['ram_gb'] = max([int(r) for r in ram_matches], default=0)
    
    # Screen size
    screen_matches = re.findall(r'(\d+(?:\.\d+)?)\s*inch', text_str)
    features['screen_size'] = max([float(s) for s in screen_matches], default=0)
    
    return features

def multimodal_feature_engineering(df, vectorizer=None, svd=None, scaler=None, 
                                 text_scaler=None, is_train=True, use_images=True):
    """
    Advanced multimodal feature engineering
    """
    print(f"\n{'='*70}")
    print(f"🔧 MULTIMODAL FEATURE ENGINEERING - {'Training' if is_train else 'Testing'}")
    print(f"{'='*70}")
    
    # Sample data for faster processing
    if is_train and len(df) > SAMPLE_SIZE:
        print(f"📊 Sampling {SAMPLE_SIZE} rows for faster training...")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Fill missing values
    df['catalog_content'] = df['catalog_content'].fillna('')
    df['image_link'] = df['image_link'].fillna('')
    
    print(f"📊 Processing {len(df):,} samples...")
    
    # 1. Text Features
    print("📝 Step 1: Processing text features...")
    
    # TF-IDF Features
    if is_train:
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        tfidf_features = vectorizer.fit_transform(df['catalog_content'])
    else:
        tfidf_features = vectorizer.transform(df['catalog_content'])
    
    # SVD reduction
    if is_train:
        svd = TruncatedSVD(n_components=150, random_state=RANDOM_SEED)
        text_features = svd.fit_transform(tfidf_features)
    else:
        text_features = svd.transform(tfidf_features)
    
    # Additional text features
    text_stats = []
    for _, row in df.iterrows():
        features = extract_text_features(row['catalog_content'])
        text_stats.append(list(features.values()))
    
    text_stats_df = pd.DataFrame(text_stats)
    
    # Scale text features
    if is_train:
        text_scaler = StandardScaler()
        text_stats_scaled = text_scaler.fit_transform(text_stats_df)
    else:
        text_stats_scaled = text_scaler.transform(text_stats_df)
    
    print(f"✅ Text features: {text_features.shape[1]} TF-IDF + {text_stats_scaled.shape[1]} stats")
    
    # 2. Image Features (if enabled)
    if use_images:
        print("🖼️  Step 2: Processing image features...")
        
        image_features_list = []
        processed_count = 0
        max_to_process = min(MAX_IMAGES, len(df)) if is_train else len(df)
        
        for i, image_url in enumerate(tqdm(df['image_link'], desc="Processing images")):
            if processed_count >= max_to_process:
                # Use zero features for remaining images
                image_features_list.append(np.zeros(20))
                continue
                
            img = download_image(image_url)
            features = extract_image_features(img)
            image_features_list.append(features)
            
            if img is not None:
                processed_count += 1
        
        image_features = np.array(image_features_list)
        
        # Scale image features
        if is_train:
            scaler = StandardScaler()
            image_features_scaled = scaler.fit_transform(image_features)
        else:
            image_features_scaled = scaler.transform(image_features)
        
        print(f"✅ Image features: {image_features_scaled.shape[1]} features from {processed_count} images")
        
        # Combine all features
        combined_features = np.hstack([text_features, text_stats_scaled, image_features_scaled])
    else:
        print("⏭️  Step 2: Skipping image processing (text-only mode)")
        combined_features = np.hstack([text_features, text_stats_scaled])
        scaler = None
    
    print(f"🔗 Final feature shape: {combined_features.shape}")
    print(f"{'='*70}\n")
    
    return combined_features, vectorizer, svd, scaler, text_scaler, df

def train_advanced_model(X_train, y_train, X_val, y_val, model_type='lightgbm', iterations=3000):
    """Train advanced model with hyperparameter optimization"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    if model_type == 'lightgbm':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': RANDOM_SEED
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=iterations,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=200)
            ]
        )
        
        val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        
    else:  # XGBoost
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'tree_method': 'hist'
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=iterations,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=200,
            verbose_eval=200
        )
        
        val_pred_log = model.predict(dval)
    
    # Calculate metrics
    val_pred = np.expm1(val_pred_log)
    val_true = np.expm1(y_val)
    val_smape = smape(val_true, val_pred)
    val_accuracy = 100 - val_smape
    
    print(f"\n✅ {model_type.upper()} Training Complete!")
    print(f"🎯 Best Iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
    print(f"📊 Validation SMAPE: {val_smape:.4f}%")
    print(f"📊 Validation Accuracy: {val_accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    return model, val_smape, val_accuracy

def main():
    """Main training function"""
    print("\n" + "="*80)
    print("🎯 MULTIMODAL TRAINING - TEXT + IMAGES")
    print("🚀 Target: 99% Accuracy (SMAPE < 1%)")
    print("="*80 + "\n")
    
    # Load data
    print("📂 Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"✅ Training samples: {len(train_df):,}")
    print(f"✅ Test samples: {len(test_df):,}")
    print(f"✅ Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():,.2f}")
    print(f"✅ Average price: ${train_df['price'].mean():.2f}")
    
    # Check for images
    has_images = train_df['image_link'].notna().sum()
    print(f"✅ Samples with images: {has_images:,} ({has_images/len(train_df)*100:.1f}%)")
    
    # Feature engineering
    print("\n🔧 Starting multimodal feature engineering...")
    
    # Process training data
    X_train, vectorizer, svd, img_scaler, text_scaler, train_sample = multimodal_feature_engineering(
        train_df, is_train=True, use_images=True
    )
    
    # Process test data (sample for speed)
    test_sample = test_df.sample(n=min(1000, len(test_df)), random_state=RANDOM_SEED)
    X_test, _, _, _, _, _ = multimodal_feature_engineering(
        test_sample, vectorizer, svd, img_scaler, text_scaler, is_train=False, use_images=True
    )
    
    # Target variable
    y_train = np.log1p(train_sample['price'].values)
    
    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"📊 Training set: {len(X_tr):,} samples")
    print(f"📊 Validation set: {len(X_val):,} samples")
    print(f"📊 Feature dimensions: {X_train.shape[1]}")
    
    # Train multiple models
    models = []
    scores = []
    
    # Train LightGBM
    lgb_model, lgb_smape, lgb_acc = train_advanced_model(
        X_tr, y_tr, X_val, y_val, 'lightgbm', iterations=4000
    )
    models.append(('LightGBM', lgb_model, lgb_smape))
    scores.append(lgb_smape)
    
    # Train XGBoost
    xgb_model, xgb_smape, xgb_acc = train_advanced_model(
        X_tr, y_tr, X_val, y_val, 'xgboost', iterations=4000
    )
    models.append(('XGBoost', xgb_model, xgb_smape))
    scores.append(xgb_smape)
    
    # Ensemble predictions
    print("\n" + "="*70)
    print("🎭 ENSEMBLE PREDICTIONS")
    print("="*70)
    
    weights = [1/score for score in scores]
    weights = [w/sum(weights) for w in weights]
    
    print("📊 Model weights:")
    for (name, _, score), weight in zip(models, weights):
        accuracy = 100 - score
        print(f"   {name}: {weight:.2%} (SMAPE: {score:.4f}%, Accuracy: {accuracy:.2f}%)")
    
    # Ensemble validation predictions
    val_preds = []
    for name, model, _ in models:
        if name == 'LightGBM':
            pred = model.predict(X_val, num_iteration=model.best_iteration)
        else:
            dval = xgb.DMatrix(X_val)
            pred = model.predict(dval)
        val_preds.append(pred)
    
    ensemble_pred_log = np.average(val_preds, axis=0, weights=weights)
    ensemble_pred = np.expm1(ensemble_pred_log)
    val_true = np.expm1(y_val)
    
    ensemble_smape = smape(val_true, ensemble_pred)
    ensemble_accuracy = 100 - ensemble_smape
    
    print(f"\n🎯 Ensemble SMAPE: {ensemble_smape:.4f}%")
    print(f"🎯 Ensemble Accuracy: {ensemble_accuracy:.2f}%")
    print("="*70)
    
    # Show sample predictions
    print("\n📊 Sample Predictions:")
    print("-" * 60)
    for i in range(min(10, len(val_true))):
        actual = val_true[i]
        predicted = ensemble_pred[i]
        error = abs(actual - predicted) / actual * 100
        print(f"Actual: ${actual:>8.2f} | Predicted: ${predicted:>8.2f} | Error: {error:>6.2f}%")
    print("-" * 60)
    
    # Test predictions (on sample)
    print(f"\n🎯 Generating test predictions on {len(test_sample):,} samples...")
    test_preds = []
    for name, model, _ in models:
        if name == 'LightGBM':
            pred = model.predict(X_test)
        else:
            dtest = xgb.DMatrix(X_test)
            pred = model.predict(dtest)
        test_preds.append(pred)
    
    test_pred_log = np.average(test_preds, axis=0, weights=weights)
    test_predictions = np.expm1(test_pred_log)
    
    # Create submission for sample
    submission = test_sample.copy()
    submission['price'] = test_predictions
    sample_output = OUTPUT_FILE.replace('.csv', '_sample.csv')
    submission.to_csv(sample_output, index=False)
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(lgb_model, os.path.join(MODELS_PATH, 'multimodal_lightgbm.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_PATH, 'multimodal_xgboost.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_PATH, 'multimodal_vectorizer.pkl'))
    joblib.dump(svd, os.path.join(MODELS_PATH, 'multimodal_svd.pkl'))
    joblib.dump(img_scaler, os.path.join(MODELS_PATH, 'multimodal_img_scaler.pkl'))
    joblib.dump(text_scaler, os.path.join(MODELS_PATH, 'multimodal_text_scaler.pkl'))
    
    # Final results
    print("\n" + "="*70)
    print("✅ MULTIMODAL TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Final Ensemble SMAPE: {ensemble_smape:.4f}%")
    print(f"🎯 Final Accuracy: {ensemble_accuracy:.2f}%")
    print(f"💾 Sample submission: {sample_output}")
    print(f"💾 Models saved in: {MODELS_PATH}/")
    print(f"📊 Features used: Text + Images ({X_train.shape[1]} total)")
    print(f"📊 Training samples: {len(train_sample):,}")
    print(f"📊 Test predictions: {len(test_predictions):,}")
    print(f"📊 Price range: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")
    print("="*70)
    
    # Example prediction
    print(f"\n🎮 Example: Gaming Laptop Price Prediction")
    print("-" * 70)
    example_text = "High-performance gaming laptop with RTX 4080, 32GB RAM, 1TB SSD, 15.6 inch 240Hz display"
    example_img = "https://example.com/gaming-laptop.jpg"  # Dummy URL
    
    example_df = pd.DataFrame({
        'catalog_content': [example_text],
        'image_link': [example_img]
    })
    
    X_example, _, _, _, _, _ = multimodal_feature_engineering(
        example_df, vectorizer, svd, img_scaler, text_scaler, is_train=False, use_images=True
    )
    
    example_preds = []
    for name, model, _ in models:
        if name == 'LightGBM':
            pred = model.predict(X_example)
        else:
            dexample = xgb.DMatrix(X_example)
            pred = model.predict(dexample)
        example_preds.append(pred)
    
    example_pred_log = np.average(example_preds, axis=0, weights=weights)
    example_price = np.expm1(example_pred_log)[0]
    
    print(f"Description: {example_text}")
    print(f"Predicted Price: ${example_price:,.2f}")
    print("-" * 70 + "\n")
    
    return ensemble_smape, ensemble_accuracy

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"🕐 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        smape_score, accuracy = main()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🕐 Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total duration: {duration/60:.2f} minutes")
        
        print(f"\n{'='*70}")
        if accuracy >= 99:
            print(f"🎉 MISSION ACCOMPLISHED! 99% ACCURACY ACHIEVED!")
        elif accuracy >= 90:
            print(f"🎯 EXCELLENT PERFORMANCE! {accuracy:.1f}% ACCURACY!")
        elif accuracy >= 80:
            print(f"🚀 GOOD PERFORMANCE! {accuracy:.1f}% ACCURACY!")
        else:
            print(f"📈 TRAINING COMPLETE! {accuracy:.1f}% ACCURACY - ROOM FOR IMPROVEMENT!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()