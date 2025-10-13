"""
Advanced Training Script - High Accuracy Model
Train model with multiple epochs and iterations to achieve 99% accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
import re
import joblib
import os
from datetime import datetime

# Paths
DATASET_PATH = '../dataset'
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.csv')
TEST_FILE = os.path.join(DATASET_PATH, 'test.csv')
OUTPUT_FILE = os.path.join(DATASET_PATH, 'test_out.csv')
MODELS_PATH = '../models'

os.makedirs(MODELS_PATH, exist_ok=True)

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def extract_numeric_features(text):
    """Extract numeric features from text"""
    features = {}
    
    # Extract all numbers
    numbers = re.findall(r'\d+', str(text))
    features['num_count'] = len(numbers)
    features['max_number'] = max([int(n) for n in numbers], default=0)
    features['avg_number'] = np.mean([int(n) for n in numbers]) if numbers else 0
    
    # Text statistics
    features['text_length'] = len(str(text))
    features['word_count'] = len(str(text).split())
    features['unique_words'] = len(set(str(text).lower().split()))
    features['avg_word_length'] = features['text_length'] / max(features['word_count'], 1)
    
    # Special keywords (high-value indicators)
    high_value_keywords = ['premium', 'pro', 'ultra', 'max', 'gaming', 'professional', 
                          'flagship', 'advanced', 'high-end', 'luxury', 'elite']
    features['high_value_score'] = sum(1 for kw in high_value_keywords if kw in str(text).lower())
    
    # Brand indicators
    brands = ['apple', 'samsung', 'sony', 'dell', 'hp', 'lenovo', 'asus', 'msi', 'razer']
    features['brand_score'] = sum(1 for brand in brands if brand in str(text).lower())
    
    # Technical specs
    tech_keywords = ['gb', 'tb', 'ghz', 'ram', 'ssd', 'nvme', 'core', 'processor', 
                    'graphics', 'display', 'screen', 'battery', 'camera']
    features['tech_score'] = sum(1 for kw in tech_keywords if kw in str(text).lower())
    
    # Storage indicators (high correlation with price)
    storage_matches = re.findall(r'(\d+)\s*(gb|tb)', str(text).lower())
    max_storage = 0
    for size, unit in storage_matches:
        size = int(size)
        if unit == 'tb':
            size *= 1024
        max_storage = max(max_storage, size)
    features['max_storage_gb'] = max_storage
    
    # RAM indicators
    ram_matches = re.findall(r'(\d+)\s*gb\s*ram', str(text).lower())
    features['ram_gb'] = max([int(r) for r in ram_matches], default=0)
    
    return features

def advanced_feature_engineering(df, vectorizer=None, svd=None, scaler=None, is_train=True):
    """
    Advanced feature engineering with multiple techniques
    """
    print(f"\n{'='*60}")
    print(f"🔧 Advanced Feature Engineering - {'Training' if is_train else 'Testing'} Set")
    print(f"{'='*60}")
    
    # Fill missing values
    df['catalog_content'] = df['catalog_content'].fillna('')
    
    # 1. TF-IDF Features (MORE features)
    print("📊 Step 1: TF-IDF Vectorization (10,000 features)...")
    if is_train:
        vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            ngram_range=(1, 3),   # Increased from (1, 2)
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        tfidf_features = vectorizer.fit_transform(df['catalog_content'])
    else:
        tfidf_features = vectorizer.transform(df['catalog_content'])
    
    # 2. SVD Dimensionality Reduction
    print("📉 Step 2: SVD Reduction (200 components)...")
    if is_train:
        svd = TruncatedSVD(n_components=200, random_state=42)  # Increased from 100
        text_features = svd.fit_transform(tfidf_features)
    else:
        text_features = svd.transform(tfidf_features)
    
    # 3. Numeric Features
    print("🔢 Step 3: Extracting numeric features...")
    numeric_features_list = []
    for idx, row in df.iterrows():
        features = extract_numeric_features(row['catalog_content'])
        numeric_features_list.append(features)
    
    numeric_df = pd.DataFrame(numeric_features_list)
    
    # 4. Scaling
    print("⚖️  Step 4: Feature scaling...")
    if is_train:
        scaler = RobustScaler()
        numeric_scaled = scaler.fit_transform(numeric_df)
    else:
        numeric_scaled = scaler.transform(numeric_df)
    
    # 5. Combine all features
    print("🔗 Step 5: Combining features...")
    combined_features = np.hstack([text_features, numeric_scaled])
    
    print(f"✅ Final feature shape: {combined_features.shape}")
    print(f"{'='*60}\n")
    
    return combined_features, vectorizer, svd, scaler

def train_lightgbm_advanced(X_train, y_train, X_val, y_val, num_iterations=5000):
    """
    Train LightGBM with advanced hyperparameters and multiple iterations
    """
    print("\n" + "="*60)
    print("🚀 Training Advanced LightGBM Model")
    print("="*60)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,           # Increased from 31
        'learning_rate': 0.03,       # Decreased for better learning
        'feature_fraction': 0.9,     # Increased
        'bagging_fraction': 0.9,     # Increased
        'bagging_freq': 5,
        'max_depth': 12,             # Added depth limit
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,            # L1 regularization
        'lambda_l2': 0.1,            # L2 regularization
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print(f"⏳ Training with {num_iterations} iterations...")
    print(f"📊 Training samples: {len(X_train):,}")
    print(f"📊 Validation samples: {len(X_val):,}")
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_iterations,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=500)
        ]
    )
    
    # Predictions
    val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.expm1(val_pred_log)
    val_true = np.expm1(y_val)
    
    val_smape = smape(val_true, val_pred)
    
    print(f"\n✅ LightGBM Training Complete!")
    print(f"🎯 Best Iteration: {model.best_iteration}")
    print(f"📊 Validation SMAPE: {val_smape:.4f}%")
    print(f"📊 Validation Accuracy: {100 - val_smape:.2f}%")
    print("="*60 + "\n")
    
    return model, val_smape

def train_xgboost_advanced(X_train, y_train, X_val, y_val, num_iterations=5000):
    """
    Train XGBoost with advanced hyperparameters
    """
    print("\n" + "="*60)
    print("🚀 Training Advanced XGBoost Model")
    print("="*60)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,             # Increased
        'learning_rate': 0.03,       # Decreased
        'subsample': 0.9,            # Increased
        'colsample_bytree': 0.9,     # Increased
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,            # L1 regularization
        'reg_lambda': 0.1,           # L2 regularization
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    print(f"⏳ Training with {num_iterations} iterations...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_iterations,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    # Predictions
    val_pred_log = model.predict(dval)
    val_pred = np.expm1(val_pred_log)
    val_true = np.expm1(y_val)
    
    val_smape = smape(val_true, val_pred)
    
    print(f"\n✅ XGBoost Training Complete!")
    print(f"🎯 Best Iteration: {model.best_iteration}")
    print(f"📊 Validation SMAPE: {val_smape:.4f}%")
    print(f"📊 Validation Accuracy: {100 - val_smape:.2f}%")
    print("="*60 + "\n")
    
    return model, val_smape

def ensemble_predictions(models, X, weights=None):
    """
    Ensemble multiple models with weighted averaging
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    predictions = []
    for model, weight in zip(models, weights):
        if isinstance(model, lgb.Booster):
            pred = model.predict(X)
        else:  # XGBoost
            dmatrix = xgb.DMatrix(X)
            pred = model.predict(dmatrix)
        predictions.append(pred * weight)
    
    return np.sum(predictions, axis=0) / sum(weights)

def main():
    """
    Main training function with multiple epochs
    """
    print("\n" + "="*70)
    print("🎯 ADVANCED MODEL TRAINING - HIGH ACCURACY MODE")
    print("="*70)
    print("Target: 99% Accuracy (SMAPE < 1%)")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"✅ Training samples: {len(train_df):,}")
    print(f"✅ Test samples: {len(test_df):,}")
    print(f"✅ Price range: ₹{train_df['price'].min():,.0f} - ₹{train_df['price'].max():,.0f}")
    print(f"✅ Average price: ₹{train_df['price'].mean():,.0f}")
    
    # Feature engineering
    print("\n🔧 Starting advanced feature engineering...")
    X_train, vectorizer, svd, scaler = advanced_feature_engineering(
        train_df, is_train=True
    )
    X_test, _, _, _ = advanced_feature_engineering(
        test_df, vectorizer, svd, scaler, is_train=False
    )
    
    # Target variable (log transform)
    y_train = np.log1p(train_df['price'].values)
    
    # Split for validation
    print("📊 Creating train/validation split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"✅ Training set: {len(X_tr):,} samples")
    print(f"✅ Validation set: {len(X_val):,} samples")
    
    # Train multiple models
    print("\n" + "="*70)
    print("🚂 TRAINING PHASE - MULTIPLE MODELS")
    print("="*70 + "\n")
    
    models = []
    scores = []
    
    # Train LightGBM (5000 iterations)
    lgb_model, lgb_score = train_lightgbm_advanced(X_tr, y_tr, X_val, y_val, num_iterations=5000)
    models.append(('LightGBM', lgb_model, lgb_score))
    scores.append(lgb_score)
    
    # Train XGBoost (5000 iterations)
    xgb_model, xgb_score = train_xgboost_advanced(X_tr, y_tr, X_val, y_val, num_iterations=5000)
    models.append(('XGBoost', xgb_model, xgb_score))
    scores.append(xgb_score)
    
    # Ensemble predictions
    print("\n" + "="*70)
    print("🎭 ENSEMBLE PREDICTIONS")
    print("="*70)
    
    # Calculate weights based on inverse SMAPE (better models get more weight)
    weights = [1/score for score in scores]
    weights = [w/sum(weights) for w in weights]  # Normalize
    
    print("📊 Model weights:")
    for (name, _, score), weight in zip(models, weights):
        print(f"   {name}: {weight:.2%} (SMAPE: {score:.4f}%)")
    
    # Ensemble validation predictions
    ensemble_pred_log = ensemble_predictions(
        [model for _, model, _ in models], X_val, weights
    )
    ensemble_pred = np.expm1(ensemble_pred_log)
    val_true = np.expm1(y_val)
    
    ensemble_smape = smape(val_true, ensemble_pred)
    ensemble_accuracy = 100 - ensemble_smape
    
    print(f"\n🎯 Ensemble SMAPE: {ensemble_smape:.4f}%")
    print(f"🎯 Ensemble Accuracy: {ensemble_accuracy:.2f}%")
    print("="*70 + "\n")
    
    # Show prediction examples
    print("📊 Sample Predictions vs Actual:")
    print("-" * 50)
    for i in range(min(10, len(val_true))):
        actual = val_true[i]
        predicted = ensemble_pred[i]
        error = abs(actual - predicted) / actual * 100
        print(f"Actual: ₹{actual:>10,.2f} | Predicted: ₹{predicted:>10,.2f} | Error: {error:>6.2f}%")
    print("-" * 50 + "\n")
    
    # Train final model on full training data
    print("🚀 Training final ensemble on FULL training data...")
    
    # Retrain LightGBM
    print("\n1️⃣ Retraining LightGBM on full data...")
    train_data_full = lgb.Dataset(X_train, label=y_train)
    final_lgb = lgb.train(
        lgb_model.params,
        train_data_full,
        num_boost_round=lgb_model.best_iteration
    )
    
    # Retrain XGBoost
    print("2️⃣ Retraining XGBoost on full data...")
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    final_xgb = xgb.train(
        xgb_params,
        dtrain_full,
        num_boost_round=xgb_model.best_iteration
    )
    
    # Generate test predictions
    print("\n🎯 Generating final test predictions...")
    test_pred_log = ensemble_predictions(
        [final_lgb, final_xgb], X_test, weights
    )
    test_predictions = np.expm1(test_pred_log)
    
    # Create submission
    print("💾 Creating submission file...")
    submission = test_df.copy()
    submission['price'] = test_predictions
    submission.to_csv(OUTPUT_FILE, index=False)
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(final_lgb, os.path.join(MODELS_PATH, 'lightgbm_advanced.pkl'))
    joblib.dump(final_xgb, os.path.join(MODELS_PATH, 'xgboost_advanced.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_PATH, 'vectorizer.pkl'))
    joblib.dump(svd, os.path.join(MODELS_PATH, 'svd.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_PATH, 'scaler.pkl'))
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Final Validation SMAPE: {ensemble_smape:.4f}%")
    print(f"🎯 Final Accuracy: {ensemble_accuracy:.2f}%")
    print(f"💾 Submission file: {OUTPUT_FILE}")
    print(f"💾 Models saved in: {MODELS_PATH}/")
    print(f"📊 Test predictions: {len(test_predictions):,}")
    print(f"📊 Price range: ₹{test_predictions.min():,.2f} - ₹{test_predictions.max():,.2f}")
    print(f"📊 Average predicted price: ₹{test_predictions.mean():,.2f}")
    print("="*70)
    
    # Show gaming laptop example
    gaming_laptop_text = "High-performance gaming laptop with RTX 4080, 32GB RAM, 1TB SSD, 15.6 inch 240Hz display"
    print(f"\n🎮 Example: Gaming Laptop Price Prediction")
    print("-" * 70)
    print(f"Description: {gaming_laptop_text}")
    
    # Make prediction
    example_df = pd.DataFrame({'catalog_content': [gaming_laptop_text]})
    X_example, _, _, _ = advanced_feature_engineering(
        example_df, vectorizer, svd, scaler, is_train=False
    )
    example_pred_log = ensemble_predictions([final_lgb, final_xgb], X_example, weights)
    example_price = np.expm1(example_pred_log)[0]
    
    print(f"Predicted Price: ₹{example_price:,.2f}")
    print("-" * 70 + "\n")
    
    return ensemble_smape, ensemble_accuracy

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"🕐 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    smape_score, accuracy = main()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n🕐 Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {duration/60:.2f} minutes")
    print(f"\n{'='*70}")
    print(f"🎉 MISSION {'ACCOMPLISHED' if accuracy >= 99 else 'IN PROGRESS'}!")
    print(f"{'='*70}\n")
