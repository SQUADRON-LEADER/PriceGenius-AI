"""
Complete training pipeline for Smart Product Pricing Challenge
Multi-modal approach combining text and image features
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import download_images
from feature_engineering import TextFeatureExtractor, ImageFeatureExtractor, create_features
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(numerator / denominator) * 100


def load_data():
    """Load training and test data"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load data
    print(f"Loading training data from {config.TRAIN_FILE}...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"Training data shape: {train_df.shape}")
    print(f"Training data columns: {train_df.columns.tolist()}")
    
    print(f"\nLoading test data from {config.TEST_FILE}...")
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Test data shape: {test_df.shape}")
    
    # Check for missing values
    print(f"\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    # Fill missing catalog_content with empty string
    train_df['catalog_content'] = train_df['catalog_content'].fillna('')
    test_df['catalog_content'] = test_df['catalog_content'].fillna('')
    
    # Basic statistics
    print(f"\nPrice statistics:")
    print(train_df['price'].describe())
    
    return train_df, test_df


def download_all_images(train_df, test_df, download=False):
    """Download all images for training and test data"""
    if not download:
        print("\n" + "=" * 80)
        print("SKIPPING IMAGE DOWNLOAD (set download=True to enable)")
        print("=" * 80)
        return
    
    print("\n" + "=" * 80)
    print("DOWNLOADING IMAGES")
    print("=" * 80)
    
    # Combine all image links
    all_image_links = pd.concat([
        train_df['image_link'],
        test_df['image_link']
    ]).unique()
    
    print(f"Total unique images to download: {len(all_image_links)}")
    
    # Download images
    download_images(all_image_links, config.IMAGES_FOLDER)
    
    print("Image download complete!")


def train_model(train_df, use_images=False, n_folds=5):
    """
    Train the complete model with cross-validation
    
    Args:
        train_df: Training dataframe
        use_images: Whether to use image features
        n_folds: Number of cross-validation folds
    
    Returns:
        models: Dictionary of trained models
        text_extractor: Fitted text feature extractor
        scaler: Fitted scaler
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    # Separate features and target
    X = train_df[['catalog_content', 'image_link']].copy()
    y = train_df['price'].values
    
    # Log transform target (helps with price prediction)
    y_log = np.log1p(y)
    
    # Extract text features
    print("\nExtracting text features from catalog_content...")
    text_extractor = TextFeatureExtractor(max_features=5000, n_components=100)
    text_features = text_extractor.fit_transform(X['catalog_content'])
    
    print(f"Text features shape: {text_features.shape}")
    
    # Extract image features if requested
    if use_images:
        print("\nExtracting image features...")
        try:
            image_extractor = ImageFeatureExtractor(config.IMAGES_FOLDER)
            if image_extractor.load_model():
                image_features = image_extractor.extract_features(X['image_link'])
                if image_features is not None:
                    features = pd.concat([
                        text_features.reset_index(drop=True),
                        image_features.reset_index(drop=True)
                    ], axis=1)
                    print(f"Combined features shape: {features.shape}")
                else:
                    features = text_features
                    print("Image features extraction failed, using only text features")
            else:
                features = text_features
                print("TensorFlow not available, using only text features")
        except Exception as e:
            print(f"Error with image features: {e}")
            features = text_features
            print("Using only text features")
    else:
        features = text_features
    
    # Scale features
    print("\nScaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Cross-validation
    print(f"\nStarting {n_folds}-fold cross-validation...")
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
    
    lgb_models = []
    xgb_models = []
    oof_predictions_lgb = np.zeros(len(X_scaled))
    oof_predictions_xgb = np.zeros(len(X_scaled))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled), 1):
        print(f"\n{'=' * 40}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'=' * 40}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        
        # Train LightGBM
        print("Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': config.RANDOM_SEED
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        lgb_models.append(lgb_model)
        
        # Predict for validation
        val_pred_lgb = lgb_model.predict(X_val)
        oof_predictions_lgb[val_idx] = val_pred_lgb
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
            tree_method='hist',
            early_stopping_rounds=50
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        xgb_models.append(xgb_model)
        
        # Predict for validation
        val_pred_xgb = xgb_model.predict(X_val)
        oof_predictions_xgb[val_idx] = val_pred_xgb
        
        # Calculate metrics for this fold
        y_val_orig = y[val_idx]
        val_pred_lgb_orig = np.expm1(val_pred_lgb)
        val_pred_xgb_orig = np.expm1(val_pred_xgb)
        val_pred_ensemble = (val_pred_lgb_orig + val_pred_xgb_orig) / 2
        
        smape_lgb = smape(y_val_orig, val_pred_lgb_orig)
        smape_xgb = smape(y_val_orig, val_pred_xgb_orig)
        smape_ens = smape(y_val_orig, val_pred_ensemble)
        
        print(f"Fold {fold} - LightGBM SMAPE: {smape_lgb:.4f}%")
        print(f"Fold {fold} - XGBoost SMAPE: {smape_xgb:.4f}%")
        print(f"Fold {fold} - Ensemble SMAPE: {smape_ens:.4f}%")
    
    # Overall OOF performance
    print("\n" + "=" * 80)
    print("OVERALL OUT-OF-FOLD PERFORMANCE")
    print("=" * 80)
    
    oof_pred_lgb_orig = np.expm1(oof_predictions_lgb)
    oof_pred_xgb_orig = np.expm1(oof_predictions_xgb)
    oof_pred_ensemble = (oof_pred_lgb_orig + oof_pred_xgb_orig) / 2
    
    overall_smape_lgb = smape(y, oof_pred_lgb_orig)
    overall_smape_xgb = smape(y, oof_pred_xgb_orig)
    overall_smape_ensemble = smape(y, oof_pred_ensemble)
    
    print(f"Overall LightGBM SMAPE: {overall_smape_lgb:.4f}%")
    print(f"Overall XGBoost SMAPE: {overall_smape_xgb:.4f}%")
    print(f"Overall Ensemble SMAPE: {overall_smape_ensemble:.4f}%")
    
    # Save models
    models = {
        'lgb': lgb_models,
        'xgb': xgb_models,
        'text_extractor': text_extractor,
        'scaler': scaler,
        'use_images': use_images
    }
    
    return models


def make_predictions(models, test_df):
    """
    Make predictions on test data
    
    Args:
        models: Dictionary of trained models
        test_df: Test dataframe
    
    Returns:
        predictions: Array of predicted prices
    """
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    # Extract features
    text_extractor = models['text_extractor']
    scaler = models['scaler']
    use_images = models['use_images']
    
    print("Extracting text features from test data...")
    text_features = text_extractor.transform(test_df['catalog_content'])
    
    if use_images:
        print("Extracting image features from test data...")
        try:
            image_extractor = ImageFeatureExtractor(config.IMAGES_FOLDER)
            if image_extractor.load_model():
                image_features = image_extractor.extract_features(test_df['image_link'])
                if image_features is not None:
                    features = pd.concat([
                        text_features.reset_index(drop=True),
                        image_features.reset_index(drop=True)
                    ], axis=1)
                else:
                    features = text_features
        except Exception as e:
            print(f"Error with image features: {e}")
            features = text_features
    else:
        features = text_features
    
    print("Scaling features...")
    X_test = scaler.transform(features)
    
    # Make predictions with all models
    print("Making predictions with LightGBM models...")
    lgb_predictions = []
    for model in models['lgb']:
        pred = model.predict(X_test)
        lgb_predictions.append(np.expm1(pred))
    lgb_pred = np.mean(lgb_predictions, axis=0)
    
    print("Making predictions with XGBoost models...")
    xgb_predictions = []
    for model in models['xgb']:
        pred = model.predict(X_test)
        xgb_predictions.append(np.expm1(pred))
    xgb_pred = np.mean(xgb_predictions, axis=0)
    
    # Ensemble predictions (simple average)
    final_predictions = (lgb_pred + xgb_pred) / 2
    
    # Ensure all predictions are positive
    final_predictions = np.maximum(final_predictions, 0.01)
    
    print(f"Predictions statistics:")
    print(f"  Min: {final_predictions.min():.2f}")
    print(f"  Max: {final_predictions.max():.2f}")
    print(f"  Mean: {final_predictions.mean():.2f}")
    print(f"  Median: {np.median(final_predictions):.2f}")
    
    return final_predictions


def save_submission(test_df, predictions, output_file):
    """Save predictions to submission file"""
    print("\n" + "=" * 80)
    print("SAVING SUBMISSION")
    print("=" * 80)
    
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nFirst few predictions:")
    print(submission.head(10))


def main():
    """Main training and prediction pipeline"""
    print("\n" + "=" * 80)
    print("SMART PRODUCT PRICING CHALLENGE - TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    train_df, test_df = load_data()
    
    # Option to download images (set to False to skip)
    # Only enable if you want to use image features
    USE_IMAGES = False  # Change to True if you want to use images
    
    if USE_IMAGES:
        download_all_images(train_df, test_df, download=False)
    
    # Train model
    models = train_model(train_df, use_images=USE_IMAGES, n_folds=5)
    
    # Save models
    print("\nSaving trained models...")
    model_path = os.path.join(config.MODELS_FOLDER, 'pricing_model.pkl')
    joblib.dump(models, model_path)
    print(f"Models saved to: {model_path}")
    
    # Make predictions
    predictions = make_predictions(models, test_df)
    
    # Save submission
    save_submission(test_df, predictions, config.OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review your submission file: {config.OUTPUT_FILE}")
    print(f"2. Submit to the leaderboard")
    print(f"3. Fill out the documentation template")
    print(f"4. If SMAPE is high, try enabling image features (USE_IMAGES=True)")


if __name__ == "__main__":
    main()
