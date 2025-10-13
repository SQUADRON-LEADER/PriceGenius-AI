"""
Main training pipeline for product pricing model
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from feature_engineering import create_features, TextFeatureExtractor, ImageFeatureExtractor
from model import PricePredictionModel
from utils import download_images
import joblib
import warnings
warnings.filterwarnings('ignore')


def main():
    print("="*70)
    print("SMART PRODUCT PRICING - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading training data...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"Training data shape: {train_df.shape}")
    print(f"Price statistics:")
    print(train_df['price'].describe())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(train_df.isnull().sum())
    
    # Step 2: Optional - Download images (if you want to use image features)
    use_images = False  # Set to True if you want to use image features
    
    if use_images:
        print("\n[Step 2] Downloading product images...")
        print("Note: This may take a while. You may need to retry if some downloads fail.")
        
        response = input("Do you want to download images? (y/n): ")
        if response.lower() == 'y':
            download_images(train_df['image_link'].tolist(), config.IMAGES_FOLDER)
            print(f"Images downloaded to {config.IMAGES_FOLDER}")
        else:
            print("Skipping image download. Model will use only text features.")
            use_images = False
    
    # Step 3: Feature extraction
    print("\n[Step 3] Extracting features...")
    
    # Initialize extractors
    text_extractor = None
    image_extractor = ImageFeatureExtractor(config.IMAGES_FOLDER) if use_images else None
    
    # Create features
    X_train, text_extractor = create_features(
        train_df,
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        is_training=True,
        use_images=use_images
    )
    
    y_train = train_df['price']
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Save feature extractor
    extractor_path = os.path.join(config.MODELS_FOLDER, 'text_extractor.pkl')
    joblib.dump(text_extractor, extractor_path)
    print(f"Text extractor saved to {extractor_path}")
    
    # Step 4: Handle outliers (optional - conservative approach)
    print("\n[Step 4] Analyzing price distribution...")
    
    # Calculate price statistics
    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr  # Using 3*IQR for conservative outlier detection
    upper_bound = q3 + 3 * iqr
    
    outliers = ((y_train < lower_bound) | (y_train > upper_bound)).sum()
    print(f"Potential outliers detected: {outliers} ({outliers/len(y_train)*100:.2f}%)")
    print(f"Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    print(f"IQR bounds: ${lower_bound:.2f} - ${upper_bound:.2f}")
    
    # Option to handle outliers
    handle_outliers = input("\nDo you want to apply log transformation to handle price skewness? (y/n): ")
    use_log_transform = handle_outliers.lower() == 'y'
    
    if use_log_transform:
        print("Applying log transformation to target variable...")
        y_train_transformed = np.log1p(y_train)
    else:
        y_train_transformed = y_train
    
    # Step 5: Train model
    print("\n[Step 5] Training model...")
    
    model = PricePredictionModel(n_folds=config.N_FOLDS, random_state=config.RANDOM_SEED)
    
    # Train with ensemble
    oof_predictions = model.train(X_train, y_train_transformed, use_ensemble=True)
    
    # Transform predictions back if log was used
    if use_log_transform:
        oof_predictions = np.expm1(oof_predictions)
        actual_prices = y_train
    else:
        actual_prices = y_train
    
    # Calculate final SMAPE
    from model import smape
    final_smape = smape(actual_prices, oof_predictions)
    print(f"\nFinal OOF SMAPE: {final_smape:.4f}%")
    
    # Step 6: Save model
    print("\n[Step 6] Saving model...")
    model_path = os.path.join(config.MODELS_FOLDER, 'price_model.pkl')
    model.save(model_path)
    
    # Save configuration
    config_dict = {
        'use_images': use_images,
        'use_log_transform': use_log_transform,
        'n_features': X_train.shape[1],
        'oof_smape': final_smape
    }
    config_path = os.path.join(config.MODELS_FOLDER, 'model_config.pkl')
    joblib.dump(config_dict, config_path)
    print(f"Configuration saved to {config_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel artifacts saved in: {config.MODELS_FOLDER}")
    print(f"- Model: price_model.pkl")
    print(f"- Text Extractor: text_extractor.pkl")
    print(f"- Configuration: model_config.pkl")
    print(f"\nOut-of-fold SMAPE: {final_smape:.4f}%")
    print("\nNext steps:")
    print("1. Run predict.py to generate predictions on test set")
    print("2. Submit test_out.csv to the competition portal")
    print("="*70)


if __name__ == "__main__":
    main()
