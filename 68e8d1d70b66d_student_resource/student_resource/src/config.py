"""
Configuration file for the product pricing model
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
IMAGES_FOLDER = os.path.join(BASE_DIR, 'images')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
FEATURES_FOLDER = os.path.join(BASE_DIR, 'features')

# Data files
TRAIN_FILE = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_FILE = os.path.join(DATASET_FOLDER, 'test.csv')
SAMPLE_TEST_FILE = os.path.join(DATASET_FOLDER, 'sample_test.csv')
OUTPUT_FILE = os.path.join(DATASET_FOLDER, 'test_out.csv')

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# Text processing
MAX_TEXT_LENGTH = 512
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Lightweight text embeddings

# Image processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_MODEL_NAME = 'microsoft/resnet-50'  # ResNet-50 for image features

# Model selection
USE_TEXT_FEATURES = True
USE_IMAGE_FEATURES = True
USE_ENSEMBLE = True

# Training parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_SEED,
    'n_estimators': 1000,
    'early_stopping_rounds': 50
}

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbosity': 0
}

# Create directories if they don't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)
