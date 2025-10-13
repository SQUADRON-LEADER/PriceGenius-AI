# Smart Product Pricing Challenge - Solution Documentation

## Team Information
- **Team Name:** [Your Team Name]
- **Date:** October 12, 2025

---

## 1. Methodology Overview

Our solution employs a **multi-modal ensemble approach** that combines text and image features to predict product prices. The methodology focuses on extracting meaningful features from both catalog content and product images, then using gradient boosting models for accurate price prediction.

### Key Approach:
- **Feature Engineering:** Comprehensive text feature extraction from catalog content
- **Model Ensemble:** Combination of LightGBM and XGBoost models
- **Target Transformation:** Log transformation of prices for better model performance
- **Cross-Validation:** 5-fold cross-validation for robust model evaluation

---

## 2. Feature Engineering Techniques

### 2.1 Text Features (Primary)

#### **A. Numeric Features from Catalog Content:**
- Number count, max/min/average numbers in text
- Pack quantity extraction (IPQ)
- Text length and word count
- Price-related numeric patterns

#### **B. TF-IDF Features:**
- **Vectorizer:** TfidfVectorizer with bigrams (1,2)
- **Max Features:** 5000 most important terms
- **Dimensionality Reduction:** TruncatedSVD to 100 components
- Captures semantic meaning of product descriptions

#### **C. Domain-Specific Features:**
- Brand/quality indicators: organic, premium, natural, fresh
- Unit measurements: ounce, pound, gram, liter
- Product category signals extracted via keyword matching

### 2.2 Image Features (Optional Enhancement)

- **Model:** MobileNetV2 pre-trained on ImageNet (3.5M parameters - well within 8B limit)
- **Architecture:** Feature extraction from pooling layer (1280 dimensions)
- **Preprocessing:** Resize to 224x224, ImageNet normalization
- Images can enhance predictions for visually-driven product categories

---

## 3. Model Architecture

### 3.1 Individual Models

#### **LightGBM Regressor:**
- **Objective:** Regression with MAE loss
- **Parameters:**
  - num_leaves: 31
  - learning_rate: 0.05
  - feature_fraction: 0.8
  - bagging_fraction: 0.8
  - max_iterations: 1000 with early stopping

#### **XGBoost Regressor:**
- **Objective:** reg:squarederror
- **Parameters:**
  - max_depth: 6
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
  - tree_method: hist (for efficiency)

### 3.2 Ensemble Strategy

- **Method:** Simple averaging of LightGBM and XGBoost predictions
- **Rationale:** Combines strengths of both algorithms
  - LightGBM: Fast, efficient with categorical features
  - XGBoost: Robust, handles outliers well
- **Cross-Validation:** 5-fold CV ensures generalization

### 3.3 Target Engineering

- **Transformation:** Log(price + 1) transformation
- **Benefits:**
  - Handles wide price range (from <$1 to >$100)
  - Reduces impact of extreme values
  - More stable gradients during training
- **Inverse Transform:** Exp(prediction) - 1 for final output

---

## 4. Data Preprocessing

### 4.1 Handling Missing Values
- Filled missing catalog_content with empty strings
- Image links validated during feature extraction

### 4.2 Feature Scaling
- **Scaler:** RobustScaler (resistant to outliers)
- Applied after feature extraction
- Fitted on training data, transformed test data

### 4.3 Outlier Handling
- Log transformation naturally handles extreme prices
- RobustScaler uses median/IQR instead of mean/std

---

## 5. Training Process

### 5.1 Cross-Validation Strategy
- **K-Folds:** 5 folds with shuffle
- **Purpose:** Prevent overfitting, assess generalization
- Out-of-fold predictions used for model evaluation

### 5.2 Model Training Pipeline
1. Load and preprocess data
2. Extract text features (TF-IDF + numeric)
3. Optional: Extract image features
4. Scale features with RobustScaler
5. Train LightGBM and XGBoost on each fold
6. Generate out-of-fold predictions
7. Ensemble predictions by averaging

### 5.3 Early Stopping
- Both models use early stopping (50 rounds)
- Prevents overfitting and reduces training time
- Validation set performance monitored

---

## 6. Evaluation Metrics

### SMAPE (Symmetric Mean Absolute Percentage Error)
- **Formula:** SMAPE = (1/n) × Σ |pred - actual| / ((|actual| + |pred|)/2) × 100
- **Why SMAPE:** 
  - Treats over/under predictions equally
  - Scale-independent (suitable for varying price ranges)
  - Bounded between 0-200%

### Expected Performance
- **Text-only features:** Expected SMAPE ~15-25%
- **With image features:** Potential improvement of 2-5%
- **Cross-validation:** Consistent across folds indicates good generalization

---

## 7. Implementation Details

### 7.1 Code Structure
```
src/
├── config.py              # Configuration and hyperparameters
├── utils.py               # Image downloading utilities
├── feature_engineering.py # Feature extraction classes
├── model.py               # Model training classes
└── train_complete.py      # Complete training pipeline
```

### 7.2 Key Scripts
- **train_complete.py:** Main training and prediction pipeline
- **feature_engineering.py:** TextFeatureExtractor and ImageFeatureExtractor
- **config.py:** Centralized configuration management

### 7.3 Dependencies
- Core: numpy, pandas, scikit-learn
- Models: lightgbm, xgboost
- Text: sentence-transformers (optional enhancement)
- Images: tensorflow (optional)

---

## 8. Predictions and Submission

### 8.1 Prediction Process
1. Load trained models and feature extractors
2. Extract features from test data (same pipeline as training)
3. Scale features using fitted scaler
4. Generate predictions from each fold's models
5. Average predictions across folds and models
6. Apply inverse log transformation
7. Ensure all predictions are positive (min: 0.01)

### 8.2 Output Format
- **Columns:** sample_id, price
- **Format:** CSV matching sample_test_out.csv
- **Validation:** All 75K test samples included

---

## 9. Key Innovations

1. **Comprehensive Text Feature Engineering:** 
   - Combines TF-IDF, numeric extraction, and domain knowledge

2. **Robust Target Transformation:**
   - Log transformation handles wide price ranges effectively

3. **Efficient Ensemble:**
   - LightGBM + XGBoost combination balances speed and accuracy

4. **Scalable Architecture:**
   - Modular design allows easy addition of new features/models

5. **Cross-Validation:**
   - Ensures reliable performance estimates

---

## 10. Potential Improvements

### Short-term:
- Hyperparameter tuning with Optuna/GridSearch
- Add CatBoost to ensemble
- Extract brand names more systematically
- Category-specific models

### Medium-term:
- Use transformer-based text embeddings (e.g., DistilBERT)
- Advanced image models (EfficientNet, Vision Transformers)
- Weighted ensemble based on validation performance

### Long-term:
- Multi-task learning (price + category)
- Attention mechanisms for text-image fusion
- External data augmentation (within rules)

---

## 11. License Compliance

✅ **All models comply with MIT/Apache 2.0 licenses:**
- LightGBM: MIT License
- XGBoost: Apache 2.0 License
- MobileNetV2: Apache 2.0 License
- Scikit-learn: BSD License

✅ **Total model parameters:** ~3.5M (MobileNetV2) - well within 8B limit

---

## 12. Conclusion

Our solution provides a solid baseline for product pricing prediction using only the provided training data. The ensemble approach with comprehensive feature engineering delivers reliable predictions while maintaining computational efficiency. The modular design allows for easy experimentation and improvements.

**Key Strengths:**
- No external data usage (fully compliant)
- Robust cross-validation
- Efficient feature engineering
- Scalable and maintainable code

---

## 13. How to Run

### Training:
```bash
cd src
python train_complete.py
```

### Quick Start (No Images):
1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Run training: `python src/train_complete.py`
3. Output will be saved to: `dataset/test_out.csv`

### With Image Features (Optional):
1. Install TensorFlow: `pip install tensorflow`
2. Set `USE_IMAGES = True` in `train_complete.py`
3. Run training (will take longer)

---

**Note:** This documentation can be customized based on actual results and any modifications made to the approach.
