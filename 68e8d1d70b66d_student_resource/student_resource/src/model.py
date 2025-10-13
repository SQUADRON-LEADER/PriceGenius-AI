"""
Model training module for product pricing
Implements ensemble of multiple models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


class PricePredictionModel:
    """Ensemble model for price prediction"""
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []
        self.feature_importance = None
        
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBRegressor(**params, n_estimators=1000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MAE',
            random_state=self.random_state,
            verbose=False,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        return model
    
    def train(self, X, y, use_ensemble=True):
        """
        Train ensemble model with cross-validation
        
        Parameters:
        - X: Feature matrix
        - y: Target values (prices)
        - use_ensemble: Whether to use ensemble of multiple models
        
        Returns:
        - oof_predictions: Out-of-fold predictions for validation
        """
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_predictions = np.zeros(len(X))
        
        print(f"\nTraining with {self.n_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\n{'='*50}")
            print(f"Fold {fold}/{self.n_folds}")
            print(f"{'='*50}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_models = {}
            fold_predictions = []
            
            # Train LightGBM
            print("Training LightGBM...")
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
            lgb_pred = lgb_model.predict(X_val)
            lgb_smape = smape(y_val, lgb_pred)
            print(f"LightGBM SMAPE: {lgb_smape:.4f}%")
            fold_models['lgb'] = lgb_model
            fold_predictions.append(lgb_pred)
            
            if use_ensemble:
                # Train XGBoost
                print("Training XGBoost...")
                xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
                xgb_pred = xgb_model.predict(X_val)
                xgb_smape = smape(y_val, xgb_pred)
                print(f"XGBoost SMAPE: {xgb_smape:.4f}%")
                fold_models['xgb'] = xgb_model
                fold_predictions.append(xgb_pred)
                
                # Train CatBoost
                print("Training CatBoost...")
                cat_model = self.train_catboost(X_train, y_train, X_val, y_val)
                cat_pred = cat_model.predict(X_val)
                cat_smape = smape(y_val, cat_pred)
                print(f"CatBoost SMAPE: {cat_smape:.4f}%")
                fold_models['cat'] = cat_model
                fold_predictions.append(cat_pred)
            
            # Ensemble prediction (average)
            fold_pred = np.mean(fold_predictions, axis=0)
            ensemble_smape = smape(y_val, fold_pred)
            print(f"Ensemble SMAPE: {ensemble_smape:.4f}%")
            
            oof_predictions[val_idx] = fold_pred
            self.models.append(fold_models)
        
        # Calculate overall OOF score
        overall_smape = smape(y, oof_predictions)
        print(f"\n{'='*50}")
        print(f"Overall OOF SMAPE: {overall_smape:.4f}%")
        print(f"{'='*50}\n")
        
        return oof_predictions
    
    def predict(self, X):
        """
        Make predictions using trained ensemble
        
        Parameters:
        - X: Feature matrix
        
        Returns:
        - predictions: Predicted prices
        """
        
        all_predictions = []
        
        for fold_idx, fold_models in enumerate(self.models):
            fold_preds = []
            
            # LightGBM prediction
            if 'lgb' in fold_models:
                lgb_pred = fold_models['lgb'].predict(X)
                fold_preds.append(lgb_pred)
            
            # XGBoost prediction
            if 'xgb' in fold_models:
                xgb_pred = fold_models['xgb'].predict(X)
                fold_preds.append(xgb_pred)
            
            # CatBoost prediction
            if 'cat' in fold_models:
                cat_pred = fold_models['cat'].predict(X)
                fold_preds.append(cat_pred)
            
            # Average predictions from all models in this fold
            fold_pred = np.mean(fold_preds, axis=0)
            all_predictions.append(fold_pred)
        
        # Average predictions across all folds
        final_predictions = np.mean(all_predictions, axis=0)
        
        # Ensure positive predictions
        final_predictions = np.maximum(final_predictions, 0.01)
        
        return final_predictions
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
