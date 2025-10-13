"""
Model Saver Utility for Amazon ML Price Predictor
Run this in your Jupyter notebook to save trained models for Streamlit app
"""

import pickle
import os
import numpy as np

def save_trained_models_for_streamlit():
    """
    Save your trained models, vectorizer, svd, and scaler for use in Streamlit
    Run this function in your notebook after training is complete
    """
    
    # Create models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created {models_dir} directory")
    
    try:
        # Save preprocessors (these should be available in your notebook)
        print("Saving preprocessors...")
        
        # Save vectorizer
        if 'vectorizer' in globals():
            with open(f"{models_dir}/vectorizer.pkl", 'wb') as f:
                pickle.dump(vectorizer, f)
            print("✅ Saved vectorizer.pkl")
        else:
            print("❌ vectorizer not found in globals()")
        
        # Save SVD
        if 'svd' in globals():
            with open(f"{models_dir}/svd.pkl", 'wb') as f:
                pickle.dump(svd, f)
            print("✅ Saved svd.pkl")
        else:
            print("❌ svd not found in globals()")
        
        # Save scaler
        if 'scaler' in globals():
            with open(f"{models_dir}/scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            print("✅ Saved scaler.pkl")
        else:
            print("❌ scaler not found in globals()")
        
        # Save trained models (from final_models dict)
        print("\nSaving trained models...")
        
        if 'final_models' in globals():
            for model_name, model in final_models.items():
                filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
                try:
                    with open(f"{models_dir}/{filename}", 'wb') as f:
                        pickle.dump(model, f)
                    print(f"✅ Saved {filename}")
                except Exception as e:
                    print(f"❌ Error saving {filename}: {e}")
        else:
            print("❌ final_models not found in globals()")
        
        # Save ensemble weights
        if 'ensemble_weights' in globals():
            with open(f"{models_dir}/ensemble_weights.pkl", 'wb') as f:
                pickle.dump(ensemble_weights, f)
            print("✅ Saved ensemble_weights.pkl")
        else:
            print("❌ ensemble_weights not found in globals()")
        
        # Save model performance stats
        model_stats = {}
        if 'results_df' in globals():
            for _, row in results_df.iterrows():
                model_stats[row['Algorithm']] = {
                    'accuracy': row['Test SMAPE'],
                    'training_time': row.get('Training Time', 0),
                    'status': 'active'
                }
        
        with open(f"{models_dir}/model_stats.pkl", 'wb') as f:
            pickle.dump(model_stats, f)
        print("✅ Saved model_stats.pkl")
        
        print(f"\n🎉 All models saved successfully to '{models_dir}' folder!")
        print("Now restart your Streamlit app to use the trained models.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def test_model_loading():
    """Test if saved models can be loaded correctly"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("❌ Models directory not found. Run save_trained_models_for_streamlit() first.")
        return False
    
    print("Testing model loading...")
    
    try:
        # Test loading preprocessors
        with open(f"{models_dir}/vectorizer.pkl", 'rb') as f:
            vectorizer_test = pickle.load(f)
        print("✅ Vectorizer loaded successfully")
        
        with open(f"{models_dir}/svd.pkl", 'rb') as f:
            svd_test = pickle.load(f)
        print("✅ SVD loaded successfully")
        
        with open(f"{models_dir}/scaler.pkl", 'rb') as f:
            scaler_test = pickle.load(f)
        print("✅ Scaler loaded successfully")
        
        # Test loading models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            with open(f"{models_dir}/{model_file}", 'rb') as f:
                model = pickle.load(f)
            print(f"✅ {model_file} loaded successfully")
        
        print("\n🎉 All models can be loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def create_sample_prediction():
    """Create a sample prediction to verify everything works"""
    print("Creating sample prediction...")
    
    try:
        # Load models
        models_dir = "models"
        
        with open(f"{models_dir}/vectorizer.pkl", 'rb') as f:
            vectorizer_loaded = pickle.load(f)
        with open(f"{models_dir}/svd.pkl", 'rb') as f:
            svd_loaded = pickle.load(f)
        with open(f"{models_dir}/scaler.pkl", 'rb') as f:
            scaler_loaded = pickle.load(f)
        
        # Sample product
        sample_text = ["Gaming laptop with RTX 4080, 32GB RAM, 1TB SSD, high performance"]
        
        # Create features
        tfidf_matrix = vectorizer_loaded.transform(sample_text)
        svd_features = svd_loaded.transform(tfidf_matrix)
        features = scaler_loaded.transform(svd_features)
        
        print(f"✅ Features created: shape {features.shape}")
        
        # Test predictions with available models
        predictions = {}
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
            
            with open(f"{models_dir}/{model_file}", 'rb') as f:
                model = pickle.load(f)
            
            try:
                if 'lightgbm' in model_file or 'xgboost' in model_file or 'catboost' in model_file:
                    pred_log = model.predict(features)
                    pred = np.expm1(pred_log[0] if hasattr(pred_log, '__iter__') else pred_log)
                else:
                    pred = model.predict(features)
                    pred = pred[0] if hasattr(pred, '__iter__') else pred
                
                predictions[model_name] = round(pred, 2)
                print(f"✅ {model_name}: ${pred:.2f}")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
        
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()))
            print(f"\n🎯 Ensemble Prediction: ${ensemble_pred:.2f}")
            print("\n🎉 Sample prediction successful! Models are working correctly.")
            return True
        else:
            print("❌ No successful predictions")
            return False
            
    except Exception as e:
        print(f"❌ Error in sample prediction: {e}")
        return False

# Instructions for use
print("""
🚀 Amazon ML Model Saver Utility

To use this in your Jupyter notebook:

1. Run this cell to load the functions
2. After training your models, run: save_trained_models_for_streamlit()
3. Test loading with: test_model_loading()
4. Verify with: create_sample_prediction()
5. Restart your Streamlit app

Your notebook should have these variables:
- vectorizer (TfidfVectorizer)
- svd (TruncatedSVD) 
- scaler (RobustScaler)
- final_models (dict of trained models)
- ensemble_weights (dict of model weights)
- results_df (DataFrame with model performance)
""")