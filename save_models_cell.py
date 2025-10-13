# Save Models for Streamlit App
# Run this cell in your notebook after training is complete

import pickle
import os

# Create models directory
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created {models_dir} directory")

try:
    # Save preprocessors
    print("Saving preprocessors...")
    
    with open(f"{models_dir}/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Saved vectorizer.pkl")
    
    with open(f"{models_dir}/svd.pkl", 'wb') as f:
        pickle.dump(svd, f)
    print("✅ Saved svd.pkl")
    
    with open(f"{models_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved scaler.pkl")
    
    # Save trained models
    print("\nSaving trained models...")
    
    for model_name, model in final_models.items():
        filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        with open(f"{models_dir}/{filename}", 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved {filename}")
    
    # Save ensemble weights
    with open(f"{models_dir}/ensemble_weights.pkl", 'wb') as f:
        pickle.dump(ensemble_weights, f)
    print("✅ Saved ensemble_weights.pkl")
    
    print(f"\n🎉 All models saved to '{models_dir}' folder!")
    print("Restart your Streamlit app to use the trained models.")
    
except Exception as e:
    print(f"❌ Error saving models: {e}")
    print("Make sure you have run all training cells first!")