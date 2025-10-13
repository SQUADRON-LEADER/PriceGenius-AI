import pickle
import os

print("Checking model types:")
models_dir = "models"
model_files = ['lightgbm_model.pkl', 'xgboost_model.pkl', 'catboost_model.pkl']

for model_file in model_files:
    if os.path.exists(f"{models_dir}/{model_file}"):
        with open(f"{models_dir}/{model_file}", 'rb') as f:
            model = pickle.load(f)
            print(f"{model_file}: {type(model)}")
    else:
        print(f"{model_file}: NOT FOUND")