# Create simple models for testing Streamlit
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

print("🔧 Creating simple test models for Streamlit...")

# Create models directory
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created {models_dir} directory")

# Create sample data for fitting transformers
sample_texts = [
    "High-performance gaming laptop with RTX 4080, 32GB RAM, 1TB SSD",
    "Apple iPhone 14 Pro with 256GB storage and A16 chip",
    "Samsung 65 inch 4K OLED Smart TV with HDR",
    "Dell XPS 13 ultrabook with Intel i7 processor",
    "Sony WH-1000XM5 wireless noise cancelling headphones"
] * 20  # Duplicate for fitting

sample_prices = np.random.uniform(100, 3000, len(sample_texts))

# Create and fit preprocessing pipeline
print("Creating preprocessing pipeline...")

# 1. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(sample_texts)

# 2. SVD for dimensionality reduction
n_features = tfidf_matrix.shape[1]
n_components = min(50, n_features - 1)  # Use smaller number for test data
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd_features = svd.fit_transform(tfidf_matrix)
print(f"Using {n_components} SVD components from {n_features} TF-IDF features")

# 3. Create mock numeric features (12 features)
numeric_features = np.random.randn(len(sample_texts), 12)

# 4. Combine features and scale
combined_features = np.hstack([svd_features, numeric_features])
scaler = RobustScaler()
scaled_features = scaler.fit_transform(combined_features)

# Create simple models
print("Creating simple models...")

# Create simple models for testing
models = {
    'LightGBM': RandomForestRegressor(n_estimators=10, random_state=42),
    'XGBoost': LinearRegression(), 
    'CatBoost': RandomForestRegressor(n_estimators=5, random_state=42)
}

# Fit models
for name, model in models.items():
    model.fit(scaled_features, sample_prices)
    print(f"Fitted {name} model")

# Save everything
print("\nSaving models and pipeline...")

# Save preprocessors
with open(f"{models_dir}/vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Saved vectorizer.pkl")

with open(f"{models_dir}/svd.pkl", 'wb') as f:
    pickle.dump(svd, f)
print("✅ Saved svd.pkl")

with open(f"{models_dir}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved scaler.pkl")

# Save models
for name, model in models.items():
    filename = f"{name.lower()}_model.pkl"
    with open(f"{models_dir}/{filename}", 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved {filename}")

# Save ensemble weights
ensemble_weights = {
    'LightGBM': 0.4,
    'XGBoost': 0.35,
    'CatBoost': 0.25
}

with open(f"{models_dir}/ensemble_weights.pkl", 'wb') as f:
    pickle.dump(ensemble_weights, f)
print("✅ Saved ensemble_weights.pkl")

# Create feature pipeline function
feature_pipeline_code = '''
import re
import numpy as np
import pandas as pd

def extract_numeric_features(text):
    """Extract numeric features from text - simplified version"""
    features = {}
    text_str = str(text)
    
    # Extract all numbers
    numbers = re.findall(r'\\d+', text_str)
    features['num_count'] = len(numbers)
    features['max_number'] = max([int(n) for n in numbers], default=0)
    features['avg_number'] = np.mean([int(n) for n in numbers]) if numbers else 0
    
    # Text statistics
    features['text_length'] = len(text_str)
    features['word_count'] = len(text_str.split())
    features['unique_words'] = len(set(text_str.lower().split()))
    features['avg_word_length'] = features['text_length'] / max(features['word_count'], 1)
    
    # Keywords
    high_value_keywords = ['premium', 'pro', 'ultra', 'max', 'gaming']
    features['high_value_score'] = sum(1 for kw in high_value_keywords if kw in text_str.lower())
    
    brands = ['apple', 'samsung', 'sony', 'dell', 'hp']
    features['brand_score'] = sum(1 for brand in brands if brand in text_str.lower())
    
    tech_keywords = ['gb', 'tb', 'ram', 'ssd', 'processor']
    features['tech_score'] = sum(1 for kw in tech_keywords if kw in text_str.lower())
    
    # Storage and RAM
    features['max_storage_gb'] = 0
    features['ram_gb'] = 0
    
    return features

def create_features_for_prediction(text_list, vectorizer, svd, scaler):
    """Create features for prediction using the saved pipeline"""
    import pandas as pd
    import numpy as np
    
    # Create DataFrame
    df = pd.DataFrame({'catalog_content': text_list})
    df['catalog_content'] = df['catalog_content'].fillna('')
    
    # 1. TF-IDF Features
    tfidf_features = vectorizer.transform(df['catalog_content'])
    
    # 2. SVD Features (200 components)
    text_features = svd.transform(tfidf_features)
    
    # 3. Numeric Features (12 components)
    numeric_features_list = []
    for content in df['catalog_content']:
        features = extract_numeric_features(content)
        ordered_features = [
            features['num_count'],
            features['max_number'], 
            features['avg_number'],
            features['text_length'],
            features['word_count'],
            features['unique_words'],
            features['avg_word_length'],
            features['high_value_score'],
            features['brand_score'],
            features['tech_score'],
            features['max_storage_gb'],
            features['ram_gb']
        ]
        numeric_features_list.append(ordered_features)
    
    numeric_df = pd.DataFrame(numeric_features_list)
    
    # 4. Combine features (SVD 200 + Numeric 12 = 212)
    combined_features = np.hstack([text_features, numeric_df.values])
    
    # 5. Scale combined features
    scaled_features = scaler.transform(combined_features)
    
    return scaled_features
'''

with open(f"{models_dir}/feature_pipeline.py", 'w') as f:
    f.write(feature_pipeline_code)
print("✅ Saved feature_pipeline.py")

print(f"\n🎉 SUCCESS! All models saved to '{models_dir}' folder")
print(f"📁 Files created:")
for file in os.listdir(models_dir):
    print(f"   📄 {file}")

print(f"\n🚀 Ready to run Streamlit!")