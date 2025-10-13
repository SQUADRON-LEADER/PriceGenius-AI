
import re
import numpy as np
import pandas as pd

def extract_numeric_features(text):
    """Extract numeric features from text - simplified version"""
    features = {}
    text_str = str(text)
    
    # Extract all numbers
    numbers = re.findall(r'\d+', text_str)
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
