"""
Feature engineering module for product pricing
Extracts features from catalog content and images
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')


class TextFeatureExtractor:
    """Extract features from catalog_content text"""
    
    def __init__(self, max_features=5000, n_components=100):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf = None
        self.svd = None
        
    def extract_numeric_features(self, text):
        """Extract numeric features from text"""
        features = {}
        
        # Extract numbers (potential prices, weights, quantities)
        numbers = re.findall(r'\d+\.?\d*', str(text))
        features['num_count'] = len(numbers)
        features['max_number'] = max([float(n) for n in numbers], default=0)
        features['min_number'] = min([float(n) for n in numbers], default=0)
        features['avg_number'] = np.mean([float(n) for n in numbers]) if numbers else 0
        
        # Extract pack quantity if mentioned
        pack_match = re.search(r'Item Pack Quantity[:\s]+(\d+)', str(text))
        features['pack_quantity'] = int(pack_match.group(1)) if pack_match else 1
        
        # Text length features
        features['text_length'] = len(str(text))
        features['word_count'] = len(str(text).split())
        
        # Brand/quality indicators (case insensitive)
        text_lower = str(text).lower()
        features['has_organic'] = int('organic' in text_lower)
        features['has_premium'] = int('premium' in text_lower or 'deluxe' in text_lower)
        features['has_natural'] = int('natural' in text_lower)
        features['has_fresh'] = int('fresh' in text_lower)
        
        # Unit indicators
        features['has_ounce'] = int('ounce' in text_lower or 'oz' in text_lower)
        features['has_pound'] = int('pound' in text_lower or 'lb' in text_lower)
        features['has_gram'] = int('gram' in text_lower or ' g ' in text_lower)
        features['has_liter'] = int('liter' in text_lower or ' l ' in text_lower)
        
        return features
    
    def fit(self, texts):
        """Fit TF-IDF and SVD on training texts"""
        print("Fitting TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            strip_accents='unicode',
            lowercase=True
        )
        
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        print(f"Fitting SVD for dimensionality reduction to {self.n_components} components...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        
        return self
    
    def transform(self, texts):
        """Transform texts to feature vectors"""
        # Extract numeric features
        numeric_features = pd.DataFrame([
            self.extract_numeric_features(text) for text in texts
        ])
        
        # TF-IDF features
        tfidf_matrix = self.tfidf.transform(texts)
        tfidf_features = self.svd.transform(tfidf_matrix)
        tfidf_df = pd.DataFrame(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(self.n_components)]
        )
        
        # Combine all features
        all_features = pd.concat([
            numeric_features.reset_index(drop=True),
            tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        return all_features
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)


class ImageFeatureExtractor:
    """Extract features from product images using pre-trained models"""
    
    def __init__(self, image_folder, image_size=(224, 224)):
        self.image_folder = image_folder
        self.image_size = image_size
        self.model = None
        
    def load_model(self):
        """Load pre-trained model for feature extraction"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.models import Model
            
            # MobileNetV2 is lightweight and efficient (3.5M parameters)
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3),
                pooling='avg'
            )
            
            self.model = base_model
            self.preprocess = preprocess_input
            print("Loaded MobileNetV2 for image feature extraction")
            return True
            
        except ImportError:
            print("TensorFlow not available. Image features will be skipped.")
            return False
    
    def extract_features(self, image_links):
        """Extract features from images"""
        from tensorflow.keras.preprocessing import image as keras_image
        import os
        from pathlib import Path
        
        if self.model is None:
            if not self.load_model():
                return None
        
        features_list = []
        
        for img_link in image_links:
            try:
                # Get image filename from link
                filename = Path(img_link).name
                img_path = os.path.join(self.image_folder, filename)
                
                if os.path.exists(img_path):
                    # Load and preprocess image
                    img = keras_image.load_img(img_path, target_size=self.image_size)
                    img_array = keras_image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess(img_array)
                    
                    # Extract features
                    features = self.model.predict(img_array, verbose=0)
                    features_list.append(features.flatten())
                else:
                    # Image not found, use zero vector
                    features_list.append(np.zeros(1280))  # MobileNetV2 output size
                    
            except Exception as e:
                print(f"Error processing image {img_link}: {e}")
                features_list.append(np.zeros(1280))
        
        features_array = np.array(features_list)
        feature_df = pd.DataFrame(
            features_array,
            columns=[f'img_feat_{i}' for i in range(features_array.shape[1])]
        )
        
        return feature_df


def create_features(df, text_extractor=None, image_extractor=None, 
                   is_training=True, use_images=False):
    """
    Create features from dataframe
    
    Parameters:
    - df: Input dataframe with catalog_content and image_link
    - text_extractor: Fitted TextFeatureExtractor (or None for training)
    - image_extractor: ImageFeatureExtractor instance
    - is_training: Whether this is training data
    - use_images: Whether to extract image features
    
    Returns:
    - features: DataFrame with all features
    - text_extractor: Fitted text extractor (if training)
    """
    
    print("Extracting text features...")
    if is_training:
        text_extractor = TextFeatureExtractor()
        text_features = text_extractor.fit_transform(df['catalog_content'])
    else:
        text_features = text_extractor.transform(df['catalog_content'])
    
    all_features = text_features
    
    if use_images and image_extractor is not None:
        print("Extracting image features...")
        image_features = image_extractor.extract_features(df['image_link'])
        if image_features is not None:
            all_features = pd.concat([
                all_features.reset_index(drop=True),
                image_features.reset_index(drop=True)
            ], axis=1)
    
    return all_features, text_extractor
