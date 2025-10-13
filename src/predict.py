# predict_single.py
"""
Simple OCT Image Classifier - Single Image Prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sys

# Import preprocessing
sys.path.append('src')
from preprocess import preprocess_image_gray
from features import extract_features

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("OCT DISEASE CLASSIFIER - SINGLE IMAGE")
print("=" * 80)

# ============================================================================
# LOAD MODELS
# ============================================================================
print("\nLoading models...")
model = load_model('models/oct_hybrid_optimized_final.h5')
effnet_model = load_model('models/effnet_feature_extractor.h5')
scaler_glcm = joblib.load('models/scaler_glcm.pkl')
scaler_effnet = joblib.load('models/scaler_effnet.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
CLASSES = label_encoder.classes_
print("✓ All models loaded!")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_effnet_features(img_path):
    """Extract EfficientNet features"""
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    img = load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rgb = np.concatenate([img_array] * 3, axis=-1)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    features = effnet_model.predict(img_rgb, verbose=0)
    return features.flatten()

# ============================================================================
# PREDICTION
# ============================================================================
def predict_image(image_path):
    """Predict disease class for OCT image"""
    
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Extract features
    gray_img = preprocess_image_gray(image_path)
    if gray_img.dtype != np.uint8:
        gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
    
    glcm_feats = extract_features(gray_img)
    effnet_feats = extract_effnet_features(image_path)
    
    # Scale features
    glcm_scaled = scaler_glcm.transform([glcm_feats])
    effnet_scaled = scaler_effnet.transform([effnet_feats])
    
    # Predict
    predictions = model.predict([glcm_scaled, effnet_scaled], verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n PREDICTED CLASS: {predicted_class}")
    print(f" CONFIDENCE: {confidence:.2f}%")
    print("\nAll Class Probabilities:")
    print("-" * 40)
    
    for i, cls in enumerate(CLASSES):
        prob = predictions[0][i] * 100
        bar = "█" * int(prob / 5)
        print(f"  {cls:8s}: {prob:5.2f}% {bar}")
    
    print("=" * 80)

# ============================================================================
# MAIN
# ============================================================================
print("\n" + "=" * 80)
image_path = input("Enter path to OCT image: ").strip('"')

if not os.path.exists(image_path):
    print(f"\n✗ Error: File not found: {image_path}")
else:
    predict_image(image_path)
