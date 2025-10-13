# visualize_prediction.py
"""
OCT Image Visualization & Analysis Tool
Shows: Original, Enhanced, Features, Heatmaps, Predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import os
import sys
from PIL import Image
import cv2

# Import preprocessing
sys.path.append('src')
from preprocess import preprocess_image_gray
from features import extract_features

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sns.set_style('whitegrid')

print("=" * 80)
print("OCT IMAGE VISUALIZATION & ANALYSIS")
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
    img = load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rgb = np.concatenate([img_array] * 3, axis=-1)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    features = effnet_model.predict(img_rgb, verbose=0)
    return features.flatten(), img_rgb

# ============================================================================
# HEATMAP GENERATION (Grad-CAM)
# ============================================================================
def generate_gradcam_heatmap(img_path):
    """Generate Grad-CAM heatmap showing important regions"""
    
    # Load and preprocess
    img = load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rgb = np.concatenate([img_array] * 3, axis=-1)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    
    # Get the last convolutional layer
    last_conv_layer = None
    for layer in effnet_model.layers[::-1]:
        if 'conv' in layer.name.lower():
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        return None
    
    # Create gradient model
    grad_model = Model(
        inputs=effnet_model.input,
        outputs=[last_conv_layer.output, effnet_model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_rgb)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]
    
    # Get gradients
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-10)
    
    # Resize to image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    return heatmap

# ============================================================================
# COMPLETE VISUALIZATION
# ============================================================================
def visualize_complete_analysis(image_path):
    """
    Complete visualization with all components:
    1. Original & Enhanced Images
    2. Feature Extraction Visualization
    3. Grad-CAM Heatmap
    4. Prediction Results
    """
    
    print(f"\nAnalyzing: {os.path.basename(image_path)}")
    
    # Load original image
    original_img = Image.open(image_path).convert('L')
    original_array = np.array(original_img)
    
    # Get enhanced image
    enhanced_img = preprocess_image_gray(image_path)
    if enhanced_img.dtype != np.uint8:
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    
    # Extract features
    glcm_feats = extract_features(enhanced_img)
    effnet_feats, _ = extract_effnet_features(image_path)
    
    # Scale features
    glcm_scaled = scaler_glcm.transform([glcm_feats])
    effnet_scaled = scaler_effnet.transform([effnet_feats])
    
    # Predict
    predictions = model.predict([glcm_scaled, effnet_scaled], verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    # Generate heatmap
    print("Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam_heatmap(image_path)
    
    # ========================================================================
    # CREATE COMPREHENSIVE VISUALIZATION
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # ========================================
    # ROW 1: Images
    # ========================================
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_array, cmap='gray')
    ax1.set_title('1. Original OCT Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Enhanced Image (CLAHE)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced_img, cmap='gray')
    ax2.set_title('2. Enhanced Image (CLAHE)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Heatmap Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    if heatmap is not None:
        # Resize original to match heatmap
        original_resized = cv2.resize(original_array, (224, 224))
        ax3.imshow(original_resized, cmap='gray')
        ax3.imshow(heatmap, cmap='jet', alpha=0.5)
        ax3.set_title('3. Grad-CAM Heatmap\n(Important Regions)', 
                     fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Heatmap\nNot Available', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('3. Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Prediction Result
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    ax4.text(0.5, 0.7, '🔍 PREDICTION', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax4.text(0.5, 0.5, predicted_class, ha='center', va='center', 
            fontsize=24, fontweight='bold', 
            color='green' if confidence > 90 else 'orange')
    ax4.text(0.5, 0.3, f'{confidence:.2f}% Confidence', 
            ha='center', va='center', fontsize=14)
    
    # ========================================
    # ROW 2: Feature Visualizations
    # ========================================
    
    # GLCM Texture Features
    ax5 = fig.add_subplot(gs[1, 0:2])
    glcm_names = ['Contrast', 'Energy', 'Correlation', 
                  'Homogeneity', 'Entropy', 'Variance']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    bars = ax5.barh(glcm_names, glcm_feats, color=colors, edgecolor='black')
    ax5.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
    ax5.set_title('4. GLCM Texture Features', fontsize=14, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, glcm_feats):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    # EfficientNet Features (top 20)
    ax6 = fig.add_subplot(gs[1, 2:4])
    top_20_indices = np.argsort(np.abs(effnet_feats))[-20:]
    top_20_values = effnet_feats[top_20_indices]
    ax6.bar(range(20), top_20_values, color='steelblue', edgecolor='black')
    ax6.set_xlabel('Feature Index (Top 20)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
    ax6.set_title('5. EfficientNet Deep Features (Top 20)', 
                 fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # ========================================
    # ROW 3: Prediction Analysis
    # ========================================
    
    # Class Probabilities
    ax7 = fig.add_subplot(gs[2, 0:2])
    probs = predictions[0] * 100
    colors_pred = ['green' if CLASSES[i] == predicted_class else 'lightgray' 
                   for i in range(len(CLASSES))]
    bars = ax7.bar(CLASSES, probs, color=colors_pred, edgecolor='black', linewidth=2)
    ax7.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax7.set_title('6. All Class Probabilities', fontsize=14, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, height,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Confidence Meter
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')
    
    # Draw confidence gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Background arc
    ax8.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=15, alpha=0.2)
    
    # Confidence arc
    conf_theta = np.linspace(0, np.pi * confidence / 100, 100)
    color = 'green' if confidence > 90 else 'orange' if confidence > 70 else 'red'
    ax8.plot(np.cos(conf_theta), np.sin(conf_theta), 
            color=color, linewidth=15)
    
    # Labels
    ax8.text(0, -0.3, f'{confidence:.1f}%', ha='center', va='center',
            fontsize=36, fontweight='bold', color=color)
    ax8.text(0, -0.5, 'Confidence Score', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax8.set_xlim(-1.2, 1.2)
    ax8.set_ylim(-0.7, 1.2)
    
    # ========================================
    # Overall Title
    # ========================================
    fig.suptitle(f'OCT Image Analysis: {os.path.basename(image_path)}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save visualization
    output_path = f'visualizations_{os.path.splitext(os.path.basename(image_path))[0]}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    # Show
    plt.show()
    
    # ========================================================================
    # PRINT DETAILED ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print(f"\n Predicted Class: {predicted_class}")
    print(f" Confidence: {confidence:.2f}%")
    print(f"\n GLCM Texture Features:")
    for name, val in zip(glcm_names, glcm_feats):
        print(f"   {name:12s}: {val:.4f}")
    print(f"\n Deep Features: {len(effnet_feats)} features extracted")
    print(f"   Top feature value: {np.max(np.abs(effnet_feats)):.4f}")
    print(f"   Feature mean: {np.mean(effnet_feats):.4f}")
    print(f"   Feature std: {np.std(effnet_feats):.4f}")
    print("\n" + "=" * 80)

# ============================================================================
# MAIN
# ============================================================================
print("\n" + "=" * 80)
image_path = input("Enter path to OCT image: ").strip('"')

if not os.path.exists(image_path):
    print(f"\n✗ Error: File not found: {image_path}")
else:
    visualize_complete_analysis(image_path)
