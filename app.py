# app.py
"""
OCT Disease Classifier - Complete Web Application with Dynamic Severity Assessment
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import cv2

# Import preprocessing
sys.path.append('src')
from preprocess import preprocess_image_gray
from features import extract_features

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# DISEASE INFORMATION DATABASE
# ============================================================================
DISEASE_INFO = {
    'CNV': {
        'name': 'Choroidal Neovascularization',
        'description': 'Abnormal blood vessel growth under the retina that can leak fluid and cause vision loss.',
        'symptoms': [
            'Sudden vision changes',
            'Distorted vision (straight lines appear wavy)',
            'Central blind spot',
            'Color vision changes'
        ],
        'causes': [
            'Age-related macular degeneration (wet AMD)',
            'High myopia',
            'Eye injury or inflammation',
            'Genetic factors'
        ],
        'treatment': [
            'Anti-VEGF injections (Lucentis, Eylea, Avastin)',
            'Photodynamic therapy (PDT)',
            'Laser treatment (in select cases)',
            'Monthly monitoring initially'
        ],
        'prognosis': 'With early treatment, 90% of patients maintain or improve vision',
        'lifestyle': [
            'Stop smoking immediately',
            'Eat leafy greens and omega-3 rich foods',
            'Use Amsler grid daily to monitor',
            'Protect eyes from UV light'
        ]
    },
    'DME': {
        'name': 'Diabetic Macular Edema',
        'description': 'Fluid buildup in the macula caused by leaking blood vessels due to diabetes.',
        'symptoms': [
            'Blurred or wavy central vision',
            'Difficulty reading',
            'Faded or washed-out colors',
            'Floaters or dark spots'
        ],
        'causes': [
            'Poorly controlled diabetes',
            'Long-standing diabetes (10+ years)',
            'High blood pressure',
            'High cholesterol'
        ],
        'treatment': [
            'Anti-VEGF injections',
            'Steroid injections or implants',
            'Focal laser treatment',
            'Blood sugar control is critical'
        ],
        'prognosis': 'Early treatment prevents vision loss in 95% of cases',
        'lifestyle': [
            'Maintain HbA1c below 7%',
            'Control blood pressure (<130/80)',
            'Regular exercise (30 min daily)',
            'Low glycemic diet'
        ]
    },
    'DRUSEN': {
        'name': 'Drusen Deposits',
        'description': 'Yellow deposits under the retina, early sign of age-related macular degeneration.',
        'symptoms': [
            'Often no symptoms in early stages',
            'Slight blurring of central vision',
            'Need for brighter light when reading',
            'Difficulty adapting to low light'
        ],
        'causes': [
            'Aging (common after age 60)',
            'Family history of AMD',
            'Smoking',
            'Cardiovascular disease'
        ],
        'treatment': [
            'AREDS2 vitamins (if intermediate AMD)',
            'Regular monitoring every 6-12 months',
            'Amsler grid self-monitoring',
            'Lifestyle modifications'
        ],
        'prognosis': 'Slow progression; 10-15% may develop wet AMD over 5 years',
        'lifestyle': [
            'Take AREDS2 supplements',
            'Eat dark leafy greens daily',
            'Stop smoking',
            'Maintain healthy weight'
        ]
    },
    'NORMAL': {
        'name': 'Normal Retina',
        'description': 'Healthy retinal structure with no signs of disease.',
        'symptoms': [
            'No symptoms',
            'Clear vision',
            'Normal color perception',
            'No distortions'
        ],
        'causes': [
            'Healthy eyes',
            'No underlying conditions'
        ],
        'treatment': [
            'Continue regular eye exams',
            'Maintain healthy lifestyle',
            'Protect eyes from UV damage',
            'No treatment needed'
        ],
        'prognosis': 'Excellent - maintain with preventive care',
        'lifestyle': [
            'Annual comprehensive eye exam',
            'Eat antioxidant-rich foods',
            'Wear UV-blocking sunglasses',
            'Avoid smoking'
        ]
    }
}

# ============================================================================
# LOAD MODELS
# ============================================================================
print("Loading models...")
model = load_model('models/oct_hybrid_optimized_final.h5')
effnet_model = load_model('models/effnet_feature_extractor.h5')
scaler_glcm = joblib.load('models/scaler_glcm.pkl')
scaler_effnet = joblib.load('models/scaler_effnet.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
CLASSES = label_encoder.classes_
print("✓ Models loaded successfully!")

# ============================================================================
# DYNAMIC SEVERITY ASSESSMENT
# ============================================================================
def calculate_dynamic_severity(predicted_class, confidence, glcm_features, effnet_features):
    """
    Calculate dynamic severity based on:
    - Predicted disease
    - Confidence level
    - Image features (contrast, entropy indicate severity)
    """
    
    # If NORMAL, assess based on confidence
    if predicted_class == 'NORMAL':
        if confidence > 95:
            return {
                'level': 'None',
                'description': 'Healthy retina with no disease detected',
                'urgency': 'Routine eye exam recommended annually',
                'risk_score': 0,
                'color': '#6bcf7f'
            }
        else:
            return {
                'level': 'Uncertain',
                'description': 'Appears normal but low confidence - recommend follow-up',
                'urgency': 'Schedule check-up within 3 months',
                'risk_score': 20,
                'color': '#ffd93d'
            }
    
    # For diseases, calculate severity based on features
    contrast = glcm_features[0]  # High contrast = more severe
    entropy = glcm_features[4]   # High entropy = more chaotic/severe
    
    # Calculate severity score (0-100)
    feature_score = (contrast / 300) * 50 + (entropy / 10) * 50
    confidence_factor = confidence / 100
    severity_score = feature_score * confidence_factor
    
    # Determine severity level and details based on disease type
    if predicted_class == 'CNV':
        if severity_score > 70:
            return {
                'level': 'Severe',
                'description': 'Advanced CNV with significant vessel growth - immediate treatment needed',
                'urgency': '🚨 URGENT: See retinal specialist within 1 week',
                'risk_score': 90,
                'color': '#ff3838'
            }
        elif severity_score > 40:
            return {
                'level': 'Moderate',
                'description': 'Active CNV detected - treatment recommended',
                'urgency': 'Schedule appointment within 1-2 weeks',
                'risk_score': 70,
                'color': '#ff6b6b'
            }
        else:
            return {
                'level': 'Early/Mild',
                'description': 'Early stage CNV - close monitoring required',
                'urgency': 'Consult specialist within 2-3 weeks',
                'risk_score': 50,
                'color': '#ffa07a'
            }
    
    elif predicted_class == 'DME':
        if severity_score > 65:
            return {
                'level': 'Severe',
                'description': 'Significant macular edema - vision at risk',
                'urgency': 'See ophthalmologist within 1-2 weeks',
                'risk_score': 80,
                'color': '#ff6b6b'
            }
        elif severity_score > 35:
            return {
                'level': 'Moderate',
                'description': 'Clinically significant macular edema present',
                'urgency': 'Schedule treatment within 2-4 weeks',
                'risk_score': 60,
                'color': '#ffa07a'
            }
        else:
            return {
                'level': 'Mild',
                'description': 'Early macular edema detected',
                'urgency': 'Follow-up in 4-6 weeks',
                'risk_score': 40,
                'color': '#ffd93d'
            }
    
    elif predicted_class == 'DRUSEN':
        if severity_score > 60:
            return {
                'level': 'Intermediate AMD',
                'description': 'Multiple drusen deposits - higher risk of progression',
                'urgency': 'Follow-up every 6 months with AREDS vitamins',
                'risk_score': 50,
                'color': '#ffa07a'
            }
        elif severity_score > 30:
            return {
                'level': 'Early AMD',
                'description': 'Moderate drusen burden detected',
                'urgency': 'Annual monitoring recommended',
                'risk_score': 30,
                'color': '#ffd93d'
            }
        else:
            return {
                'level': 'Minimal',
                'description': 'Few small drusen - normal aging changes',
                'urgency': 'Routine follow-up in 12 months',
                'risk_score': 15,
                'color': '#4facfe'
            }
    
    # Default fallback
    return {
        'level': 'Moderate',
        'description': 'Standard severity assessment',
        'urgency': 'Consult with eye care professional',
        'risk_score': 50,
        'color': '#667eea'
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_effnet_features(img_path):
    """Extract EfficientNet features"""
    img = load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rgb = np.concatenate([img_array] * 3, axis=-1)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    features = effnet_model.predict(img_rgb, verbose=0)
    return features.flatten()

def generate_heatmap(img_path):
    """Generate simple attention heatmap"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    
    # Simple edge-based heatmap
    edges = cv2.Canny(img_resized, 50, 150)
    heatmap = cv2.GaussianBlur(edges, (21, 21), 0)
    heatmap = heatmap.astype(float) / (heatmap.max() + 1e-10)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original
    img_colored = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(img_colored, 0.6, heatmap_colored, 0.4, 0)
    
    return blended

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html', classes=CLASSES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load original image
        original_img = Image.open(filepath).convert('L')
        original_array = np.array(original_img)
        
        # Get enhanced image
        enhanced_img = preprocess_image_gray(filepath)
        if enhanced_img.dtype != np.uint8:
            enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        # Extract features
        glcm_feats = extract_features(enhanced_img)
        effnet_feats = extract_effnet_features(filepath)
        
        # Scale features
        glcm_scaled = scaler_glcm.transform([glcm_feats])
        effnet_scaled = scaler_effnet.transform([effnet_feats])
        
        # Predict
        predictions = model.predict([glcm_scaled, effnet_scaled], verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        # Get all probabilities
        all_probs = {
            cls: float(predictions[0][i] * 100)
            for i, cls in enumerate(CLASSES)
        }
        
        # Calculate dynamic severity
        severity_info = calculate_dynamic_severity(
            predicted_class, 
            confidence, 
            glcm_feats,
            effnet_feats
        )
        
        # Get base disease info and add dynamic severity
        disease_info = DISEASE_INFO[predicted_class].copy()
        disease_info['severity'] = severity_info['level']
        disease_info['urgency'] = severity_info['urgency']
        disease_info['risk_score'] = severity_info['risk_score']
        disease_info['severity_color'] = severity_info['color']
        disease_info['severity_description'] = severity_info['description']
        
        # Generate heatmap
        heatmap = generate_heatmap(filepath)
        
        # GLCM feature names and values
        glcm_names = ['Contrast', 'Energy', 'Correlation', 
                      'Homogeneity', 'Entropy', 'Variance']
        glcm_features = {
            name: float(val) for name, val in zip(glcm_names, glcm_feats)
        }
        
        # Convert images to base64
        original_b64 = image_to_base64(original_array)
        enhanced_b64 = image_to_base64(enhanced_img)
        heatmap_b64 = image_to_base64(heatmap)
        
        # Return results
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'glcm_features': glcm_features,
            'disease_info': disease_info,
            'severity_info': severity_info,
            'images': {
                'original': original_b64,
                'enhanced': enhanced_b64,
                'heatmap': heatmap_b64
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 OCT Disease Classifier Web App")
    print("=" * 80)
    print("\n📍 Open browser: http://localhost:5000")
    print("\n" + "=" * 80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
