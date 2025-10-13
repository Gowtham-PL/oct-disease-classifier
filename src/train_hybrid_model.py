# train_hybrid_model_optimized_final.py
"""
Optimized Hybrid Model for OCT Image Classification
Combines EfficientNetB0 deep features with GLCM texture features
Incorporates latest best practices for medical image classification
"""

import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Input,
                                      GlobalAveragePooling2D, Concatenate, 
                                      Multiply, Reshape, Activation, Add)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                         ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Custom imports
from preprocess import preprocess_image_gray
from features import extract_features
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = "data/OCT2017/train"
TEST_PATH = "data/OCT2017/test"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 0.001
MIN_LR = 1e-7

print("=" * 80)
print("OCT IMAGE CLASSIFICATION - HYBRID MODEL")
print("=" * 80)
print(f"Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Classes: {CLASSES}")
print("=" * 80)

# ============================================================================
# ATTENTION & ENHANCEMENT BLOCKS
# ============================================================================
def squeeze_excitation_block(input_tensor, ratio=16, name='se_block'):
    """
    Squeeze-and-Excitation block for channel attention
    Improves feature representation by 3-5% in medical imaging
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global average pooling
    se = GlobalAveragePooling2D(name=f'{name}_gap')(input_tensor)
    se = Reshape((1, 1, channels), name=f'{name}_reshape')(se)
    
    # Excitation: FC layers
    se = Dense(channels // ratio, activation='relu', 
               kernel_initializer='he_normal', name=f'{name}_fc1')(se)
    se = Dense(channels, activation='sigmoid', 
               kernel_initializer='he_normal', name=f'{name}_fc2')(se)
    
    # Scale
    se = Multiply(name=f'{name}_multiply')([input_tensor, se])
    return se

# ============================================================================
# LOAD EFFICIENTNET WITH FIX FOR KERAS 3.8
# ============================================================================
def load_efficientnet_fixed():
    """
    Load EfficientNetB0 - simpler direct loading
    """
    print("\n" + "=" * 80)
    print("LOADING EFFICIENTNET FEATURE EXTRACTOR")
    print("=" * 80)
    
    # Delete any corrupted cached file
    import shutil
    cache_path = os.path.expanduser('~/.keras/models/efficientnetb0_notop.h5')
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print("✓ Removed old cached file")
        except:
            pass
    
    # Load directly - Keras will download fresh
    try:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        print("✓ Successfully loaded ImageNet pretrained weights")
    except Exception as e:
        print(f"⚠ Error loading weights: {e}")
        print("⚠ Continuing without pretrained weights")
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    
    # Add SE block
    x = base_model.output
    x = squeeze_excitation_block(x, ratio=16, name='effnet_se')
    x = GlobalAveragePooling2D(name='effnet_gap')(x)
    x = BatchNormalization(name='effnet_bn')(x)
    x = Dropout(0.3, name='effnet_dropout')(x)
    
    enhanced_model = Model(inputs=base_model.input, outputs=x, name='EfficientNet_Enhanced')
    
    # Fine-tune last 30 layers
    for layer in enhanced_model.layers[:-30]:
        layer.trainable = False
    for layer in enhanced_model.layers[-30:]:
        layer.trainable = True
    
    print(f"✓ Model loaded with {sum([1 for l in enhanced_model.layers if l.trainable])} trainable layers")
    print("=" * 80)
    
    return enhanced_model


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================
def extract_effnet_features(img_path, model, cache={}):
    """
    Extract EfficientNet features from grayscale OCT images
    Converts grayscale to RGB by channel replication
    """
    if img_path in cache:
        return cache[img_path]
    
    # Load as grayscale
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    img_array = img_to_array(img)
    
    # Convert to RGB (replicate channels)
    img_rgb = np.concatenate([img_array] * 3, axis=-1)
    
    # Preprocess for EfficientNet
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    
    # Extract features
    features = model.predict(img_rgb, verbose=0)
    features = features.flatten()
    
    cache[img_path] = features
    return features

# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================
def get_advanced_augmentation():
    """
    State-of-the-art augmentation for medical images
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,  # OCT images shouldn't be flipped vertically
        fill_mode='reflect',
        brightness_range=[0.8, 1.2]
    )

# ============================================================================
# DATA LOADING WITH OPTIMIZATION
# ============================================================================
def load_dataset(data_path, effnet_model, augment=True, aug_factor=2, max_images_per_class=5000):
    """
    Load and process dataset with efficient feature extraction
    max_images_per_class: Limit number of images to process per class
    """
    X_glcm_list = []
    X_effnet_list = []
    y_list = []
    
    datagen = get_advanced_augmentation() if augment else None
    
    for cls in CLASSES:
        cls_folder = os.path.join(data_path, cls)
        files = [f for f in os.listdir(cls_folder) 
                 if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        # LIMIT TO max_images_per_class
        if len(files) > max_images_per_class:
            print(f"\n  Found {len(files)} images in {cls}, using first {max_images_per_class}")
            files = files[:max_images_per_class]
        else:
            print(f"\n  Processing {cls}: {len(files)} images")
        
        for filename in tqdm(files, desc=f"  {cls}", ncols=80):
            img_path = os.path.join(cls_folder, filename)
            
            try:
                # Extract EfficientNet features (RGB) - computed once
                effnet_feats = extract_effnet_features(img_path, effnet_model)
                
                # Extract GLCM features (grayscale)
                gray_img = preprocess_image_gray(img_path)
                
                # Ensure uint8 type
                if gray_img.dtype != np.uint8:
                    gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
                
                glcm_feats = extract_features(gray_img)
                
                # Original sample
                X_glcm_list.append(glcm_feats)
                X_effnet_list.append(effnet_feats)
                y_list.append(cls)
                
                # Augmented samples (only augment GLCM, reuse EfficientNet)
                if augment and datagen:
                    gray_3d = np.expand_dims(gray_img, axis=-1)
                    for _ in range(aug_factor):
                        # Apply augmentation
                        aug_img = datagen.random_transform(gray_3d).squeeze()
                        
                        # CRITICAL: Ensure uint8 after augmentation
                        if aug_img.dtype != np.uint8:
                            # Augmentation returns float, convert to uint8
                            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                        
                        # Extract features from augmented image
                        aug_glcm = extract_features(aug_img)
                        
                        X_glcm_list.append(aug_glcm)
                        X_effnet_list.append(effnet_feats)  # Reuse
                        y_list.append(cls)
                
            except Exception as e:
                print(f"\n  ⚠ Skipped {filename}: {e}")
                continue
    
    return np.array(X_glcm_list), np.array(X_effnet_list), np.array(y_list)



# ============================================================================
# HYBRID FUSION MODEL WITH ATTENTION
# ============================================================================
def build_hybrid_fusion_model(glcm_dim, effnet_dim, num_classes):
    """
    Advanced hybrid model with dual-branch architecture and attention fusion
    """
    # Input layers
    input_glcm = Input(shape=(glcm_dim,), name='glcm_input')
    input_effnet = Input(shape=(effnet_dim,), name='effnet_input')
    
    # ========================================
    # GLCM Branch (Texture Features)
    # ========================================
    glcm_branch = Dense(128, activation='relu', 
                        kernel_regularizer=l2(0.001),
                        name='glcm_dense1')(input_glcm)
    glcm_branch = BatchNormalization(name='glcm_bn1')(glcm_branch)
    glcm_branch = Dropout(0.3, name='glcm_drop1')(glcm_branch)
    
    glcm_branch = Dense(64, activation='relu',
                        kernel_regularizer=l2(0.001),
                        name='glcm_dense2')(glcm_branch)
    glcm_branch = BatchNormalization(name='glcm_bn2')(glcm_branch)
    glcm_branch = Dropout(0.2, name='glcm_drop2')(glcm_branch)
    
    # ========================================
    # EfficientNet Branch (Deep Features)
    # ========================================
    effnet_branch = Dense(512, activation='relu',
                          kernel_regularizer=l2(0.001),
                          name='effnet_dense1')(input_effnet)
    effnet_branch = BatchNormalization(name='effnet_bn1')(effnet_branch)
    effnet_branch = Dropout(0.4, name='effnet_drop1')(effnet_branch)
    
    effnet_branch = Dense(256, activation='relu',
                          kernel_regularizer=l2(0.001),
                          name='effnet_dense2')(effnet_branch)
    effnet_branch = BatchNormalization(name='effnet_bn2')(effnet_branch)
    effnet_branch = Dropout(0.3, name='effnet_drop2')(effnet_branch)
    
    # ========================================
    # Feature Fusion with Attention
    # ========================================
    # Concatenate features
    merged = Concatenate(name='feature_fusion')([glcm_branch, effnet_branch])
    
    # Attention mechanism for adaptive feature weighting
    attention = Dense(merged.shape[-1], activation='sigmoid',
                     name='attention_weights')(merged)
    attended_features = Multiply(name='attended_features')([merged, attention])
    
    # ========================================
    # Classification Head
    # ========================================
    x = Dense(512, activation='relu',
              kernel_regularizer=l2(0.001),
              name='classifier_dense1')(attended_features)
    x = BatchNormalization(name='classifier_bn1')(x)
    x = Dropout(0.5, name='classifier_drop1')(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(0.001),
              name='classifier_dense2')(x)
    x = BatchNormalization(name='classifier_bn2')(x)
    x = Dropout(0.4, name='classifier_drop2')(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=l2(0.001),
              name='classifier_dense3')(x)
    x = Dropout(0.3, name='classifier_drop3')(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Build model
    model = Model(inputs=[input_glcm, input_effnet], outputs=output, 
                  name='Hybrid_OCT_Classifier')
    
    return model

# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def cosine_decay_with_warmup(epoch, lr):
    """
    Cosine annealing with warmup for better convergence
    """
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
        return MIN_LR + (INITIAL_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress))

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    # Load EfficientNet feature extractor
    effnet_model = load_efficientnet_fixed()
    
    # Load training data
    print("\n" + "=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    X_glcm_train, X_effnet_train, y_train = load_dataset(
        TRAIN_PATH, effnet_model, augment=True, aug_factor=2
    )
    
    # Load test data
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    X_glcm_test, X_effnet_test, y_test = load_dataset(
        TEST_PATH, effnet_model, augment=False
    )
    
    # Encode labels
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(y_train_enc)}")
    print(f"  Test samples: {len(y_test_enc)}")
    print(f"  Class distribution: {dict(zip(CLASSES, np.bincount(y_train_enc)))}")
    
    # Scale features independently
    scaler_glcm = StandardScaler()
    scaler_effnet = StandardScaler()
    
    X_glcm_train = scaler_glcm.fit_transform(X_glcm_train)
    X_glcm_test = scaler_glcm.transform(X_glcm_test)
    
    X_effnet_train = scaler_effnet.fit_transform(X_effnet_train)
    X_effnet_test = scaler_effnet.transform(X_effnet_test)
    
    print("✓ Features normalized")
    
    # Train-validation split
    X_glcm_train, X_glcm_val, X_effnet_train, X_effnet_val, y_train_enc, y_val_enc = train_test_split(
        X_glcm_train, X_effnet_train, y_train_enc,
        test_size=0.15, random_state=42, stratify=y_train_enc
    )
    
    print(f"  Validation samples: {len(y_val_enc)}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"  Class weights: {class_weight_dict}")
    
    # Build hybrid model
    print("\n" + "=" * 80)
    print("BUILDING HYBRID MODEL")
    print("=" * 80)
    model = build_hybrid_fusion_model(
        glcm_dim=X_glcm_train.shape[1],
        effnet_dim=X_effnet_train.shape[1],
        num_classes=len(CLASSES)
    )
    
    # Compile model
    optimizer = Adam(learning_rate=INITIAL_LR, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    os.makedirs("models", exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'models/best_hybrid_oct_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=MIN_LR,
            verbose=1
        ),
        LearningRateScheduler(cosine_decay_with_warmup, verbose=1)
    ]
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    history = model.fit(
        [X_glcm_train, X_effnet_train],
        y_train_enc,
        validation_data=([X_glcm_val, X_effnet_val], y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    
    y_pred_probs = model.predict([X_glcm_test, X_effnet_test], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 80)
    print(classification_report(y_test_enc, y_pred, 
                                target_names=le.classes_, 
                                digits=4))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_test_enc, y_pred)
    print(cm)
    
    # Overall metrics
    test_loss, test_acc = model.evaluate(
        [X_glcm_test, X_effnet_test], 
        y_test_enc, 
        verbose=0
    )
    
    print("\n" + "=" * 80)
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"FINAL TEST LOSS: {test_loss:.4f}")
    print("=" * 80)
    
    # Save all artifacts
    print("\n" + "=" * 80)
    print("SAVING MODELS AND ARTIFACTS")
    print("=" * 80)
    
    model.save("models/oct_hybrid_optimized_final.h5")
    effnet_model.save("models/effnet_feature_extractor.h5")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(scaler_glcm, "models/scaler_glcm.pkl")
    joblib.dump(scaler_effnet, "models/scaler_effnet.pkl")
    
    print("✓ Model saved: models/oct_hybrid_optimized_final.h5")
    print("✓ Feature extractor saved: models/effnet_feature_extractor.h5")
    print("✓ Scalers and encoders saved")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
