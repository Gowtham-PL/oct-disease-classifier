# Trained Model Files

Due to GitHub's file size limitations (100MB), the trained model files are hosted separately.

## Required Files

You need these 5 files to run the application:

1. **oct_hybrid_optimized_final.h5** (~85 MB)
   - Main hybrid classifier model

2. **effnet_feature_extractor.h5** (~45 MB)
   - EfficientNetB0 feature extractor

3. **scaler_glcm.pkl** (~2 KB)
   - GLCM feature scaler

4. **scaler_effnet.pkl** (~5 KB)
   - EfficientNet feature scaler

5. **label_encoder.pkl** (~1 KB)
   - Disease class label encoder

## Download Options
###GOOGLE DRIVE:
https://drive.google.com/drive/folders/1TPVd1zS01nICwz4AQRiIoN9SqaN9qZli?usp=sharing

## Installation

After downloading, your `models/` folder should look like this:

models/
├── oct_hybrid_optimized_final.h5
├── effnet_feature_extractor.h5
├── scaler_glcm.pkl
├── scaler_effnet.pkl
├── label_encoder.pkl
└── README.md (this file)

## Model Details

### Training Configuration

- **Dataset**: OCT2017 (84,495 images)
- **Training Images**: 60,000 (with augmentation)
- **Validation Split**: 20%
- **Batch Size**: 32
- **Epochs**: 100 (early stopping applied)
- **Optimizer**: Adam
- **Learning Rate**: 0.001

### Performance

- **Test Accuracy**: 94%
- **Training Time**: ~3 hours (GPU)
- **Inference Time**: ~3 seconds per image (CPU)

