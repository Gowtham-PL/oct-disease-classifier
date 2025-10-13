# features.py
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte


def extract_features(img):
    """
    Extract texture features from grayscale image:
    - GLCM: contrast, energy, correlation, homogeneity
    - Entropy
    - Variance
    Returns a 1D numpy array
    """
    # CRITICAL FIX: Ensure uint8 type for graycomatrix
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    
    # Verify we have uint8
    assert img.dtype == np.uint8, f"Image must be uint8, got {img.dtype}"
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        img, 
        distances=distances, 
        angles=angles, 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    
    # Extract GLCM properties
    feats = [
        graycoprops(glcm, prop).mean() 
        for prop in ['contrast', 'energy', 'correlation', 'homogeneity']
    ]

    # Entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
    hist += 1e-10  # avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    feats.append(entropy)

    # Variance
    feats.append(np.var(img))
    
    return np.array(feats)
