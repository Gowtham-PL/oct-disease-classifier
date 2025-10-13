# preprocess.py
from skimage import io
from skimage.exposure import equalize_adapthist
import numpy as np
from PIL import Image


def preprocess_image_gray(path):
    """
    Preprocess OCT image - returns uint8 numpy array
    """
    # Use PIL to ensure clean uint8 loading
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = np.array(img)  # Convert to numpy array (uint8)
    
    # Apply CLAHE
    img_enhanced = equalize_adapthist(img, clip_limit=0.03)
    
    # Convert back to uint8 (CLAHE returns float 0-1)
    img_final = (img_enhanced * 255).astype(np.uint8)
    
    return img_final
