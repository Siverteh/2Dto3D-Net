import cv2
import numpy as np

def extract_sift_features(image, n_features=2000):
    """
    Extract SIFT features from an image.
    
    Args:
        image: Input image (grayscale).
        n_features: Maximum number of features to detect.
        
    Returns:
        Keypoints and descriptors.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def extract_features_from_image_set(images, method='sift', n_features=2000):
    """
    Extract features from a list of images.
    
    Args:
        images: List of (image, filename) tuples.
        method: Feature extraction method ('sift', 'surf', or 'orb').
        n_features: Maximum number of features per image.
        
    Returns:
        Dictionary mapping filenames to (keypoints, descriptors) tuples.
    """
    features_dict = {}
    
    for img, filename in images:
        if method.lower() == 'sift':
            keypoints, descriptors = extract_sift_features(img, n_features)
        elif method.lower() == 'surf':
            from .surf_extractor import extract_surf_features
            keypoints, descriptors = extract_surf_features(img, n_features)
        elif method.lower() == 'orb':
            from .orb_extractor import extract_orb_features
            keypoints, descriptors = extract_orb_features(img, n_features)
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
        
        features_dict[filename] = (keypoints, descriptors)
    
    return features_dict