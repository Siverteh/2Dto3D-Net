import cv2
import numpy as np

def extract_surf_features(image, n_features=2000, hessian_threshold=100):
    """
    Extract SURF (Speeded-Up Robust Features) from an image.
    
    Note: SURF is patented and not available in default OpenCV builds.
    If using OpenCV with contrib modules, the xfeatures2d module should be available.
    
    Args:
        image: Input image (grayscale).
        n_features: Number of features to retain (approximate).
        hessian_threshold: Threshold for the keypoint detector. Higher values mean fewer keypoints.
        
    Returns:
        Keypoints and descriptors.
    """
    try:
        # Check if OpenCV was built with xfeatures2d
        cv2.xfeatures2d
    except AttributeError:
        raise ImportError("SURF is not available in your OpenCV build. Try using SIFT or ORB instead.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    
    # Limit number of features if needed
    if len(keypoints) > n_features:
        # Sort keypoints by response (strongest first)
        indices = np.argsort([-kp.response for kp in keypoints])[:n_features]
        keypoints = [keypoints[i] for i in indices]
        descriptors = descriptors[indices]
    
    return keypoints, descriptors

def extract_surf_features_with_config(image, upright=False, extended=False, n_octaves=4, n_octave_layers=3, hessian_threshold=100):
    """
    Extract SURF features with detailed configuration.
    
    Args:
        image: Input image (grayscale).
        upright: If True, don't compute orientation (faster but less rotation invariant).
        extended: If True, use 128-dimensional descriptors instead of 64.
        n_octaves: Number of octaves in the scale pyramid.
        n_octave_layers: Number of layers within each octave.
        hessian_threshold: Threshold for the keypoint detector.
        
    Returns:
        Keypoints and descriptors.
    """
    try:
        # Check if OpenCV was built with xfeatures2d
        cv2.xfeatures2d
    except AttributeError:
        raise ImportError("SURF is not available in your OpenCV build. Try using SIFT or ORB instead.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create SURF detector with custom parameters
    surf = cv2.xfeatures2d.SURF_create(
        hessianThreshold=hessian_threshold,
        nOctaves=n_octaves,
        nOctaveLayers=n_octave_layers,
        extended=extended,
        upright=upright
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def extract_dense_surf_features(image, step_size=20, patch_size=20, hessian_threshold=100):
    """
    Extract SURF features on a dense grid for more uniform coverage.
    
    Args:
        image: Input image (grayscale).
        step_size: Grid step size in pixels.
        patch_size: Size of patches to extract descriptors.
        hessian_threshold: Threshold for the keypoint detector.
        
    Returns:
        Keypoints and descriptors.
    """
    try:
        # Check if OpenCV was built with xfeatures2d
        cv2.xfeatures2d
    except AttributeError:
        raise ImportError("SURF is not available in your OpenCV build. Try using SIFT or ORB instead.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create SURF extractor
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    
    height, width = gray.shape
    half_patch = patch_size // 2
    
    keypoints = []
    
    # Create grid of keypoints
    for y in range(half_patch, height - half_patch, step_size):
        for x in range(half_patch, width - half_patch, step_size):
            keypoints.append(cv2.KeyPoint(x, y, patch_size))
    
    # Compute descriptors at each keypoint location
    _, descriptors = surf.compute(gray, keypoints)
    
    return keypoints, descriptors