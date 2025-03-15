import cv2
import numpy as np

def extract_orb_features(image, n_features=2000, scale_factor=1.2, n_levels=8, edge_threshold=31):
    """
    Extract ORB (Oriented FAST and Rotated BRIEF) features from an image.
    
    Args:
        image: Input image (grayscale).
        n_features: Maximum number of features to detect.
        scale_factor: Pyramid decimation ratio.
        n_levels: Number of pyramid levels.
        edge_threshold: Size of the border where features are not detected.
        
    Returns:
        Keypoints and descriptors.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def adaptive_non_maximal_suppression(keypoints, descriptors, num_to_retain=1000, radius=10):
    """
    Apply Adaptive Non-Maximal Suppression (ANMS) to select a well-distributed
    subset of keypoints.
    
    Args:
        keypoints: List of keypoints.
        descriptors: Feature descriptors.
        num_to_retain: Number of keypoints to retain.
        radius: Suppression radius.
        
    Returns:
        Filtered keypoints and descriptors.
    """
    if len(keypoints) <= num_to_retain:
        return keypoints, descriptors
    
    # Extract keypoint responses (strength) and positions
    responses = np.array([kp.response for kp in keypoints])
    positions = np.array([kp.pt for kp in keypoints])
    
    # Sort keypoints by response (strongest first)
    indices = np.argsort(-responses)
    
    # Initialize variables
    selected_indices = []
    selected_positions = []
    
    # Select keypoints
    for idx in indices:
        pt = positions[idx]
        
        # Skip if too close to existing points
        too_close = False
        for sel_pt in selected_positions:
            dist = np.sqrt((pt[0] - sel_pt[0])**2 + (pt[1] - sel_pt[1])**2)
            if dist < radius:
                too_close = True
                break
        
        if not too_close:
            selected_indices.append(idx)
            selected_positions.append(pt)
        
        # Stop when enough points have been selected
        if len(selected_indices) >= num_to_retain:
            break
    
    # Extract selected keypoints and descriptors
    selected_keypoints = [keypoints[i] for i in selected_indices]
    selected_descriptors = descriptors[selected_indices]
    
    return selected_keypoints, selected_descriptors

def extract_distributed_orb_features(image, n_features=2000, anms_radius=10):
    """
    Extract ORB features with better spatial distribution using ANMS.
    
    Args:
        image: Input image (grayscale).
        n_features: Maximum number of features to detect.
        anms_radius: Radius for adaptive non-maximal suppression.
        
    Returns:
        Keypoints and descriptors.
    """
    # Extract more features than needed
    keypoints, descriptors = extract_orb_features(image, n_features * 2)
    
    # Apply ANMS to get better distribution
    if len(keypoints) > n_features:
        keypoints, descriptors = adaptive_non_maximal_suppression(
            keypoints, descriptors, n_features, anms_radius)
    
    return keypoints, descriptors

def extract_orb_features_with_mask(image, mask=None, n_features=2000):
    """
    Extract ORB features with an optional mask to restrict detection regions.
    
    Args:
        image: Input image (grayscale).
        mask: Optional binary mask. Features are only detected where mask is non-zero.
        n_features: Maximum number of features to detect.
        
    Returns:
        Keypoints and descriptors.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    
    return keypoints, descriptors