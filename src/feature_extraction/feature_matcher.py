import cv2
import numpy as np



def match_features(desc1, desc2, ratio_threshold=0.7, method='flann'):
    """
    Match features between two images using either FLANN or brute force.
    
    Args:
        desc1, desc2: Feature descriptors from two images.
        ratio_threshold: Lowe's ratio test threshold.
        method: Matching method ('flann' or 'bf').
        
    Returns:
        List of good matches.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    if method.lower() == 'flann':
        # Check if we're working with binary descriptors (ORB)
        if desc1.dtype == np.uint8:
            # FLANN parameters for binary descriptors
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        else:
            # FLANN parameters for SIFT/SURF
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            
        search_params = dict(checks=50)
        
        # Create FLANN matcher
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # Create brute-force matcher
        if desc1.dtype == np.uint8:
            # For binary descriptors like ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # For float descriptors like SIFT/SURF
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Find top 2 matches for each descriptor
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches

def geometric_verification(kp1, kp2, matches, method='fundamental', threshold=3.0):
    """
    Perform geometric verification using RANSAC.
    
    Args:
        kp1, kp2: Keypoints from two images.
        matches: List of matches.
        method: Verification method ('fundamental' or 'homography').
        threshold: RANSAC threshold.
        
    Returns:
        Verified matches and the geometric model.
    """
    if len(matches) < 8:
        return [], None
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    if method.lower() == 'fundamental':
        # Find fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, 0.99)
        
        # Handle case when F is not found
        if F is None or F.shape != (3, 3):
            return [], None
        
        # Select inlier matches
        inlier_mask = mask.ravel().astype(bool)
        inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
        
        return inlier_matches, F
    
    elif method.lower() == 'homography':
        # Find homography with RANSAC
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
        
        # Handle case when H is not found
        if H is None:
            return [], None
        
        # Select inlier matches
        inlier_mask = mask.ravel().astype(bool)
        inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
        
        return inlier_matches, H
    
    else:
        raise ValueError(f"Unsupported verification method: {method}")

def match_image_pairs(features_dict, image_pairs=None, ratio_threshold=0.7, 
                     geometric_verify=True, min_matches=20):
    """
    Match features between pairs of images.
    
    Args:
        features_dict: Dictionary mapping filenames to (keypoints, descriptors).
        image_pairs: List of image filename pairs to match. If None, match consecutive images.
        ratio_threshold: Lowe's ratio test threshold.
        geometric_verify: Whether to perform geometric verification.
        min_matches: Minimum number of matches required.
        
    Returns:
        Dictionary mapping image pairs to matches.
    """
    filenames = list(features_dict.keys())
    
    if image_pairs is None:
        # Match consecutive images by default
        image_pairs = [(filenames[i], filenames[i+1]) for i in range(len(filenames)-1)]
    
    matches_dict = {}
    
    for img1_name, img2_name in image_pairs:
        kp1, desc1 = features_dict[img1_name]
        kp2, desc2 = features_dict[img2_name]
        
        # Match descriptors
        matches = match_features(desc1, desc2, ratio_threshold)
        
        if geometric_verify and len(matches) >= 8:
            matches, _ = geometric_verification(kp1, kp2, matches)
        
        if len(matches) >= min_matches:
            matches_dict[(img1_name, img2_name)] = (kp1, kp2, matches)
    
    return matches_dict