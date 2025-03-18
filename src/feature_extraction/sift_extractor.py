import cv2
import numpy as np

def extract_sift_features(image, n_features=2000, contrast_threshold=0.01, edge_threshold=15):
    """
    Extract SIFT features from an image with enhanced parameters for better coverage.
    
    Args:
        image: Input image (grayscale).
        n_features: Maximum number of features to detect.
        contrast_threshold: Lower value gets more features in low-contrast regions.
        edge_threshold: Higher value keeps more features near edges.
        
    Returns:
        Keypoints and descriptors.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Enhance contrast to bring out more details
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Create SIFT detector with aggressive parameters to extract more features
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,  # Significantly lower to get more features
        edgeThreshold=edge_threshold,          # Higher to keep more edge features
        sigma=1.6                              # Default sigma
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(enhanced_gray, None)
    
    return keypoints, descriptors

def extract_grid_sift_features(image, n_features=2000, grid_size=4):
    """
    Extract SIFT features evenly across the image using a grid-based approach.
    
    Args:
        image: Input image.
        n_features: Target total number of features.
        grid_size: Number of grid cells in each dimension.
        
    Returns:
        Combined keypoints and descriptors from all grid cells.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Enhance contrast to bring out more details
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Create a binary mask for the dinosaur (assuming black background)
    _, dino_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Divide image into grid cells
    h, w = enhanced_gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    # Calculate features per cell (with a bit of overlap allowed)
    features_per_cell = n_features // (grid_size * grid_size) * 2
    
    all_keypoints = []
    all_descriptors = []
    
    # Process each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Define cell boundaries
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            # Create mask for this cell combined with dinosaur mask
            cell_mask = np.zeros_like(enhanced_gray)
            cell_mask[y_start:y_end, x_start:x_end] = 255
            combined_mask = cv2.bitwise_and(cell_mask, dino_mask)
            
            # Skip cells with very little dinosaur content
            if np.sum(combined_mask) / 255 < 100:  # Fewer than 100 white pixels
                continue
            
            # Create SIFT detector for this cell
            cell_sift = cv2.SIFT_create(
                nfeatures=features_per_cell,
                contrastThreshold=0.01,
                edgeThreshold=15
            )
            
            # Extract features in this cell
            cell_keypoints, cell_descriptors = cell_sift.detectAndCompute(enhanced_gray, combined_mask)
            
            if cell_keypoints and len(cell_keypoints) > 0:
                all_keypoints.extend(cell_keypoints)
                if len(all_descriptors) == 0 and cell_descriptors is not None:
                    all_descriptors = cell_descriptors
                elif cell_descriptors is not None:
                    all_descriptors = np.vstack((all_descriptors, cell_descriptors))
    
    # If we have too few features, add more using standard extraction
    if len(all_keypoints) < n_features // 2:
        extra_sift = cv2.SIFT_create(
            nfeatures=n_features - len(all_keypoints),
            contrastThreshold=0.008,  # Even lower threshold to get more features
            edgeThreshold=20
        )
        
        extra_keypoints, extra_descriptors = extra_sift.detectAndCompute(enhanced_gray, dino_mask)
        
        if extra_keypoints and len(extra_keypoints) > 0:
            all_keypoints.extend(extra_keypoints)
            if len(all_descriptors) == 0:
                all_descriptors = extra_descriptors
            else:
                all_descriptors = np.vstack((all_descriptors, extra_descriptors))
    
    return all_keypoints, all_descriptors

def extract_multiscale_sift_features(image, n_features=2000, scales=[0.5, 0.75, 1.5, 2.0]):
    """
    Extract SIFT features at multiple scales for better coverage.
    
    Args:
        image: Input image.
        n_features: Target number of features per scale.
        scales: List of scale factors to use.
        
    Returns:
        Combined keypoints and descriptors from all scales.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create a binary mask for the dinosaur (assuming black background)
    _, dino_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    all_keypoints = []
    all_descriptors = []
    
    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=0.01,
        edgeThreshold=15
    )
    
    # Extract features at each scale
    for scale in scales:
        # Skip scales that are too extreme
        if scale < 0.4 or scale > 2.5:
            continue
            
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        
        # Skip if too small
        if width < 50 or height < 50:
            continue
        
        # Resize image and mask
        resized_img = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(dino_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(resized_img)
        
        # Extract features at this scale
        scale_keypoints, scale_descriptors = sift.detectAndCompute(enhanced_img, resized_mask)
        
        # Adjust keypoint coordinates and size for the original image scale
        if scale_keypoints and len(scale_keypoints) > 0:
            for kp in scale_keypoints:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                kp.size = kp.size / scale
            
            all_keypoints.extend(scale_keypoints)
            
            if len(all_descriptors) == 0 and scale_descriptors is not None:
                all_descriptors = scale_descriptors
            elif scale_descriptors is not None:
                all_descriptors = np.vstack((all_descriptors, scale_descriptors))
    
    return all_keypoints, all_descriptors

def extract_features_from_image_set(images, method='sift', n_features=2000, contrast_threshold=0.009):
    """
    Extract features from a list of images with improved extraction strategies.
    
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
            # Combine multiple extraction strategies for max coverage
            
            # 1. Standard extraction with low threshold
            keypoints1, descriptors1 = extract_sift_features(img, n_features=n_features//2, 
                                                          contrast_threshold=contrast_threshold)
            
            # 2. Grid-based extraction
            keypoints2, descriptors2 = extract_grid_sift_features(img, n_features=n_features//2, 
                                                              grid_size=5)
            
            # 3. Multi-scale extraction
            keypoints3, descriptors3 = extract_multiscale_sift_features(img, n_features=n_features//3,
                                                                    scales=[0.5, 0.75, 1.5, 2.0])
            
            # Combine all keypoints - ensure they're lists
            keypoints = list(keypoints1) if isinstance(keypoints1, tuple) else list(keypoints1)
            if keypoints2:
                keypoints.extend(list(keypoints2) if isinstance(keypoints2, tuple) else keypoints2)
            if keypoints3:
                keypoints.extend(list(keypoints3) if isinstance(keypoints3, tuple) else keypoints3)
            
            # Combine all descriptors
            descriptors = None
            for desc in [descriptors1, descriptors2, descriptors3]:
                if desc is not None and len(desc) > 0:
                    if descriptors is None:
                        descriptors = desc
                    else:
                        descriptors = np.vstack((descriptors, desc))
            
            print(f"{filename}: Extracted {len(keypoints)} keypoints")
            
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