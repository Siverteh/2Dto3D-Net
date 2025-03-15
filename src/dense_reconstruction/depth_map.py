import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def compute_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to an image.
    
    Args:
        image: Input image.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
        
    Returns:
        Filtered image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    return filtered

def compute_stereo_rectification(img1, img2, K, R1, t1, R2, t2):
    """
    Compute stereo rectification for a pair of images.
    
    Args:
        img1, img2: Input images.
        K: Camera intrinsic matrix.
        R1, t1: Rotation and translation for first camera.
        R2, t2: Rotation and translation for second camera.
        
    Returns:
        Rectification maps and Q matrix for disparity-to-depth mapping.
    """
    # Get image dimensions
    h, w = img1.shape[:2]
    
    # Calculate relative pose between cameras
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    # Compute rectification
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, np.zeros(5), K, np.zeros(5), (w, h), R_rel, t_rel, 
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    
    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R1_rect, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R2_rect, P2, (w, h), cv2.CV_32FC1)
    
    return map1x, map1y, map2x, map2y, Q, P1, P2

def rectify_stereo_pair(img1, img2, map1x, map1y, map2x, map2y):
    """
    Rectify a pair of stereo images.
    
    Args:
        img1, img2: Input images.
        map1x, map1y, map2x, map2y: Rectification maps.
        
    Returns:
        Rectified images.
    """
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    return rect1, rect2

def compute_disparity_map(img1_rect, img2_rect, method='sgbm', 
                         min_disparity=0, num_disparities=64, 
                         block_size=5, uniqueness_ratio=10, 
                         speckle_window_size=100, speckle_range=32):
    """
    Compute disparity map between rectified stereo images.
    
    Args:
        img1_rect, img2_rect: Rectified stereo images.
        method: 'sgbm' for Semi-Global Block Matching or 'bm' for Block Matching.
        min_disparity: Minimum possible disparity value.
        num_disparities: Maximum disparity minus minimum disparity.
        block_size: Size of the block window.
        uniqueness_ratio: Margin in percentage by which the best cost value should exceed the second best value.
        speckle_window_size: Maximum size of smooth disparity regions to consider their noise.
        speckle_range: Maximum disparity variation within each connected component.
        
    Returns:
        Disparity map.
    """
    # Convert to grayscale if needed
    if len(img1_rect.shape) == 3:
        img1_gray = cv2.cvtColor(img1_rect, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_rect, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1_rect
        img2_gray = img2_rect
    
    if method.lower() == 'sgbm':
        # Create StereoSGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    else:
        # Create StereoBM object
        stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        stereo.setUniquenessRatio(uniqueness_ratio)
        stereo.setSpeckleWindowSize(speckle_window_size)
        stereo.setSpeckleRange(speckle_range)
    
    # Compute disparity
    disparity = stereo.compute(img1_gray, img2_gray)
    
    # Convert to float and scale
    disparity = disparity.astype(np.float32) / 16.0
    
    return disparity

def filter_disparity_map(disparity, min_disparity=0, max_disparity=64, 
                       wls_lambda=8000, wls_sigma=1.5, use_confidence=True):
    """
    Filter disparity map using WLS filter.
    
    Args:
        disparity: Input disparity map.
        min_disparity: Minimum valid disparity value.
        max_disparity: Maximum valid disparity value.
        wls_lambda: Weight of the smoothness term.
        wls_sigma: Sigma for edge-preserving filter.
        use_confidence: Whether to use confidence-based filtering.
        
    Returns:
        Filtered disparity map.
    """
    # Create mask for valid disparities
    mask = (disparity > min_disparity) & (disparity < max_disparity)
    
    # Apply simple mask-based filtering
    filtered_disparity = disparity.copy()
    filtered_disparity[~mask] = 0
    
    # Apply Gaussian smoothing to valid regions
    smoothed = gaussian_filter(filtered_disparity, sigma=1.0)
    filtered_disparity[mask] = smoothed[mask]
    
    return filtered_disparity

def compute_depth_map(disparity, Q):
    """
    Convert disparity map to depth map.
    
    Args:
        disparity: Input disparity map.
        Q: Disparity-to-depth mapping matrix.
        
    Returns:
        Depth map.
    """
    # Create 3D points from disparity
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Extract Z component as the depth map
    depth_map = points_3d[:, :, 2]
    
    # Create mask for valid depths
    mask = (disparity > 0) & (depth_map > 0) & (depth_map < 1000)
    
    # Set invalid depths to zero
    depth_map[~mask] = 0
    
    return depth_map, mask

def compute_confidence_map(disparity, img1, img2, block_size=5):
    """
    Compute confidence map for disparity values.
    
    Args:
        disparity: Input disparity map.
        img1, img2: Original stereo images.
        block_size: Size of the block for computing confidence.
        
    Returns:
        Confidence map with values between 0 and 1.
    """
    h, w = disparity.shape
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Create confidence map
    confidence = np.zeros((h, w), dtype=np.float32)
    
    # Create border mask
    border = block_size // 2
    mask = np.ones((h, w), dtype=bool)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    
    # Compute confidence based on photometric consistency
    for y in range(border, h - border):
        for x in range(border, w - border):
            if disparity[y, x] <= 0:
                continue
            
            # Calculate corresponding point in right image
            x_right = int(x - disparity[y, x])
            if x_right < border or x_right >= w - border:
                continue
            
            # Extract patches
            patch1 = img1_gray[y-border:y+border+1, x-border:x+border+1]
            patch2 = img2_gray[y-border:y+border+1, x_right-border:x_right+border+1]
            
            # Compute normalized cross-correlation
            ncc = cv2.matchTemplate(patch1, patch2, cv2.TM_CCORR_NORMED)[0, 0]
            confidence[y, x] = max(0, ncc)
    
    return confidence

def estimate_depth_map_for_view(img_ref, imgs_src, K, poses_ref, poses_src, 
                              min_disparity=0, num_disparities=64, 
                              block_size=5, filter_depths=True):
    """
    Estimate depth map for a reference view using multiple source views.
    
    Args:
        img_ref: Reference image.
        imgs_src: List of source images.
        K: Camera intrinsic matrix.
        poses_ref: (R, t) for reference camera.
        poses_src: List of (R, t) for source cameras.
        min_disparity: Minimum disparity value.
        num_disparities: Number of disparity levels.
        block_size: Block size for stereo matching.
        filter_depths: Whether to filter depth maps.
        
    Returns:
        Depth map for reference view and confidence map.
    """
    R_ref, t_ref = poses_ref
    depth_maps = []
    confidence_maps = []
    
    for i, (img_src, (R_src, t_src)) in enumerate(zip(imgs_src, poses_src)):
        print(f"Processing source view {i+1}/{len(imgs_src)}...")
        
        # Compute rectification
        map1x, map1y, map2x, map2y, Q, P1, P2 = compute_stereo_rectification(
            img_ref, img_src, K, R_ref, t_ref, R_src, t_src)
        
        # Rectify images
        img1_rect, img2_rect = rectify_stereo_pair(
            img_ref, img_src, map1x, map1y, map2x, map2y)
        
        # Compute disparity
        disparity = compute_disparity_map(
            img1_rect, img2_rect, 
            min_disparity=min_disparity, 
            num_disparities=num_disparities, 
            block_size=block_size)
        
        # Filter disparity
        if filter_depths:
            disparity = filter_disparity_map(
                disparity, 
                min_disparity=min_disparity, 
                max_disparity=min_disparity + num_disparities)
        
        # Convert to depth
        depth, mask = compute_depth_map(disparity, Q)
        
        # Compute confidence
        confidence = compute_confidence_map(disparity, img1_rect, img2_rect)
        
        # Apply mask to confidence
        confidence[~mask] = 0
        
        depth_maps.append(depth)
        confidence_maps.append(confidence)
    
    # Fuse depth maps
    if len(depth_maps) == 1:
        return depth_maps[0], confidence_maps[0]
    else:
        return fuse_depth_maps(depth_maps, confidence_maps)

def fuse_depth_maps(depth_maps, confidence_maps):
    """
    Fuse multiple depth maps into a single depth map.
    
    Args:
        depth_maps: List of depth maps.
        confidence_maps: List of confidence maps.
        
    Returns:
        Fused depth map and confidence map.
    """
    # Initialize fused maps
    h, w = depth_maps[0].shape
    fused_depth = np.zeros((h, w), dtype=np.float32)
    fused_confidence = np.zeros((h, w), dtype=np.float32)
    
    # Fuse based on confidence-weighted average
    for depth, confidence in zip(depth_maps, confidence_maps):
        # Only consider valid depths
        mask = (depth > 0) & (confidence > 0)
        
        fused_depth[mask] += depth[mask] * confidence[mask]
        fused_confidence[mask] += confidence[mask]
    
    # Normalize by total confidence
    valid_mask = fused_confidence > 0
    fused_depth[valid_mask] /= fused_confidence[valid_mask]
    
    return fused_depth, fused_confidence