import cv2
import numpy as np
from tqdm import tqdm
from .depth_map import estimate_depth_map_for_view

def select_source_views(target_idx, camera_poses, num_views=2, min_angle=5, max_angle=40):
    """
    Select source views for a target view based on viewing angle.
    
    Args:
        target_idx: Index of the target view.
        camera_poses: List of camera poses (R, t).
        num_views: Number of source views to select.
        min_angle: Minimum triangulation angle in degrees.
        max_angle: Maximum triangulation angle in degrees.
        
    Returns:
        Indices of selected source views.
    """
    n_images = len(camera_poses)
    
    # Skip if not enough images
    if n_images <= 1:
        return []
    
    target_R, target_t = camera_poses[target_idx]
    target_pos = -target_R.T @ target_t
    target_dir = target_R.T @ np.array([0, 0, 1])  # Camera looks along Z axis
    
    # Compute viewing angles between target and all other views
    angles = []
    for i, (R, t) in enumerate(camera_poses):
        if i == target_idx:
            angles.append(float('inf'))  # Don't select the target itself
            continue
        
        # Compute position and direction of source camera
        source_pos = -R.T @ t
        source_dir = R.T @ np.array([0, 0, 1])
        
        # Compute baseline vector
        baseline = source_pos - target_pos
        baseline_length = np.linalg.norm(baseline)
        
        if baseline_length < 1e-6:
            angles.append(float('inf'))  # Cameras at same position
            continue
        
        # Compute angle between viewing directions
        cos_angle = np.dot(target_dir, source_dir)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Penalize very small or very large angles
        if angle < min_angle or angle > max_angle:
            angle += 1000  # Add penalty
        
        angles.append(angle)
    
    # Select views with smallest angles
    sorted_indices = np.argsort(angles)
    selected_indices = sorted_indices[:num_views]
    
    return selected_indices

def compute_depth_maps(images, camera_poses, K, config):
    """
    Compute depth maps for all images.
    
    Args:
        images: List of images.
        camera_poses: Dictionary mapping image names to (R, t).
        K: Camera intrinsic matrix.
        config: Configuration parameters.
        
    Returns:
        Dictionary of depth maps, confidence maps, and fused depth maps.
    """
    # Extract parameters from config
    min_disparity = config.get('min_disparity', 0)
    num_disparities = config.get('num_disparities', 64)
    block_size = config.get('block_size', 5)
    filter_depths = config.get('filter_depths', True)
    num_source_views = config.get('num_source_views', 2)
    
    depth_maps = {}
    confidence_maps = {}
    fused_depth_maps = {}
    
    image_names = list(camera_poses.keys())
    
    # Process each image
    for idx, target_name in enumerate(tqdm(image_names, desc="Computing depth maps")):
        # Get target image
        target_img = next(img for img, name in images if name == target_name)
        target_pose = camera_poses[target_name]
        
        # Select source views
        source_indices = select_source_views(
            idx, [camera_poses[name] for name in image_names], 
            num_views=num_source_views)
        
        if not source_indices:
            print(f"No suitable source views found for {target_name}")
            continue
        
        # Get source images and poses
        source_names = [image_names[i] for i in source_indices]
        source_images = [next(img for img, name in images if name == name) for name in source_names]
        source_poses = [camera_poses[name] for name in source_names]
        
        # Estimate depth map
        depth_map, conf_map = estimate_depth_map_for_view(
            target_img, source_images, K, target_pose, source_poses,
            min_disparity=min_disparity, 
            num_disparities=num_disparities,
            block_size=block_size,
            filter_depths=filter_depths)
        
        # Store results
        depth_maps[target_name] = depth_map
        confidence_maps[target_name] = conf_map
        
        print(f"Computed depth map for {target_name} using source views: {source_names}")
    
    return depth_maps, confidence_maps

def consistency_check(depth_maps, confidence_maps, camera_poses, K, threshold=0.01):
    """
    Perform consistency check across depth maps.
    
    Args:
        depth_maps: Dictionary of depth maps.
        confidence_maps: Dictionary of confidence maps.
        camera_poses: Dictionary of camera poses.
        K: Camera intrinsic matrix.
        threshold: Relative depth difference threshold.
        
    Returns:
        Dictionary of filtered depth maps.
    """
    filtered_depth_maps = {}
    image_names = list(depth_maps.keys())
    
    for idx, reference_name in enumerate(tqdm(image_names, desc="Consistency check")):
        # Get reference depth map and pose
        reference_depth = depth_maps[reference_name]
        reference_R, reference_t = camera_poses[reference_name]
        
        # Create mask for consistent depths
        h, w = reference_depth.shape
        consistency_mask = np.zeros((h, w), dtype=bool)
        
        # Count number of consistent observations for each pixel
        consistent_count = np.zeros((h, w), dtype=np.int32)
        
        # Initialize filtered depth map
        filtered_depth = np.zeros((h, w), dtype=np.float32)
        
        # Process each pixel in reference depth map
        for y in range(h):
            for x in range(w):
                # Skip invalid depths
                if reference_depth[y, x] <= 0:
                    continue
                
                # Get 3D point in reference camera coordinates
                depth = reference_depth[y, x]
                point_ref = np.array([
                    (x - K[0, 2]) * depth / K[0, 0],
                    (y - K[1, 2]) * depth / K[1, 1],
                    depth
                ])
                
                # Transform to world coordinates
                point_world = reference_R @ point_ref + reference_t
                
                # Count consistent observations
                count = 0
                sum_depth = 0
                
                # Check consistency with other depth maps
                for target_name in image_names:
                    if target_name == reference_name:
                        continue
                    
                    target_R, target_t = camera_poses[target_name]
                    target_depth = depth_maps[target_name]
                    
                    # Transform point to target camera coordinates
                    point_target = target_R.T @ (point_world - target_t)
                    
                    # Skip points behind camera
                    if point_target[2] <= 0:
                        continue
                    
                    # Project to target image
                    x_target = int(K[0, 0] * point_target[0] / point_target[2] + K[0, 2])
                    y_target = int(K[1, 1] * point_target[1] / point_target[2] + K[1, 2])
                    
                    # Skip points outside image
                    if (x_target < 0 or x_target >= w or 
                        y_target < 0 or y_target >= h):
                        continue
                    
                    # Get depth in target image
                    depth_target = target_depth[y_target, x_target]
                    
                    # Skip invalid depths
                    if depth_target <= 0:
                        continue
                    
                    # Check depth consistency
                    rel_diff = abs(point_target[2] - depth_target) / point_target[2]
                    if rel_diff < threshold:
                        count += 1
                        sum_depth += point_target[2]
                
                # Mark pixel as consistent if observed in enough views
                if count >= 1:
                    consistency_mask[y, x] = True
                    consistent_count[y, x] = count
                    filtered_depth[y, x] = sum_depth / count
        
        # Apply consistency mask
        filtered_depth_maps[reference_name] = filtered_depth * consistency_mask
        
        # Print statistics
        valid_pixels = np.sum(reference_depth > 0)
        consistent_pixels = np.sum(consistency_mask)
        if valid_pixels > 0:
            print(f"{reference_name}: {consistent_pixels}/{valid_pixels} pixels consistent ({consistent_pixels/valid_pixels:.2%})")
    
    return filtered_depth_maps

def process_mvs(images, camera_poses, K, mvs_config):
    """
    Complete Multi-View Stereo processing pipeline.
    
    Args:
        images: List of (image, filename) tuples.
        camera_poses: Dictionary mapping filenames to (R, t).
        K: Camera intrinsic matrix.
        mvs_config: MVS configuration parameters.
        
    Returns:
        Dictionary of depth maps and filtered depth maps.
    """
    print("Starting Multi-View Stereo processing...")
    
    # Compute initial depth maps
    depth_maps, confidence_maps = compute_depth_maps(
        images, camera_poses, K, mvs_config)
    
    # Perform consistency check
    filtered_depth_maps = consistency_check(
        depth_maps, confidence_maps, camera_poses, K, 
        threshold=mvs_config.get('consistency_threshold', 0.01))
    
    # Return both raw and filtered depth maps
    return {
        'raw_depth_maps': depth_maps,
        'confidence_maps': confidence_maps,
        'filtered_depth_maps': filtered_depth_maps
    }