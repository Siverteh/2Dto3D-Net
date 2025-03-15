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
        
        if len(source_indices) == 0:
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
    Optimized consistency check across depth maps using vectorized operations.
    
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
    
    # Precompute camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for idx, reference_name in enumerate(tqdm(image_names, desc="Consistency check")):
        # Get reference depth map and pose
        reference_depth = depth_maps[reference_name]
        reference_R, reference_t = camera_poses[reference_name]
        
        h, w = reference_depth.shape
        
        # Create arrays to store results
        consistent_depths = np.zeros((h, w), dtype=np.float32)
        consistency_count = np.zeros((h, w), dtype=np.int32)
        
        # Find valid depths in reference image
        valid_mask = reference_depth > 0
        y_coords, x_coords = np.where(valid_mask)
        
        # Skip if no valid depths
        if len(y_coords) == 0:
            filtered_depth_maps[reference_name] = np.zeros_like(reference_depth)
            continue
        
        # Get depths at valid pixels
        depths = reference_depth[valid_mask]
        
        # Create batch of 3D points (more efficient to process in smaller batches)
        batch_size = 10000  # Process points in batches to avoid memory issues
        for batch_start in range(0, len(y_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(y_coords))
            batch_y = y_coords[batch_start:batch_end]
            batch_x = x_coords[batch_start:batch_end]
            batch_depths = depths[batch_start:batch_end]
            
            # Convert pixels to 3D points in reference camera coordinates
            batch_points = np.zeros((len(batch_y), 3))
            batch_points[:, 0] = (batch_x - cx) * batch_depths / fx
            batch_points[:, 1] = (batch_y - cy) * batch_depths / fy
            batch_points[:, 2] = batch_depths
            
            # Transform to world coordinates (vectorized)
            batch_points_world = reference_R @ batch_points.T + reference_t.reshape(3, 1)
            batch_points_world = batch_points_world.T  # Shape becomes (N, 3)
            
            # Check consistency with other views
            for target_name in image_names:
                if target_name == reference_name:
                    continue
                
                target_R, target_t = camera_poses[target_name]
                target_depth = depth_maps[target_name]
                
                # Transform points to target camera coordinates (vectorized)
                R_target_inv = target_R.T
                points_in_target = R_target_inv @ (batch_points_world.T - target_t.reshape(3, 1))
                points_in_target = points_in_target.T  # Shape becomes (N, 3)
                
                # Check if points are in front of camera
                valid_z = points_in_target[:, 2] > 0
                
                if not np.any(valid_z):
                    continue
                
                # Project valid points to target image coordinates
                proj_x = np.zeros_like(valid_z, dtype=np.int32)
                proj_y = np.zeros_like(valid_z, dtype=np.int32)
                
                # Only project points with valid z
                z_valid = points_in_target[valid_z, 2]
                
                proj_x[valid_z] = np.round(fx * points_in_target[valid_z, 0] / z_valid + cx).astype(np.int32)
                proj_y[valid_z] = np.round(fy * points_in_target[valid_z, 1] / z_valid + cy).astype(np.int32)
                
                # Check which projections are within image bounds
                in_bounds = (proj_x >= 0) & (proj_x < w) & (proj_y >= 0) & (proj_y < h) & valid_z
                
                if not np.any(in_bounds):
                    continue
                
                # Get the depths in target image at projected coordinates
                target_depths = np.zeros_like(in_bounds, dtype=np.float32)
                target_depths[in_bounds] = target_depth[proj_y[in_bounds], proj_x[in_bounds]]
                
                # Only consider valid depths in target image
                valid_target = target_depths > 0
                
                if not np.any(valid_target):
                    continue
                
                # Calculate relative depth difference for all valid projections
                z_projected = points_in_target[:, 2]
                valid_check = valid_target & (z_projected > 0)
                
                # Calculate relative depth difference
                rel_diff = np.zeros_like(valid_check, dtype=np.float32)
                rel_diff[valid_check] = np.abs(z_projected[valid_check] - target_depths[valid_check]) / z_projected[valid_check]
                
                # Find consistent depths
                consistent = valid_check & (rel_diff < threshold)
                
                if not np.any(consistent):
                    continue
                
                # Update consistency count and sum for valid points
                for i in range(len(batch_y)):
                    if consistent[i]:
                        y, x = batch_y[i], batch_x[i]
                        consistency_count[y, x] += 1
                        consistent_depths[y, x] += z_projected[i]
        
        # Create filtered depth map
        filtered_depth = np.zeros((h, w), dtype=np.float32)
        valid_mask = consistency_count > 0
        filtered_depth[valid_mask] = consistent_depths[valid_mask] / consistency_count[valid_mask]
        
        # Store result
        filtered_depth_maps[reference_name] = filtered_depth
        
        # Print statistics
        valid_pixels = np.sum(reference_depth > 0)
        consistent_pixels = np.sum(valid_mask)
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