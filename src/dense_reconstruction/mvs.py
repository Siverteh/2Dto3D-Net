import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_closing, binary_dilation
from .depth_map import estimate_depth_map_for_view

def select_source_views(target_idx, camera_poses, num_views=2, min_angle=3, max_angle=60):
    """
    Enhanced source view selection optimized for objects captured in a circular pattern.
    
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
    
    # For circular captures, we want to get views from different sides
    # Calculate average camera center (approximate object center)
    positions = []
    for i, (R, t) in enumerate(camera_poses):
        pos = -R.T @ t  # Camera center
        dir = R.T @ np.array([0, 0, 1])  # Looking direction
        positions.append((i, pos, dir))
    
    positions_array = np.array([pos for _, pos, _ in positions])
    center = np.mean(positions_array, axis=0)
    
    # Calculate vectors from center to each camera
    center_to_camera = positions_array - center
    # Normalize these vectors
    norms = np.linalg.norm(center_to_camera, axis=1, keepdims=True)
    center_to_camera = center_to_camera / norms
    
    # Vector from center to target camera
    target_vec = center_to_camera[target_idx]
    
    # Compute scores for each camera
    scores = []
    for i, (idx, pos, dir) in enumerate(positions):
        if idx == target_idx:
            scores.append((idx, float('inf')))  # Don't select the target itself
            continue
        
        # Compute vector from center to this camera
        cam_vec = center_to_camera[i]
        
        # Compute angle between vectors (angular distance around the circle)
        cos_angle = np.clip(np.dot(target_vec, cam_vec), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Create score: prefer cameras at specific angles
        # We want some nearby views (15-20°) for matching and some distant views (60-90°) for triangulation
        if angle < min_angle:  # Too close
            score = 1000 + (min_angle - angle)
        elif angle > max_angle:  # Too far
            score = 1000 + (angle - max_angle)
        elif 15 <= angle <= 25 or 55 <= angle <= 90:  # Ideal ranges
            score = abs(20 - angle) if angle <= 25 else abs(70 - angle)
        else:
            score = min(abs(angle - 20), abs(angle - 70)) + 10  # Other angles with penalty
        
        # Factor in the viewing direction similarity
        dir_similarity = np.dot(target_dir, dir)
        
        # Penalize cameras that don't look at a similar spot
        if dir_similarity < 0.7:  # More than ~45° different view direction
            score += 50  # Add penalty
        
        scores.append((idx, score))
    
    # Sort by score (lowest is best)
    sorted_scores = sorted(scores, key=lambda x: x[1])
    
    # Select views with best scores
    selected_indices = [idx for idx, _ in sorted_scores[:num_views]]
    
    return selected_indices

def compute_depth_maps(images, camera_poses, K, config):
    """
    Compute depth maps for all images with enhanced parameters for shiny objects.
    
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
    
    image_names = list(camera_poses.keys())
    
    # Process each image
    for idx, target_name in enumerate(tqdm(image_names, desc="Computing depth maps")):
        # Get target image
        target_img = next(img for img, name in images if name == target_name)
        target_pose = camera_poses[target_name]
        
        # Preprocess target image if necessary (e.g., for shiny surfaces)
        # This enhancement helps with textureless regions
        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_img.copy()
            
        # Apply CLAHE to enhance texture details
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_gray = clahe.apply(target_gray)
        
        # Select source views using enhanced selection
        source_indices = select_source_views(
            idx, [camera_poses[name] for name in image_names], 
            num_views=num_source_views,
            min_angle=3,  # More permissive minimum angle
            max_angle=60  # Larger maximum angle
        )
        
        if len(source_indices) == 0:
            print(f"No suitable source views found for {target_name}")
            continue
        
        # Get source images and poses
        source_names = [image_names[i] for i in source_indices]
        source_images = [next(img for img, name in images if name == name) for name in source_names]
        source_poses = [camera_poses[name] for name in source_names]
        
        # Preprocess source images similarly
        enhanced_source_images = []
        for source_img in source_images:
            # This is just to ensure we're not modifying the original images
            enhanced_img = source_img.copy()
            enhanced_source_images.append(enhanced_img)
        
        # Estimate depth map with enhanced parameters
        depth_map, conf_map = estimate_depth_map_for_view(
            target_img, enhanced_source_images, K, target_pose, source_poses,
            min_disparity=min_disparity, 
            num_disparities=num_disparities,
            block_size=block_size,
            filter_depths=filter_depths)
        
        # Additional filtering for noisy reconstructions if needed
        # This helps remove isolated noise points while preserving structural features
        if filter_depths:
            # Apply median filter to remove speckle noise
            depth_map = cv2.medianBlur(depth_map, 3)
            
            # Fill small holes using morphological operations
            valid_mask = depth_map > 0
            valid_mask = binary_closing(valid_mask, structure=np.ones((3,3)))
            
            # Remove isolated points
            depth_map[~valid_mask] = 0
        
        # Store results
        depth_maps[target_name] = depth_map
        confidence_maps[target_name] = conf_map
        
        print(f"Computed depth map for {target_name} using source views: {source_names}")
    
    return depth_maps, confidence_maps

def consistency_check(depth_maps, confidence_maps, camera_poses, K, threshold=0.01):
    """
    Enhanced consistency check with special handling for shiny/textureless objects.
    
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
                
                # *** IMPROVEMENT: Depth-dependent threshold ***
                # Allow more error for farther points and less for closer points
                depth_scale_factor = np.ones_like(valid_check, dtype=np.float32)
                depth_scale_factor[z_projected > 30] = 1.5  # 50% more error allowed for deeper points
                
                # Find consistent depths - use adaptive threshold
                consistent = valid_check & (rel_diff < threshold * depth_scale_factor)
                
                # *** IMPROVEMENT: Fallback to more lenient threshold if too few matches ***
                # If we're getting very few consistent points, try a more lenient threshold
                if np.sum(consistent) < 10 and np.sum(valid_check) > 100:
                    consistent = valid_check & (rel_diff < threshold * 5)  # Much more lenient
                
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
        
        # *** IMPROVEMENT: Accept points with any consistency rather than requiring multiple views ***
        # Original required at least 2 views for a point to be considered valid
        valid_mask = consistency_count > 0
        
        filtered_depth[valid_mask] = consistent_depths[valid_mask] / consistency_count[valid_mask]
        
        # *** IMPROVEMENT: Apply morphological operations to clean up the depth map ***
        # This helps remove noise while preserving the structure
        if np.any(valid_mask):
            # Apply closing to fill small holes
            valid_mask_filtered = binary_closing(valid_mask, structure=np.ones((3,3)))
            
            # FIX: Use just dilation instead of trying to get labels
            dilated_mask = binary_dilation(valid_mask_filtered, structure=np.ones((3,3)))
            
            # Only keep points in the filtered mask
            keep_mask = valid_mask & dilated_mask
            filtered_depth[~keep_mask] = 0
        
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
    Complete Multi-View Stereo processing pipeline optimized for shiny/textureless objects.
    
    Args:
        images: List of (image, filename) tuples.
        camera_poses: Dictionary mapping filenames to (R, t).
        K: Camera intrinsic matrix.
        mvs_config: MVS configuration parameters.
        
    Returns:
        Dictionary of depth maps and filtered depth maps.
    """
    print("Starting enhanced Multi-View Stereo processing...")
    print("Using configuration optimized for challenging objects:")
    
    # Set default values optimized for shiny objects if not provided
    enhanced_config = mvs_config.copy()
    
    # Provide some defaults if not specified
    #if 'num_disparities' not in enhanced_config:
    enhanced_config['num_disparities'] = 160  # Increased range for challenging objects
    
    #if 'block_size' not in enhanced_config:
    enhanced_config['block_size'] = 11  # Larger block size for textureless areas
    
    #if 'consistency_threshold' not in enhanced_config:
    enhanced_config['consistency_threshold'] = 0.08  # More forgiving threshold
    
    #if 'num_source_views' not in enhanced_config:
    enhanced_config['num_source_views'] = 4  # Use more views for better triangulation
    
    # Report the configuration being used
    for key, value in enhanced_config.items():
        print(f"  - {key}: {value}")
    
    # Compute initial depth maps with enhanced settings
    depth_maps, confidence_maps = compute_depth_maps(
        images, camera_poses, K, enhanced_config)
    
    # Perform enhanced consistency check
    filtered_depth_maps = consistency_check(
        depth_maps, confidence_maps, camera_poses, K, 
        threshold=enhanced_config.get('consistency_threshold', 0.08))
    
    # Return both raw and filtered depth maps
    return {
        'raw_depth_maps': depth_maps,
        'confidence_maps': confidence_maps,
        'filtered_depth_maps': filtered_depth_maps
    }