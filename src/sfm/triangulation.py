import cv2
import numpy as np

def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from 2D correspondences.
    
    Args:
        P1, P2: Projection matrices for two cameras.
        pts1, pts2: Matched image points.
        
    Returns:
        3D points in homogeneous coordinates.
    """
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T

def compute_triangulation_angle(pt3d, camera1_center, camera2_center):
    """
    Compute triangulation angle between two cameras and a 3D point.
    
    Args:
        pt3d: 3D point coordinates.
        camera1_center: First camera center.
        camera2_center: Second camera center.
        
    Returns:
        Triangulation angle in degrees.
    """
    # Vectors from 3D point to camera centers
    vec1 = camera1_center - pt3d
    vec2 = camera2_center - pt3d
    
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def compute_reprojection_error(points_3d, points_2d, camera_matrix, rvec, tvec):
    """
    Compute reprojection error for 3D points.
    
    Args:
        points_3d: 3D points.
        points_2d: Original 2D points.
        camera_matrix: Camera intrinsic matrix.
        rvec, tvec: Camera rotation and translation vectors.
        
    Returns:
        Mean reprojection error.
    """
    # Project 3D points back to the image
    points_2d_proj, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    points_2d_proj = points_2d_proj.reshape(-1, 2)
    
    # Calculate Euclidean distance between original and reprojected points
    errors = np.sqrt(np.sum((points_2d - points_2d_proj) ** 2, axis=1))
    
    return errors

def filter_triangulated_points(points_3d, errors, threshold=4.0):
    """
    Filter triangulated points based on reprojection error.
    
    Args:
        points_3d: 3D points.
        errors: Reprojection errors.
        threshold: Maximum allowed reprojection error.
        
    Returns:
        Filtered 3D points.
    """
    # Create mask for points with low reprojection error
    mask = errors < threshold
    
    # Apply mask
    filtered_points = points_3d[mask]
    
    return filtered_points

def triangulate_all_points(camera_poses, feature_matches, K):
    """
    Triangulate points across multiple views.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        feature_matches: Dictionary of feature matches {(img1, img2): (kp1, kp2, matches)}.
        K: Camera intrinsic matrix.
        
    Returns:
        Dictionary of 3D points and their observations.
    """
    # Initialize dictionaries
    points_3d = []  # List to store all triangulated 3D points
    point_colors = []  # List to store point colors (if available)
    point_observations = []  # List to store image observations for each point
    
    # Process each image pair
    for (img1_name, img2_name), (kp1, kp2, matches) in feature_matches.items():
        if img1_name not in camera_poses or img2_name not in camera_poses:
            continue
        
        # Get camera poses
        R1, t1 = camera_poses[img1_name]
        R2, t2 = camera_poses[img2_name]
        
        # Create projection matrices
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P2 = K @ np.hstack((R2, t2.reshape(3, 1)))
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Triangulate points
        points_3d_pair = triangulate_points(P1, P2, pts1, pts2)
        
        # Compute reprojection error for first camera
        errors1 = compute_reprojection_error(
            points_3d_pair, pts1, K, cv2.Rodrigues(R1)[0], t1.reshape(3, 1))
        
        # Compute reprojection error for second camera
        errors2 = compute_reprojection_error(
            points_3d_pair, pts2, K, cv2.Rodrigues(R2)[0], t2.reshape(3, 1))
        
        # Average reprojection error
        avg_errors = (errors1 + errors2) / 2
        
        # Compute camera centers
        camera1_center = -R1.T @ t1
        camera2_center = -R2.T @ t2
        
        # Add points and their observations
        for i, (pt3d, error) in enumerate(zip(points_3d_pair, avg_errors)):
            # Skip points with high reprojection error
            if error > 4.0:
                continue
            
            # Compute triangulation angle
            angle = compute_triangulation_angle(pt3d, camera1_center, camera2_center)
            
            # Skip points with small triangulation angle (potentially unstable)
            if angle < 3.0 or angle > 175.0:
                continue
            
            # Add the 3D point
            points_3d.append(pt3d)
            
            # Add observations (which images see this point)
            observations = {
                img1_name: (kp1[matches[i].queryIdx].pt, matches[i].queryIdx),
                img2_name: (kp2[matches[i].trainIdx].pt, matches[i].trainIdx)
            }
            point_observations.append(observations)
    
    print(f"Triangulated {len(points_3d)} 3D points")
    return points_3d, point_observations

def merge_triangulated_points(points_3d, point_observations, threshold=1e-3):
    """
    Merge 3D points that are very close to each other.
    
    Args:
        points_3d: List of 3D points.
        point_observations: List of image observations for each point.
        threshold: Distance threshold for merging points.
        
    Returns:
        Merged points and observations.
    """
    if len(points_3d) == 0:
        return [], []
    
    # Convert to numpy array for faster processing
    points_array = np.array(points_3d)
    
    # Initialize merged points and observations
    merged_points = []
    merged_observations = []
    
    # Set of indices that have already been merged
    processed_indices = set()
    
    for i in range(len(points_array)):
        if i in processed_indices:
            continue
        
        # Get current point
        current_point = points_array[i]
        current_obs = point_observations[i]
        
        # Find close points
        distances = np.sqrt(np.sum((points_array - current_point) ** 2, axis=1))
        close_indices = np.where(distances < threshold)[0]
        
        # Skip if only the current point is found
        if len(close_indices) == 1 and close_indices[0] == i:
            merged_points.append(current_point)
            merged_observations.append(current_obs)
            processed_indices.add(i)
            continue
        
        # Merge close points
        merged_point = np.mean(points_array[close_indices], axis=0)
        
        # Merge observations
        merged_obs = {}
        for idx in close_indices:
            obs = point_observations[idx]
            for img_name, (pt, kp_idx) in obs.items():
                if img_name in merged_obs:
                    # If we already have an observation in this image, keep the one with smaller kp_idx
                    if kp_idx < merged_obs[img_name][1]:
                        merged_obs[img_name] = (pt, kp_idx)
                else:
                    merged_obs[img_name] = (pt, kp_idx)
        
        # Add merged point and observations
        merged_points.append(merged_point)
        merged_observations.append(merged_obs)
        
        # Mark all close points as processed
        processed_indices.update(close_indices)
    
    print(f"Merged {len(points_3d)} points into {len(merged_points)} points")
    return merged_points, merged_observations