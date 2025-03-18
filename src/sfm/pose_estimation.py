import cv2
import numpy as np

def decompose_essential_matrix(E):
    """
    Decompose essential matrix into rotation and translation.
    
    Args:
        E: Essential matrix.
        
    Returns:
        List of four possible (R, t) combinations.
    """
    U, _, Vt = np.linalg.svd(E)
    
    # Ensure rotation matrix with positive determinant
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    # Create W matrix for decomposition
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Calculate rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Calculate translation vectors
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # Four possible combinations
    return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

def check_points_in_front(R, t, pts1, pts2, K):
    """
    Count points that lie in front of both cameras.
    
    Args:
        R, t: Camera rotation and translation.
        pts1, pts2: Matched image points.
        K: Camera intrinsic matrix.
        
    Returns:
        Number of points in front of both cameras.
    """
    # Create projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    
    # Transform points to second camera frame
    points_3d_cam2 = R @ points_3d + t.reshape(3, 1)
    
    # Count points in front of both cameras (positive Z)
    points_in_front = np.sum((points_3d[2] > 0) & (points_3d_cam2[2] > 0))
    
    return points_in_front

def estimate_pose_from_essential(E, K, pts1, pts2):
    """
    Estimate camera pose from essential matrix.
    
    Args:
        E: Essential matrix.
        K: Camera intrinsic matrix.
        pts1, pts2: Matched points.
        
    Returns:
        Best (R, t) combination.
    """
    # Decompose essential matrix
    poses = decompose_essential_matrix(E)
    
    # Find the pose with most points in front of both cameras
    best_pose = None
    max_points_in_front = -1
    
    for R, t in poses:
        # Check if points are in front of both cameras
        points_in_front = check_points_in_front(R, t, pts1, pts2, K)
        
        if points_in_front > max_points_in_front:
            max_points_in_front = points_in_front
            best_pose = (R, t)
    
    return best_pose

def estimate_relative_pose(kp1, kp2, matches, K):
    """
    Estimate relative camera pose between two images.
    
    Args:
        kp1, kp2: Keypoints from two images.
        matches: List of matches.
        K: Camera intrinsic matrix.
        
    Returns:
        R, t: Camera rotation and translation.
    """
    if len(matches) < 8:
        return None, None
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Calculate essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None or E.shape != (3, 3):
        return None, None
    
    # Select inliers
    inliers_mask = mask.ravel() == 1
    pts1_inliers = pts1[inliers_mask]
    pts2_inliers = pts2[inliers_mask]
    
    # Recover pose from essential matrix
    R, t = estimate_pose_from_essential(E, K, pts1_inliers, pts2_inliers)
    
    return R, t

def pnp_absolute_pose(points_3d, points_2d, K, method=8):
    """
    Estimate absolute camera pose using PnP.
    
    Args:
        points_3d: 3D points in world coordinates.
        points_2d: Corresponding 2D points in image coordinates.
        K: Camera intrinsic matrix.
        method: PnP method to use.
        
    Returns:
        R, t: Camera rotation and translation.
    """
    if len(points_3d) < 6:
        print(f"Warning: Only {len(points_3d)} points for PnP. At least 6 recommended.")
    
    # Convert points to correct format
    points_3d = np.array(points_3d).astype(np.float32)
    points_2d = np.array(points_2d).astype(np.float32)
    
    # Initial camera pose estimate
    dist_coeffs = np.zeros(4)  # Assuming no lens distortion
    
    # Solve PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, dist_coeffs, 
        iterationsCount=10000,
        reprojectionError=0.65,
        confidence=0.99999
    )
    
    if not success:
        return None, None, []
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    
    # Calculate reprojection error for inliers
    if inliers is not None and len(inliers) > 0:
        inlier_indices = inliers.ravel()
        inlier_points_3d = points_3d[inlier_indices]
        inlier_points_2d = points_2d[inlier_indices]
        
        # Project 3D points
        projected_points, _ = cv2.projectPoints(inlier_points_3d, rvec, tvec, K, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate error
        error = np.sqrt(np.sum((inlier_points_2d - projected_points) ** 2, axis=1))
        mean_error = np.mean(error)
        
        print(f"PnP successful with {len(inliers)} inliers. Mean reprojection error: {mean_error:.2f} pixels")
    else:
        print("Warning: PnP successful but no inliers found.")
    
    return R, t, inliers if inliers is not None else []

def estimate_poses_incremental(matches_dict, K, min_matches=20):
    """
    Estimate camera poses incrementally starting from consecutive frames.
    
    Args:
        matches_dict: Dictionary mapping image pairs to matches.
        K: Camera intrinsic matrix.
        min_matches: Minimum number of matches required.
        
    Returns:
        Dictionary of camera poses {image_name: (R, t)}.
    """
    # Sort image pairs by filename to ensure sequential processing
    all_filenames = set()
    for img1, img2 in matches_dict.keys():
        all_filenames.add(img1)
        all_filenames.add(img2)
    
    # Sort filenames to get sequential order
    sorted_filenames = sorted(list(all_filenames))
    
    # Find consecutive image pairs for initialization
    init_pair = None
    for i in range(len(sorted_filenames) - 1):
        pair = (sorted_filenames[i], sorted_filenames[i+1])
        if pair in matches_dict and len(matches_dict[pair][2]) >= min_matches:
            init_pair = pair
            break
            
    if init_pair is None:
        # Fall back to best matching pair if no good consecutive pairs
        sorted_pairs = sorted(matches_dict.keys(), 
                            key=lambda pair: len(matches_dict[pair][2]), 
                            reverse=True)
        init_pair = sorted_pairs[0]
    
    img1_name, img2_name = init_pair
    kp1, kp2, matches = matches_dict[init_pair]
    
    print(f"Initializing with pair: {img1_name} and {img2_name} ({len(matches)} matches)")
    
    # Estimate initial relative pose
    R_rel, t_rel = estimate_relative_pose(kp1, kp2, matches, K)
    
    if R_rel is None or t_rel is None:
        print("Error: Failed to estimate initial relative pose.")
        return {}
    
    # Initialize camera poses
    camera_poses = {
        img1_name: (np.eye(3), np.zeros(3)),       # First camera at origin
        img2_name: (R_rel, t_rel)                  # Second camera relative to first
    }
    
    # Track points that have been triangulated
    triangulated_points = {}
    point_index_counter = 0
    
    # Create a more efficient observation map
    # Maps (image_name, keypoint_idx) to 3D point index
    observations_by_view = {}
    
    # Initial triangulation
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R_rel, t_rel.reshape(3, 1)))
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # Store initial triangulated points
    for i, (m, pt3d) in enumerate(zip(matches, points_3d)):
        # Check triangulation quality
        pt_cam1 = pt3d  # Already in camera 1 frame
        pt_cam2 = R_rel @ pt3d + t_rel
        
        # Only keep points in front of both cameras
        if pt_cam1[2] > 0 and pt_cam2[2] > 0:
            triangulated_points[point_index_counter] = pt3d
            
            # Add observations
            observations_by_view[(img1_name, m.queryIdx)] = point_index_counter
            observations_by_view[(img2_name, m.trainIdx)] = point_index_counter
            
            point_index_counter += 1
    
    print(f"Triangulated {len(triangulated_points)} initial points")
    
    # Process remaining images SEQUENTIALLY
    processed_images = set([img1_name, img2_name])
    
    # Attempt to add each image in sequence first
    for img_name in sorted_filenames:
        if img_name in processed_images:
            continue
            
        print(f"Attempting to add image: {img_name}")
        
        # Find all matches between this image and already processed images
        points_3d = []
        points_2d = []
        point_indices = []
        
        # Collect 2D-3D correspondences
        for pair in matches_dict.keys():
            img_a, img_b = pair
            
            # Only use pairs where one image is processed and the other is the current target
            if (img_a == img_name and img_b in processed_images) or (img_b == img_name and img_a in processed_images):
                kp_a, kp_b, pair_matches = matches_dict[pair]
                
                for match in pair_matches:
                    if img_a == img_name:
                        # Current image is img_a
                        view_kp_idx = (img_b, match.trainIdx)
                        query_kp = kp_a[match.queryIdx].pt
                    else:
                        # Current image is img_b
                        view_kp_idx = (img_a, match.queryIdx)
                        query_kp = kp_b[match.trainIdx].pt
                    
                    # Check if this keypoint corresponds to a triangulated point
                    if view_kp_idx in observations_by_view:
                        pt_idx = observations_by_view[view_kp_idx]
                        pt3d = triangulated_points[pt_idx]
                        
                        points_3d.append(pt3d)
                        points_2d.append(query_kp)
                        point_indices.append(pt_idx)
        
        print(f"Found {len(points_3d)} 2D-3D correspondences for {img_name}")
        
        # Allow as few as 4 correspondences (minimum for PnP)
        if len(points_3d) >= 4:
            # Estimate pose using PnP
            R, t, inliers = pnp_absolute_pose(points_3d, points_2d, K)
            
            if R is not None and t is not None and len(inliers) >= 4:
                camera_poses[img_name] = (R, t)
                processed_images.add(img_name)
                
                print(f"Added pose for {img_name} using {len(inliers)} / {len(points_3d)} points")
                
                # Triangulate new points with all processed cameras
                for processed_img in processed_images:
                    if processed_img == img_name:
                        continue
                        
                    pair = (img_name, processed_img) if (img_name, processed_img) in matches_dict else (processed_img, img_name)
                    
                    if pair in matches_dict:
                        kp_a, kp_b, pair_matches = matches_dict[pair]
                        
                        # Get camera poses
                        R_a, t_a = camera_poses[pair[0]]
                        R_b, t_b = camera_poses[pair[1]]
                        
                        # Create projection matrices
                        P_a = K @ np.hstack((R_a, t_a.reshape(3, 1)))
                        P_b = K @ np.hstack((R_b, t_b.reshape(3, 1)))
                        
                        # Extract matched points
                        pts_a = np.float32([kp_a[m.queryIdx].pt for m in pair_matches])
                        pts_b = np.float32([kp_b[m.trainIdx].pt for m in pair_matches])
                        
                        # Triangulate new points
                        points_4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
                        new_points_3d = (points_4d[:3] / points_4d[3]).T
                        
                        # Add new triangulated points
                        triangulated_count = 0
                        
                        for i, (match, pt3d) in enumerate(zip(pair_matches, new_points_3d)):
                            # Convert to both camera coordinates to check if in front
                            pt_in_cam_a = R_a.T @ (pt3d - t_a)
                            pt_in_cam_b = R_b.T @ (pt3d - t_b)
                            
                            if pt_in_cam_a[2] <= 0 or pt_in_cam_b[2] <= 0:
                                continue
                                
                            # Check if either observation already exists
                            view_a = (pair[0], match.queryIdx)
                            view_b = (pair[1], match.trainIdx)
                            
                            if view_a in observations_by_view or view_b in observations_by_view:
                                continue
                                
                            # Add new point
                            triangulated_points[point_index_counter] = pt3d
                            observations_by_view[view_a] = point_index_counter
                            observations_by_view[view_b] = point_index_counter
                            point_index_counter += 1
                            triangulated_count += 1
                            
                        print(f"  Triangulated {triangulated_count} new points with {pair[0]} and {pair[1]}")
            else:
                print(f"  PnP failed for {img_name}")
        else:
            print(f"  Not enough correspondences for {img_name}")
    
    # Final statistics
    print(f"Estimated poses for {len(camera_poses)}/{len(all_filenames)} cameras")
    print(f"Total triangulated points: {len(triangulated_points)}")
    return camera_poses

# Define improved loop closure function
def force_loop_closure(camera_poses):
    """
    Enhanced loop closure with robust distribution of error
    """
    # Get ordered filenames
    filenames = sorted(list(camera_poses.keys()), 
                      key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Calculate camera centers
    centers = []
    rotations = []
    for name in filenames:
        R, t = camera_poses[name]
        center = -R.T @ t
        centers.append(center)
        rotations.append(R)
    
    centers = np.array(centers)
    
    # Calculate drift
    drift = centers[-1] - centers[0]
    drift_magnitude = np.linalg.norm(drift)
    print(f"Loop closure drift: {drift_magnitude:.4f} units")
    
    # Create corrected centers
    corrected_centers = np.copy(centers)
    
    # Use a smoother sinusoidal correction function
    # This distributes the correction more evenly
    for i in range(1, len(centers)):
        # Use a sinusoidal function for smooth distribution
        ratio = i / len(centers)
        weight = 0.5 * (1 - np.cos(np.pi * ratio))
        
        # Apply correction
        correction = drift * weight
        corrected_centers[i] = centers[i] - correction
    
    # Calculate corrected poses
    corrected_poses = {}
    
    for i, name in enumerate(filenames):
        R = rotations[i]
        new_center = corrected_centers[i]
        new_t = -R @ new_center
        corrected_poses[name] = (R, new_t)
    
    print("Smooth sinusoidal loop closure correction applied")
    return corrected_poses