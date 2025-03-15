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
        iterationsCount=5000,
        reprojectionError=8.0,
        confidence=0.99
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
    Estimate camera poses incrementally starting from an initial pair.
    
    Args:
        matches_dict: Dictionary mapping image pairs to matches.
        K: Camera intrinsic matrix.
        min_matches: Minimum number of matches required.
        
    Returns:
        Dictionary of camera poses {image_name: (R, t)}.
    """
    attempt_counter = {}  # Track failed attempts per image
    max_attempts = 5     # Maximum attempts before giving up
    
    # Sort image pairs by number of matches (descending)
    sorted_pairs = sorted(matches_dict.keys(), 
                          key=lambda pair: len(matches_dict[pair][2]), 
                          reverse=True)
    
    # Check if we have any pairs with enough matches
    if not sorted_pairs or len(matches_dict[sorted_pairs[0]][2]) < min_matches:
        print("Error: Not enough matches for pose estimation.")
        return {}
    
    # Start with the pair that has the most matches
    init_pair = sorted_pairs[0]
    img1_name, img2_name = init_pair
    kp1, kp2, matches = matches_dict[init_pair]
    
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
    observation_map = {}  # Maps 3D point index to image and keypoint indices
    
    # Triangulate initial points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R_rel, t_rel.reshape(3, 1)))
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # Store triangulated points and their observations
    for i, (m, pt3d) in enumerate(zip(matches, points_3d)):
        triangulated_points[i] = pt3d
        
        # Record observations
        if i not in observation_map:
            observation_map[i] = []
        
        observation_map[i].append((img1_name, m.queryIdx))
        observation_map[i].append((img2_name, m.trainIdx))
    
    # Process remaining images
    processed_images = set([img1_name, img2_name])
    
    while True:
        # Find image pairs where one image is processed and one is not
        next_pair = None
        max_matches = 0
        
        for pair in sorted_pairs:
            img_a, img_b = pair
            
            if img_a in processed_images and img_b not in processed_images:
                if len(matches_dict[pair][2]) > max_matches:
                    next_pair = pair
                    max_matches = len(matches_dict[pair][2])
            
            elif img_b in processed_images and img_a not in processed_images:
                if len(matches_dict[pair][2]) > max_matches:
                    next_pair = (img_b, img_a)  # Swap to ensure processed image is first
                    max_matches = len(matches_dict[pair][2])
        
        if next_pair is None or max_matches < min_matches:
            break  # No more images to process
        
        # Extract information
        known_img, new_img = next_pair
        
        # Track attempts for this image
        if new_img not in attempt_counter:
            attempt_counter[new_img] = 1
        else:
            attempt_counter[new_img] += 1
            # Check if we've tried too many times with this image
            if attempt_counter[new_img] > max_attempts:
                print(f"Maximum attempts reached for {new_img}. Permanently skipping.")
                processed_images.add(new_img)  # Mark as processed to avoid further attempts
                continue
        
        if (known_img, new_img) in matches_dict:
            kp_known, kp_new, matches = matches_dict[(known_img, new_img)]
        else:
            kp_new, kp_known, matches_rev = matches_dict[(new_img, known_img)]
            # Convert matches from new->known to known->new
            matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in matches_rev]
        
        # Find correspondences between new image and already triangulated points
        points_3d = []
        points_2d = []
        
        for i, m in enumerate(matches):
            # Find if this keypoint in the known image corresponds to a triangulated point
            for pt_idx, observations in observation_map.items():
                for obs_img, obs_kp_idx in observations:
                    if obs_img == known_img and obs_kp_idx == m.queryIdx:
                        # Found a correspondence
                        points_3d.append(triangulated_points[pt_idx])
                        points_2d.append(kp_new[m.trainIdx].pt)
                        break
        
        if len(points_3d) < 6:
            print(f"Warning: Not enough correspondences for {new_img}. Skipping.")
            continue
        
        # Estimate pose using PnP
        R_new, t_new, inliers = pnp_absolute_pose(points_3d, points_2d, K)
        
        if R_new is None or t_new is None:
            print(f"Warning: Failed to estimate pose for {new_img}. Skipping.")
            continue
        
        # Add new camera pose
        camera_poses[new_img] = (R_new, t_new)
        processed_images.add(new_img)
        
        print(f"Added pose for {new_img} using {len(inliers)} / {len(points_3d)} points")
        
        # Find new matches to triangulate
        for pair in sorted_pairs:
            img_a, img_b = pair
            
            if img_a in processed_images and img_b in processed_images:
                kp_a, kp_b, matches_ab = matches_dict[pair]
                
                # Add this check right here:
                if img_a not in camera_poses or img_b not in camera_poses:
                    continue  # Skip if either image doesn't have a camera pose
                    
                # Get camera poses
                R_a, t_a = camera_poses[img_a]
                R_b, t_b = camera_poses[img_b]
                
                # Create projection matrices
                P_a = K @ np.hstack((R_a, t_a.reshape(3, 1)))
                P_b = K @ np.hstack((R_b, t_b.reshape(3, 1)))
                
                # Extract matched points
                pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches_ab])
                pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches_ab])
                
                # Triangulate new points
                points_4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
                new_points_3d = (points_4d[:3] / points_4d[3]).T
                
                # Add new triangulated points
                start_idx = len(triangulated_points)
                
                for i, (m, pt3d) in enumerate(zip(matches_ab, new_points_3d)):
                    # Skip if this match already has a triangulated point
                    already_triangulated = False
                    
                    for observations in observation_map.values():
                        for obs_img, obs_kp_idx in observations:
                            if (obs_img == img_a and obs_kp_idx == m.queryIdx) or \
                               (obs_img == img_b and obs_kp_idx == m.trainIdx):
                                already_triangulated = True
                                break
                        if already_triangulated:
                            break
                    
                    if already_triangulated:
                        continue
                    
                    # Add new triangulated point
                    pt_idx = start_idx + i
                    triangulated_points[pt_idx] = pt3d
                    
                    # Record observations
                    if pt_idx not in observation_map:
                        observation_map[pt_idx] = []
                    
                    observation_map[pt_idx].append((img_a, m.queryIdx))
                    observation_map[pt_idx].append((img_b, m.trainIdx))
    
    print(f"Estimated poses for {len(camera_poses)} cameras")
    return camera_poses