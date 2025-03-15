import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2

def rotate_points(points, rot_vecs):
    """
    Rotate points by given rotation vectors.
    
    Args:
        points: Nx3 array of points.
        rot_vecs: Array of rotation vectors.
        
    Returns:
        Rotated points.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project_points(points, camera_params, K):
    """
    Project 3D points to 2D using camera parameters.
    
    Args:
        points: Nx3 array of 3D points.
        camera_params: Camera parameters (rotation vector and translation).
        K: Camera intrinsic matrix.
        
    Returns:
        Projected 2D points.
    """
    n_cameras = camera_params.shape[0] // 6
    n_points = points.shape[0]
    
    points_proj = np.zeros((n_cameras, n_points, 2))
    
    for i in range(n_cameras):
        rot_vecs = camera_params[i*6:i*6+3].reshape(1, 3)
        trans_vecs = camera_params[i*6+3:i*6+6].reshape(1, 3)
        
        # Rotate points
        points_rot = rotate_points(points, rot_vecs)
        
        # Translate points
        points_trans = points_rot + trans_vecs
        
        # Project to 2D
        x = points_trans[:, 0] / points_trans[:, 2]
        y = points_trans[:, 1] / points_trans[:, 2]
        
        # Apply camera intrinsics
        points_proj[i, :, 0] = K[0, 0] * x + K[0, 2]
        points_proj[i, :, 1] = K[1, 1] * y + K[1, 2]
    
    return points_proj

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """
    Compute reprojection error.
    
    Args:
        params: Camera parameters and 3D points.
        n_cameras: Number of cameras.
        n_points: Number of 3D points.
        camera_indices: Camera indices for each observation.
        point_indices: Point indices for each observation.
        points_2d: Observed 2D points.
        K: Camera intrinsic matrix.
        
    Returns:
        Reprojection error.
    """
    # Extract camera parameters and 3D points
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    # Project 3D points to 2D
    points_proj = project_points(points_3d, camera_params, K)
    
    # Calculate error
    points_proj = points_proj[camera_indices, point_indices]
    error = (points_proj - points_2d).ravel()
    
    return error

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Create Jacobian sparsity pattern for bundle adjustment.
    
    Args:
        n_cameras: Number of cameras.
        n_points: Number of 3D points.
        camera_indices: Camera indices for each observation.
        point_indices: Point indices for each observation.
        
    Returns:
        Sparse Jacobian matrix.
    """
    m = camera_indices.size * 2  # Number of residuals
    n = n_cameras * 6 + n_points * 3  # Number of parameters
    
    A = lil_matrix((m, n), dtype=int)
    
    # Fill in the Jacobian matrix
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    
    return A

def perform_bundle_adjustment(camera_poses, points_3d, point_observations, K, iterations=20):
    """
    Perform bundle adjustment to refine camera poses and 3D points.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        points_3d: List of 3D points.
        point_observations: List of point observations.
        K: Camera intrinsic matrix.
        iterations: Maximum number of iterations.
        
    Returns:
        Refined camera poses and 3D points.
    """
    # Skip if no points
    if len(points_3d) == 0:
        return camera_poses, points_3d
    
    # Prepare camera parameters
    camera_names = list(camera_poses.keys())
    n_cameras = len(camera_names)
    camera_params = np.zeros((n_cameras, 6))
    
    for i, name in enumerate(camera_names):
        R, t = camera_poses[name]
        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(R)
        camera_params[i, :3] = rvec.ravel()
        camera_params[i, 3:] = t.ravel()
    
    # Prepare point parameters
    n_points = len(points_3d)
    point_params = np.array(points_3d)
    
    # Prepare observations
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for i, obs in enumerate(point_observations):
        for img_name, (pt, _) in obs.items():
            if img_name in camera_names:
                camera_idx = camera_names.index(img_name)
                camera_indices.append(camera_idx)
                point_indices.append(i)
                points_2d.append(pt)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    # Combine all parameters
    params = np.hstack((camera_params.ravel(), point_params.ravel()))
    
    # Create sparsity pattern
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    # Define optimization problem
    def fun(params):
        return reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K)
    
    # Run optimization
    result = least_squares(
        fun,
        params,
        jac_sparsity=A,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        max_nfev=iterations
    )
    
    # Extract optimized parameters
    camera_params_opt = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    point_params_opt = result.x[n_cameras * 6:].reshape((n_points, 3))
    
    # Update camera poses
    camera_poses_opt = {}
    for i, name in enumerate(camera_names):
        rvec = camera_params_opt[i, :3]
        tvec = camera_params_opt[i, 3:]
        R, _ = cv2.Rodrigues(rvec)
        camera_poses_opt[name] = (R, tvec)
    
    # Compute final reprojection error
    final_error = np.sqrt(np.mean(result.fun**2))
    print(f"Bundle adjustment complete. Final reprojection error: {final_error:.4f} pixels")
    
    return camera_poses_opt, point_params_opt.tolist()

def prepare_bundle_adjustment_data(camera_poses, feature_matches, K, min_observations=2):
    """
    Prepare data for bundle adjustment.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        feature_matches: Dictionary of feature matches {(img1, img2): (kp1, kp2, matches)}.
        K: Camera intrinsic matrix.
        min_observations: Minimum number of observations required for each point.
        
    Returns:
        Tuple of (points_3d, point_observations) for bundle adjustment.
    """
    # Triangulate all points
    points_3d, point_observations = triangulate_all_points(camera_poses, feature_matches, K)
    
    # Filter points with too few observations
    valid_indices = []
    for i, obs in enumerate(point_observations):
        if len(obs) >= min_observations:
            valid_indices.append(i)
    
    filtered_points = [points_3d[i] for i in valid_indices]
    filtered_observations = [point_observations[i] for i in valid_indices]
    
    # Merge points that are close to each other
    merged_points, merged_observations = merge_triangulated_points(filtered_points, filtered_observations)
    
    print(f"Prepared {len(merged_points)} points with at least {min_observations} observations for bundle adjustment")
    
    return merged_points, merged_observations

def run_global_ba(camera_poses, feature_matches, K):
    """
    Run global bundle adjustment.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        feature_matches: Dictionary of feature matches {(img1, img2): (kp1, kp2, matches)}.
        K: Camera intrinsic matrix.
        
    Returns:
        Refined camera poses and 3D points.
    """
    # Prepare data
    points_3d, point_observations = prepare_bundle_adjustment_data(camera_poses, feature_matches, K)
    
    # Run bundle adjustment
    refined_poses, refined_points = perform_bundle_adjustment(camera_poses, points_3d, point_observations, K)
    
    return refined_poses, refined_points, point_observations