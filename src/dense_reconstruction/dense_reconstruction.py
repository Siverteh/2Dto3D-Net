    # Dense Point Cloud Generation from Depth Maps
print("\nGenerating dense point cloud from depth maps...")

# Import necessary libraries
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def backproject_depth_map(depth_map, image, K, R, t, min_depth=0.1, max_depth=10.0):
    """
    Backproject depth map to 3D points.
    
    Args:
        depth_map: HxW depth map
        image: HxWxC color image
        K: 3x3 camera intrinsic matrix
        R: 3x3 camera rotation matrix
        t: 3x1 camera translation vector
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        
    Returns:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of colors
    """
    h, w = depth_map.shape
    
    # Create mask of valid depths
    mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    if not np.any(mask):
        print("No valid depths to backproject!")
        return np.empty((0, 3)), np.empty((0, 3))
    
    # Get pixel coordinates of valid depths
    y_coords, x_coords = np.where(mask)
    
    # Get corresponding depths
    depths = depth_map[mask]
    
    # Get camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Backproject to camera space
    x_cam = (x_coords - cx) * depths / fx
    y_cam = (y_coords - cy) * depths / fy
    z_cam = depths
    
    # Combine into camera space points
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Transform to world space
    # R is camera to world rotation, so we need its transpose
    points_world = (R.T @ (points_cam.T - t.reshape(3, 1))).T
    
    # Get colors
    if len(image.shape) == 3:
        colors = image[y_coords, x_coords] / 255.0
    else:
        # For grayscale images
        colors_gray = image[y_coords, x_coords] / 255.0
        colors = np.column_stack([colors_gray, colors_gray, colors_gray])
    
    return points_world, colors

def create_foreground_mask(image, threshold=15, min_area=1000, edge_strength=150, selective_cutting=False):
    """
    Create a foreground mask with balanced edge refinement.
    
    Args:
        image: Input image
        threshold: Threshold for foreground segmentation
        min_area: Minimum area of components to keep
        edge_strength: Threshold for edge detection (higher = fewer edges)
        selective_cutting: If True, only cut in specified regions (default: False)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to separate foreground from background
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Close small holes
    kernel_close = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    
    # Create mask with only large components
    refined_mask = np.zeros_like(closed)
    
    # Skip label 0 (background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            refined_mask[labels == i] = 255
    
    # Find edges with higher threshold for stronger edges only
    edges = cv2.Canny(gray, edge_strength, edge_strength * 1.5)
    
    # Use smaller kernel for less aggressive dilation
    kernel = np.ones((1, 1), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    
    # Apply edge cutting based on selective_cutting parameter
    if selective_cutting:
        # You can modify this part to create custom region masks
        # that don't specifically target the legs
        h, w = refined_mask.shape
        
        # Create a general mask for areas where edge detection should be applied
        # This example uses the top half of the image only
        region_mask = np.zeros_like(refined_mask, dtype=bool)
        region_mask[:int(h*0.5), :] = True  # Apply only to top half, avoid legs
        
        # Only apply edge cutting in the region mask
        cut_mask = np.zeros_like(refined_mask, dtype=bool)
        cut_mask[region_mask] = dilated_edges[region_mask] > 0
        
        # Cut the mask only at selected edge locations
        refined_mask[cut_mask] = 0
    else:
        # Apply edge cutting globally but only for strong edges
        # This is generally safer for preserving the full object
        refined_mask[dilated_edges > 0] = 0
    
    return refined_mask > 0

def project_points_to_image(points_3d, K, R, t, img_shape):
    """
    Project 3D points to image coordinates.
    """
    # Convert points to camera coordinates
    points_cam = R @ points_3d.T + t.reshape(3, 1)
    
    # Get depths (Z coordinates in camera frame)
    depths = points_cam[2, :]
    
    # Skip points behind the camera
    mask = depths > 0
    
    # Project to image coordinates
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[mask, 0] = K[0, 0] * points_cam[0, mask] / depths[mask] + K[0, 2]
    points_2d[mask, 1] = K[1, 1] * points_cam[1, mask] / depths[mask] + K[1, 2]
    
    # Check which points are in image bounds
    h, w = img_shape[:2]
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    mask = mask & in_bounds
    
    return points_2d, depths, mask

def propagate_foreground_depth(sparse_depth, sparse_mask, image, max_distance=30):
    """
    Propagate depth only within foreground regions.
    """
    h, w = sparse_depth.shape
    
    # Find valid sparse points
    y_coords, x_coords = np.where(sparse_mask)
    
    if len(y_coords) == 0:
        print("No valid sparse points found!")
        return np.zeros_like(sparse_depth), np.zeros_like(sparse_depth)
    
    # Get valid depths and coordinates
    valid_coords = np.column_stack([x_coords, y_coords])
    valid_depths = sparse_depth[y_coords, x_coords]
    
    print(f"Propagating from {len(valid_coords)} sparse points...")
    
    # Create foreground mask
    foreground_mask = create_foreground_mask(image)
    
    # Find foreground pixels
    fg_y, fg_x = np.where(foreground_mask)
    fg_coords = np.column_stack([fg_x, fg_y])
    
    if len(fg_coords) == 0:
        print("No foreground pixels detected!")
        return np.zeros_like(sparse_depth), np.zeros_like(sparse_depth)
    
    print(f"Propagating depths to {len(fg_coords)} foreground pixels...")
    
    # Use KD-tree for efficient distance computation
    tree = cKDTree(valid_coords)
    
    # Query for all foreground pixels at once
    dists, indices = tree.query(fg_coords, k=5)  # Use 5 nearest neighbors for better quality
    
    # Initialize dense depth and confidence maps
    dense_depth = np.zeros((h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    
    # Compute weighted average at foreground pixels
    for i in range(len(fg_coords)):
        x, y = fg_coords[i]
        dist = dists[i]
        idx = indices[i]
        
        # Skip if too far from any sparse point
        if dist[0] > max_distance:
            continue
        
        # Compute weights - steeper falloff for better detail
        weights = np.exp(-dist / (max_distance/5))
        total_weight = np.sum(weights)
        
        if total_weight > 1e-10:
            # Compute weighted depth
            weighted_depth = np.sum(valid_depths[idx] * weights) / total_weight
            
            # Assign to dense maps
            dense_depth[y, x] = weighted_depth
            confidence[y, x] = np.exp(-dist[0] / (max_distance/2))
    
    return dense_depth, confidence

def generate_depth_map(points_3d, colors_3d, image, K, R, t):
    """
    Generate a depth map for a single view using sparse points and foreground constraint.
    """
    # Project sparse points to image
    sparse_depth = np.zeros(image.shape[:2], dtype=np.float32)
    sparse_mask = np.zeros(image.shape[:2], dtype=bool)
    
    points_2d, depths, mask = project_points_to_image(points_3d, K, R, t, image.shape)
    
    # Fill sparse depth map
    h, w = image.shape[:2]
    valid_points_2d = points_2d[mask].astype(int)
    valid_depths = depths[mask]
    
    for i in range(len(valid_points_2d)):
        x, y = valid_points_2d[i]
        if 0 <= x < w and 0 <= y < h:
            sparse_depth[y, x] = valid_depths[i]
            sparse_mask[y, x] = True
    
    # Count valid sparse pixels
    valid_count = np.sum(sparse_mask)
    
    if valid_count < 10:
        print("Too few valid sparse points!")
        return None
    
    # Propagate depth to foreground
    dense_depth, _ = propagate_foreground_depth(sparse_depth, sparse_mask, image)
    
    return dense_depth

def generate_dense_point_cloud(depth_maps, images, camera_poses, K, output_dir):
    """
    Generate dense point cloud from depth maps.
    
    Args:
        depth_maps: Dictionary mapping image names to depth maps
        images: List of (image, name) tuples
        camera_poses: Dictionary mapping image names to (R, t) tuples
        K: Camera intrinsic matrix
        output_dir: Output directory
    """
    # Create output directory
    pc_dir = os.path.join(output_dir, "dense_point_cloud")
    os.makedirs(pc_dir, exist_ok=True)
    
    all_points = []
    all_colors = []
    
    # Process each depth map
    for image_name, depth_map in tqdm(depth_maps.items(), desc="Backprojecting depth maps"):
        # Get corresponding image and camera pose
        image = next(img for img, name in images if name == image_name)
        R, t = camera_poses[image_name]
        
        # Skip if no depth map
        if depth_map is None or not np.any(depth_map > 0):
            print(f"Skipping {image_name} - no valid depth")
            continue
        
        # Backproject depth map to 3D points
        points, colors = backproject_depth_map(depth_map, image, K, R, t)
        
        # Skip if no points
        if len(points) == 0:
            print(f"Skipping {image_name} - no valid points")
            continue
        
        print(f"Generated {len(points)} points from {image_name}")
        
        # Add to combined point cloud
        all_points.append(points)
        all_colors.append(colors)
    
    # Combine all points
    if not all_points:
        print("No valid points generated!")
        return None
    
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    print(f"Combined point cloud has {len(combined_points)} points")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save raw point cloud
    raw_pc_file = os.path.join(pc_dir, "dense_raw.ply")
    o3d.io.write_point_cloud(raw_pc_file, pcd)
    print(f"Saved raw dense point cloud to {raw_pc_file}")
    
    # Downsample point cloud
    print("Downsampling point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
    print(f"Downsampled to {len(pcd_down.points)} points")
    
    # Remove outliers
    print("Removing outliers...")
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"After outlier removal: {len(pcd_clean.points)} points")
    
    # Save processed point cloud
    clean_pc_file = os.path.join(pc_dir, "dense_clean.ply")
    o3d.io.write_point_cloud(clean_pc_file, pcd_clean)
    print(f"Saved clean dense point cloud to {clean_pc_file}")
    
    return pcd_clean

def create_surface_mesh(pcd, output_dir):
    """
    Create surface mesh from point cloud.
    """
    # Create output directory
    mesh_dir = os.path.join(output_dir, "dense_mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Estimate normals if not already present
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)
    
    # Create mesh using Poisson reconstruction
    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    
    # Remove low-density vertices
    if len(densities) > 0:
        print("Removing low-density vertices...")
        density_threshold = np.quantile(np.asarray(densities), 0.05)  # More lenient threshold
        vertices_to_remove = np.asarray(densities) < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean up mesh
    print("Cleaning mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    # Save mesh
    mesh_file = os.path.join(mesh_dir, "dense_poisson.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    print(f"Saved Poisson mesh to {mesh_file}")
    
    # Create ball pivoting mesh
    print("Performing Ball Pivoting reconstruction...")
    
    # Estimate radius for ball pivoting
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2.0 * avg_dist
    
    radii = [radius, radius * 2, radius * 4]
    bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # Clean up mesh
    bp_mesh.compute_vertex_normals()
    
    # Save ball pivoting mesh
    bp_mesh_file = os.path.join(mesh_dir, "dense_ballpivot.ply")
    o3d.io.write_triangle_mesh(bp_mesh_file, bp_mesh)
    print(f"Saved Ball Pivoting mesh to {bp_mesh_file}")
    
    return mesh, bp_mesh

def generate_depth_maps_all_views(points_3d, colors_3d, images, camera_poses, K):
    """
    Generate depth maps for all views, either from cache or by computation.
    """
    depth_maps = {}
    
    for idx, (image_name, pose) in enumerate(tqdm(camera_poses.items(), desc="Generating depth maps")):
        print(f"\nProcessing view {idx+1}/{len(camera_poses)}: {image_name}")
        
        # Get image
        image = next(img for img, name in images if name == image_name)
        
        # Get camera pose
        R, t = pose
        
        # Generate depth map
        depth_map = generate_depth_map(points_3d, colors_3d, image, K, R, t)
        
        if depth_map is not None:
            depth_maps[image_name] = depth_map
            print(f"Generated depth map for {image_name}")
        else:
            print(f"Failed to generate depth map for {image_name}")
    
    return depth_maps

# Main execution
def create_dense_reconstruction(points_array, colors, images_black, camera_poses, K, output_dir):
    """
    Create dense reconstruction from sparse points.
    """
    print("\nCreating dense reconstruction from sparse points...")
    
    # Step 1: Generate depth maps for all views
    print("\nStep 1: Generating depth maps...")
    depth_maps = generate_depth_maps_all_views(points_array, colors, images_black, camera_poses, K)
    
    # Step 2: Create dense point cloud from depth maps
    print("\nStep 2: Creating dense point cloud...")
    dense_pcd = generate_dense_point_cloud(depth_maps, images_black, camera_poses, K, output_dir)
    
    # Step 3: Create surface mesh from point cloud
    return {
        'depth_maps': depth_maps,
        'point_cloud': dense_pcd
    }