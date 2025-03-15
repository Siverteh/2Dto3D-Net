import numpy as np
import open3d as o3d
from tqdm import tqdm

def depth_map_to_point_cloud(depth_map, image, K, R, t, confidence_map=None, confidence_threshold=0.5):
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth_map: Depth map.
        image: RGB image.
        K: Camera intrinsic matrix.
        R: Camera rotation matrix.
        t: Camera translation vector.
        confidence_map: Optional confidence map.
        confidence_threshold: Threshold for confidence map.
        
    Returns:
        List of 3D points and colors.
    """
    h, w = depth_map.shape
    points = []
    colors = []
    
    # Get camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Process each pixel
    for y in range(h):
        for x in range(w):
            depth = depth_map[y, x]
            
            # Skip invalid depths
            if depth <= 0:
                continue
            
            # Skip low confidence points
            if confidence_map is not None and confidence_map[y, x] < confidence_threshold:
                continue
            
            # Compute 3D point in camera coordinates
            z = depth
            x_cam = (x - cx) * z / fx
            y_cam = (y - cy) * z / fy
            point_camera = np.array([x_cam, y_cam, z])
            
            # Transform to world coordinates
            point_world = R @ point_camera + t
            
            # Add point and color
            points.append(point_world)
            if len(image.shape) == 3:
                colors.append(image[y, x] / 255.0)  # Normalize to [0, 1]
            else:
                # For grayscale images
                gray = image[y, x] / 255.0
                colors.append(np.array([gray, gray, gray]))
    
    return np.array(points), np.array(colors)

def create_dense_point_cloud(depth_maps, images, camera_poses, K, confidence_maps=None, confidence_threshold=0.5):
    """
    Create dense point cloud from multiple depth maps.
    
    Args:
        depth_maps: Dictionary of depth maps.
        images: List of (image, filename) tuples.
        camera_poses: Dictionary mapping filenames to (R, t).
        K: Camera intrinsic matrix.
        confidence_maps: Optional dictionary of confidence maps.
        confidence_threshold: Threshold for confidence maps.
        
    Returns:
        Array of points and colors.
    """
    all_points = []
    all_colors = []
    
    # Process each depth map
    for i, (filename, depth_map) in enumerate(tqdm(depth_maps.items(), desc="Creating point cloud")):
        # Get image
        image = next(img for img, name in images if name == filename)
        
        # Get camera pose
        R, t = camera_poses[filename]
        
        # Get confidence map if available
        conf_map = confidence_maps.get(filename) if confidence_maps else None
        
        # Convert depth map to point cloud
        points, colors = depth_map_to_point_cloud(
            depth_map, image, K, R, t, conf_map, confidence_threshold)
        
        # Add points and colors
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
            print(f"Added {len(points)} points from {filename}")
    
    # Combine all points and colors
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        print(f"Created point cloud with {len(combined_points)} points")
        return combined_points, combined_colors
    else:
        print("No points were generated")
        return np.empty((0, 3)), np.empty((0, 3))

def filter_point_cloud(points, colors, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    """
    Filter point cloud to remove noise and outliers.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors.
        voxel_size: Voxel size for downsampling.
        nb_neighbors: Number of neighbors for outlier removal.
        std_ratio: Standard deviation ratio for outlier removal.
        
    Returns:
        Filtered points and colors.
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Original point cloud has {len(points)} points")
    
    # Voxel downsampling
    print(f"Downsampling with voxel size {voxel_size}...")
    downsampled = pcd.voxel_down_sample(voxel_size)
    
    # Statistical outlier removal
    print(f"Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    filtered, _ = downsampled.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # Convert back to numpy arrays
    filtered_points = np.asarray(filtered.points)
    filtered_colors = np.asarray(filtered.colors)
    
    print(f"Filtered point cloud has {len(filtered_points)} points")
    return filtered_points, filtered_colors

def compute_normals(points, colors, radius=0.1, max_nn=30):
    """
    Compute normals for point cloud.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors.
        radius: Radius for nearest neighbor search.
        max_nn: Maximum number of neighbors.
        
    Returns:
        Points, colors, and normals.
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    
    # Convert to numpy arrays
    normals = np.asarray(pcd.normals)
    
    print(f"Computed normals for {len(points)} points")
    return points, colors, normals

def create_surface_mesh(points, colors, normals, method='poisson', depth=9):
    """
    Create surface mesh from point cloud.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors.
        normals: Nx3 array of normals.
        method: 'poisson' or 'ball_pivoting'.
        depth: Depth parameter for Poisson reconstruction.
        
    Returns:
        Triangle mesh.
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Create mesh
    if method.lower() == 'poisson':
        print(f"Performing Poisson surface reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    else:
        # Estimate radius for ball pivoting
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        print(f"Performing ball pivoting reconstruction (radius={radius:.6f})...")
        radii = [radius, radius * 2, radius * 4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    
    # Clean up mesh
    mesh.compute_vertex_normals()
    
    print(f"Created mesh with {len(mesh.triangles)} triangles")
    return mesh

def save_point_cloud(points, colors, filename):
    """
    Save point cloud to file.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors.
        filename: Output filename (PLY format).
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def save_mesh(mesh, filename):
    """
    Save mesh to file.
    
    Args:
        mesh: Open3D triangle mesh.
        filename: Output filename.
    """
    # Save to file
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Saved mesh to {filename}")

def process_dense_reconstruction(images, camera_poses, K, depth_maps, confidence_maps, config):
    """
    Process dense reconstruction pipeline.
    
    Args:
        images: List of (image, filename) tuples.
        camera_poses: Dictionary mapping filenames to (R, t).
        K: Camera intrinsic matrix.
        depth_maps: Dictionary of depth maps.
        confidence_maps: Dictionary of confidence maps.
        config: Configuration parameters.
        
    Returns:
        Dictionary with point cloud, filtered point cloud, and mesh.
    """
    # Extract parameters from config
    voxel_size = config.get('voxel_size', 0.01)
    nb_neighbors = config.get('nb_neighbors', 20)
    std_ratio = config.get('std_ratio', 2.0)
    confidence_threshold = config.get('confidence_threshold', 0.5)
    
    # Create dense point cloud
    points, colors = create_dense_point_cloud(
        depth_maps, images, camera_poses, K, 
        confidence_maps, confidence_threshold)
    
    # Filter point cloud
    filtered_points, filtered_colors = filter_point_cloud(
        points, colors, voxel_size, nb_neighbors, std_ratio)
    
    # Compute normals
    _, _, normals = compute_normals(filtered_points, filtered_colors)
    
    # Create surface mesh if point cloud is not empty
    if len(filtered_points) > 0:
        mesh = create_surface_mesh(filtered_points, filtered_colors, normals)
    else:
        mesh = None
    
    return {
        'points': points,
        'colors': colors,
        'filtered_points': filtered_points,
        'filtered_colors': filtered_colors,
        'normals': normals,
        'mesh': mesh
    }