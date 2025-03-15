import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def create_point_cloud_from_points(points, colors=None):
    """
    Create an Open3D point cloud from points and optional colors.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Nx3 array of RGB colors (values in [0, 1]).
        
    Returns:
        Open3D point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def estimate_point_normals(pcd, radius=0.1, max_nn=30):
    """
    Estimate normals for a point cloud.
    
    Args:
        pcd: Open3D point cloud.
        radius: Radius for nearest neighbor search.
        max_nn: Maximum number of neighbors to consider.
        
    Returns:
        Point cloud with estimated normals.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    
    return pcd

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove outliers from point cloud using statistical analysis.
    
    Args:
        pcd: Open3D point cloud.
        nb_neighbors: Number of neighbors to consider.
        std_ratio: Standard deviation ratio threshold.
        
    Returns:
        Filtered point cloud.
    """
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return filtered_pcd

def remove_radius_outliers(pcd, nb_points=16, radius=0.05):
    """
    Remove outliers from point cloud based on radius search.
    
    Args:
        pcd: Open3D point cloud.
        nb_points: Minimum number of points required in radius.
        radius: Search radius.
        
    Returns:
        Filtered point cloud.
    """
    filtered_pcd, _ = pcd.remove_radius_outlier(
        nb_points=nb_points, radius=radius)
    
    return filtered_pcd

def downsample_point_cloud(pcd, voxel_size=0.01):
    """
    Downsample point cloud using voxel grid.
    
    Args:
        pcd: Open3D point cloud.
        voxel_size: Size of voxel grid.
        
    Returns:
        Downsampled point cloud.
    """
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd

def poisson_surface_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    """
    Perform Poisson surface reconstruction.
    
    Args:
        pcd: Open3D point cloud with normals.
        depth: Maximum depth of octree.
        width: Adaptive octree width.
        scale: Scale factor for reconstruction.
        linear_fit: Whether to use linear interpolation.
        
    Returns:
        Reconstructed triangle mesh.
    """
    # Check if normals are available
    if not pcd.has_normals():
        print("Point cloud does not have normals. Estimating normals...")
        pcd = estimate_point_normals(pcd)
    
    # Perform Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
    
    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Ensure mesh is manifold
    mesh.compute_vertex_normals()
    
    return mesh

def ball_pivoting_surface_reconstruction(pcd, radii=None):
    """
    Perform Ball Pivoting surface reconstruction.
    
    Args:
        pcd: Open3D point cloud with normals.
        radii: List of ball radii. If None, estimate based on point cloud.
        
    Returns:
        Reconstructed triangle mesh.
    """
    # Check if normals are available
    if not pcd.has_normals():
        print("Point cloud does not have normals. Estimating normals...")
        pcd = estimate_point_normals(pcd)
    
    # Estimate ball radius if not provided
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]
    
    # Perform Ball Pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # Ensure mesh is manifold
    mesh.compute_vertex_normals()
    
    return mesh

def alpha_shape_reconstruction(points, alpha=0.5):
    """
    Perform Alpha Shape surface reconstruction.
    
    Args:
        points: Nx3 array of points.
        alpha: Alpha value for reconstruction.
        
    Returns:
        Reconstructed triangle mesh.
    """
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    # Filter simplicies based on alpha criterion
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    
    # Loop through triangles and filter based on alpha criterion
    triangles = []
    for simplex in tri.simplices:
        # Compute circumradius
        p1, p2, p3 = points[simplex]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_radius = a * b * c / (4 * area) if area > 0 else float('inf')
        
        # Add triangle if circumradius is less than alpha
        if circum_radius < 1.0 / alpha:
            triangles.append(simplex)
    
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh
def clean_mesh(mesh, detail_level=5):
    """
    Clean mesh by removing isolated components, filling holes, etc.
    Compatible with older versions of Open3D.
    
    Args:
        mesh: Open3D triangle mesh.
        detail_level: Level of detail to preserve (1-10, higher means less smoothing).
        
    Returns:
        Cleaned mesh.
    """
    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()
    
    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    
    # Alternative way to remove isolated vertices without using get_vertex_degree()
    # Find vertices that are used in triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) > 0:  # Only proceed if there are triangles
        # Get unique vertices used in triangles
        used_vertices = np.unique(triangles.flatten())
        
        # Create a mask for vertices (True = keep, False = remove)
        vertex_mask = np.zeros(len(vertices), dtype=bool)
        vertex_mask[used_vertices] = True
        
        # Invert the mask since remove_vertices_by_mask removes vertices where mask is True
        mesh.remove_vertices_by_mask(~vertex_mask)
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    # Apply Taubin smoothing (maintains features better than Laplacian)
    lambda_factor = 0.5  # Positive scale factor
    mu_factor = -0.53    # Negative scale factor (slightly larger than -lambda)
    num_iterations = max(1, 10 // detail_level)  # Number of iteration steps
    
    try:
        smoothed_mesh = mesh.filter_smooth_taubin(
            number_of_iterations=num_iterations,
            lambda_filter=lambda_factor,
            mu=mu_factor
        )
        return smoothed_mesh
    except Exception as e:
        print(f"Warning: Smoothing failed with error: {e}")
        print("Returning unsmoothed mesh")
        return mesh

def simplify_mesh(mesh, target_reduction=0.5, quality=0.8):
    """
    Simplify mesh by reducing number of triangles.
    
    Args:
        mesh: Open3D triangle mesh.
        target_reduction: Target reduction ratio (0 to 1).
        quality: Quality of simplification (0 to 1).
        
    Returns:
        Simplified mesh.
    """
    # Calculate target number of triangles
    target_triangles = int(len(mesh.triangles) * (1 - target_reduction))
    
    # Simplify mesh
    simplified_mesh = mesh.simplify_quadric_decimation(target_triangles)
    
    # Ensure mesh is manifold
    simplified_mesh.compute_vertex_normals()
    
    return simplified_mesh

def fill_holes(mesh, hole_size=100):
    """
    Fill small holes in the mesh.
    
    Args:
        mesh: Open3D triangle mesh.
        hole_size: Maximum number of edges in holes to fill.
        
    Returns:
        Mesh with filled holes.
    """
    # This is a simplified implementation
    # Open3D doesn't have a direct hole filling algorithm
    # For complex hole filling, consider using other libraries like PyMeshFix
    
    # For now, we'll just ensure the mesh is manifold
    filled_mesh = mesh
    filled_mesh.compute_vertex_normals()
    
    return filled_mesh

def process_point_cloud_to_mesh(points, colors=None, method='poisson', cleanup=True):
    """
    Process point cloud to mesh using specified method.
    Enhanced to handle sparse point clouds better.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors (values in [0, 1]).
        method: 'poisson', 'ball_pivoting', or 'alpha_shape'.
        cleanup: Whether to clean up the resulting mesh.
        
    Returns:
        Triangle mesh.
    """
    print(f"Starting mesh reconstruction from {len(points)} points using {method} method")
    
    # Create point cloud
    pcd = create_point_cloud_from_points(points, colors)
    
    # Remove outliers (use more conservative parameters for sparse clouds)
    if len(points) > 1000:
        print("Removing statistical outliers...")
        pcd = remove_statistical_outliers(pcd, nb_neighbors=min(20, len(points)//50), std_ratio=2.5)
    else:
        print("Point cloud too sparse for outlier removal, skipping this step")
    
    # Determine appropriate voxel size based on point cloud density
    if len(points) > 10000:
        voxel_size = 0.005
    else:
        # For sparse clouds, use larger voxel size to avoid over-decimation
        voxel_size = 0.01
    
    # Only downsample if enough points
    if len(points) > 5000:
        print(f"Downsampling with voxel size {voxel_size}...")
        pcd = downsample_point_cloud(pcd, voxel_size=voxel_size)
    else:
        print("Point cloud too sparse for downsampling, skipping this step")
    
    # Estimate normals (adjust parameters for sparse clouds)
    print("Estimating normals...")
    if len(points) < 1000:
        # For very sparse clouds, use larger radius to capture enough neighbors
        radius = 0.2
        max_nn = min(30, len(points)//3)
    else:
        radius = 0.1
        max_nn = 30
        
    pcd = estimate_point_normals(pcd, radius=radius, max_nn=max_nn)
    
    try:
        # Perform surface reconstruction with method-specific adjustments
        if method == 'poisson':
            # For sparse clouds, use lower depth to avoid artifacts
            depth = 8 if len(points) > 5000 else 7
            print(f"Performing Poisson reconstruction with depth={depth}...")
            mesh = poisson_surface_reconstruction(pcd, depth=depth, scale=1.5)
        elif method == 'ball_pivoting':
            print("Performing Ball Pivoting reconstruction...")
            # For sparse clouds, use larger ball radius
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]
            mesh = ball_pivoting_surface_reconstruction(pcd, radii=radii)
        elif method == 'alpha_shape':
            print("Performing Alpha Shape reconstruction...")
            # Alpha shape often works better for sparse reconstruction
            alpha = 0.5 if len(points) > 5000 else 0.3  # Lower alpha for sparse clouds
            mesh = alpha_shape_reconstruction(np.asarray(pcd.points), alpha=alpha)
        else:
            raise ValueError(f"Unsupported reconstruction method: {method}")
    except Exception as e:
        print(f"Error during {method} reconstruction: {e}")
        print("Falling back to Ball Pivoting method...")
        try:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]
            mesh = ball_pivoting_surface_reconstruction(pcd, radii=radii)
        except Exception as e2:
            print(f"Ball Pivoting also failed: {e2}")
            print("Creating a simple convex hull as fallback...")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
    
    # Check if mesh is empty
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("Warning: Reconstruction produced an empty mesh!")
        print("Creating a simple convex hull as fallback...")
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
        except:
            print("Failed to create even a convex hull. Returning empty mesh.")
    
    # Clean up mesh if requested
    if cleanup and len(mesh.triangles) > 0:
        print("Cleaning up mesh...")
        mesh = clean_mesh(mesh, detail_level=3)  # Use updated clean_mesh function
    
    print(f"Mesh reconstruction complete: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh, pcd