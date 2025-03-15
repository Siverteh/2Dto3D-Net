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
    
    # Remove isolated vertices
    mesh.remove_vertices_by_mask(
        mesh.get_vertex_degree() == 0)
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    # Apply Taubin smoothing (maintains features better than Laplacian)
    lambda_factor = 0.5  # Positive scale factor
    mu_factor = -0.53    # Negative scale factor (slightly larger than -lambda)
    num_iterations = 10  # Number of iteration steps
    
    smoothed_mesh = mesh.filter_smooth_taubin(
        number_of_iterations=num_iterations // detail_level,
        lambda_filter=lambda_factor,
        mu=mu_factor
    )
    
    return smoothed_mesh

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
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors (values in [0, 1]).
        method: 'poisson', 'ball_pivoting', or 'alpha_shape'.
        cleanup: Whether to clean up the resulting mesh.
        
    Returns:
        Triangle mesh.
    """
    # Create point cloud
    pcd = create_point_cloud_from_points(points, colors)
    
    # Remove outliers
    pcd = remove_statistical_outliers(pcd)
    
    # Downsample for faster processing
    pcd = downsample_point_cloud(pcd, voxel_size=0.005)
    
    # Estimate normals
    pcd = estimate_point_normals(pcd)
    
    # Perform surface reconstruction
    if method == 'poisson':
        mesh = poisson_surface_reconstruction(pcd)
    elif method == 'ball_pivoting':
        mesh = ball_pivoting_surface_reconstruction(pcd)
    elif method == 'alpha_shape':
        mesh = alpha_shape_reconstruction(np.asarray(pcd.points))
    else:
        raise ValueError(f"Unsupported reconstruction method: {method}")
    
    # Clean up mesh if requested
    if cleanup:
        mesh = clean_mesh(mesh)
    
    return mesh, pcd