

# Import necessary libraries
import numpy as np
import open3d as o3d
import os

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
    
    return mesh

def smooth_mesh(mesh, method="taubin", iterations=10, **kwargs):
    """
    Smooth a triangle mesh using different algorithms.
    
    Args:
        mesh: Open3D triangle mesh
        method: Smoothing method ('laplacian', 'taubin', 'humphrey', or 'subdivision')
        iterations: Number of smoothing iterations
        **kwargs: Additional parameters:
            - lambda_filter: Positive filter parameter for Taubin (default=0.5)
            - mu: Negative filter parameter for Taubin (default=-0.53)
            - strength: Smoothing strength for Laplacian (default=0.5)
            - preserve_features: Whether to preserve sharp features (default=True)
            - feature_angle: Angle threshold for feature preservation (default=20)
    
    Returns:
        Smoothed Open3D triangle mesh
    """
    import open3d as o3d
    import numpy as np
    import copy
    
    # Create a copy of the input mesh to avoid modifying the original
    smoothed_mesh = copy.deepcopy(mesh)
    
    # Make sure the mesh is manifold for best results
    smoothed_mesh.remove_degenerate_triangles()
    smoothed_mesh.remove_duplicated_triangles()
    smoothed_mesh.remove_duplicated_vertices()
    smoothed_mesh.remove_non_manifold_edges()
    
    # Ensure the mesh has vertex normals
    if not smoothed_mesh.has_vertex_normals():
        print("Computing vertex normals...")
        smoothed_mesh.compute_vertex_normals()
    
    vertices = np.asarray(smoothed_mesh.vertices)
    triangles = np.asarray(smoothed_mesh.triangles)
    num_vertices = len(vertices)
    
    if method == "laplacian":
        # Laplacian smoothing with configurable strength
        strength = kwargs.get('strength', 0.5)
        preserve_features = kwargs.get('preserve_features', True)
        feature_angle = kwargs.get('feature_angle', 20) * np.pi / 180  # Convert to radians
        
        print(f"Applying Laplacian smoothing ({iterations} iterations, strength={strength})")
        
        # Create vertex adjacency list
        adjacency = [[] for _ in range(num_vertices)]
        for triangle in triangles:
            for i in range(3):
                adjacency[triangle[i]].append(triangle[(i+1)%3])
                adjacency[triangle[i]].append(triangle[(i+2)%3])
        
        # Remove duplicates in adjacency lists
        for i in range(num_vertices):
            adjacency[i] = list(set(adjacency[i]))
        
        # Compute vertex normals if feature preservation is enabled
        if preserve_features:
            normals = np.asarray(smoothed_mesh.vertex_normals)
        
        # Iterative smoothing
        for iteration in range(iterations):
            new_vertices = np.copy(vertices)
            
            for i in range(num_vertices):
                if not adjacency[i]:
                    continue
                
                # Calculate centroid of neighbors
                neighbor_count = len(adjacency[i])
                centroid = np.zeros(3)
                for neighbor in adjacency[i]:
                    centroid += vertices[neighbor]
                centroid /= neighbor_count
                
                # Apply smoothing with strength control
                displacement = centroid - vertices[i]
                
                # Feature preservation (if enabled)
                if preserve_features:
                    normal = normals[i]
                    for neighbor in adjacency[i]:
                        neighbor_normal = normals[neighbor]
                        dot_product = np.dot(normal, neighbor_normal)
                        
                        # Check if angle between normals is greater than threshold
                        if dot_product < np.cos(feature_angle):
                            # Reduce smoothing strength near features
                            strength_factor = 0.1
                            displacement *= strength_factor
                            break
                
                new_vertices[i] = vertices[i] + displacement * strength
            
            vertices = new_vertices
        
        # Update mesh vertices
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
    elif method == "taubin":
        # Taubin smoothing (alternating positive and negative Laplacian)
        lambda_filter = kwargs.get('lambda_filter', 0.5)
        mu = kwargs.get('mu', -0.53)
        
        print(f"Applying Taubin smoothing ({iterations} iterations, lambda={lambda_filter}, mu={mu})")
        
        # Create vertex adjacency list
        adjacency = [[] for _ in range(num_vertices)]
        for triangle in triangles:
            for i in range(3):
                adjacency[triangle[i]].append(triangle[(i+1)%3])
                adjacency[triangle[i]].append(triangle[(i+2)%3])
        
        # Remove duplicates in adjacency lists
        for i in range(num_vertices):
            adjacency[i] = list(set(adjacency[i]))
        
        # Iterative smoothing
        for iteration in range(iterations):
            # First pass: positive Laplacian with lambda
            new_vertices = np.copy(vertices)
            
            for i in range(num_vertices):
                if not adjacency[i]:
                    continue
                
                # Calculate centroid of neighbors
                neighbor_count = len(adjacency[i])
                centroid = np.zeros(3)
                for neighbor in adjacency[i]:
                    centroid += vertices[neighbor]
                centroid /= neighbor_count
                
                # Apply positive Laplacian
                displacement = centroid - vertices[i]
                new_vertices[i] = vertices[i] + displacement * lambda_filter
            
            vertices = new_vertices
            
            # Second pass: negative Laplacian with mu
            new_vertices = np.copy(vertices)
            
            for i in range(num_vertices):
                if not adjacency[i]:
                    continue
                
                # Calculate centroid of neighbors
                neighbor_count = len(adjacency[i])
                centroid = np.zeros(3)
                for neighbor in adjacency[i]:
                    centroid += vertices[neighbor]
                centroid /= neighbor_count
                
                # Apply negative Laplacian
                displacement = centroid - vertices[i]
                new_vertices[i] = vertices[i] + displacement * mu
            
            vertices = new_vertices
        
        # Update mesh vertices
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
    elif method == "humphrey":
        # Humphrey's smoothing (improved Taubin)
        lambda_filter = kwargs.get('lambda_filter', 0.35)
        iterations *= 2  # Humphrey does more iterations
        
        print(f"Applying Humphrey smoothing ({iterations} iterations, lambda={lambda_filter})")
        
        # Create vertex adjacency list
        adjacency = [[] for _ in range(num_vertices)]
        for triangle in triangles:
            for i in range(3):
                adjacency[triangle[i]].append(triangle[(i+1)%3])
                adjacency[triangle[i]].append(triangle[(i+2)%3])
        
        # Remove duplicates in adjacency lists
        for i in range(num_vertices):
            adjacency[i] = list(set(adjacency[i]))
        
        # Iterative smoothing
        for iteration in range(iterations):
            # Calculate parameter for this iteration (alternates between positive and negative)
            alpha = lambda_filter if iteration % 2 == 0 else -lambda_filter
            
            new_vertices = np.copy(vertices)
            
            for i in range(num_vertices):
                if not adjacency[i]:
                    continue
                
                # Calculate centroid of neighbors
                neighbor_count = len(adjacency[i])
                centroid = np.zeros(3)
                for neighbor in adjacency[i]:
                    centroid += vertices[neighbor]
                centroid /= neighbor_count
                
                # Apply smoothing
                displacement = centroid - vertices[i]
                new_vertices[i] = vertices[i] + displacement * alpha
            
            vertices = new_vertices
        
        # Update mesh vertices
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
    elif method == "subdivision":
        # Loop subdivision for smoother appearance
        print(f"Applying Loop subdivision (iterations={iterations})")
        
        for _ in range(iterations):
            smoothed_mesh = smoothed_mesh.subdivide_loop(1)
            
            # Optional smoothing after subdivision
            if kwargs.get('post_smooth', True):
                smoothed_mesh = smooth_mesh(
                    smoothed_mesh, 
                    method="taubin", 
                    iterations=1, 
                    lambda_filter=0.3, 
                    mu=-0.32
                )
    
    # Re-compute normals after smoothing
    smoothed_mesh.compute_vertex_normals()
    
    return smoothed_mesh