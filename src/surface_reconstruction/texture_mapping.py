import numpy as np
import open3d as o3d
import cv2
import math
from collections import defaultdict

def generate_uv_mapping(mesh):
    """
    Generate UV mapping for a mesh using parameterization.
    
    Args:
        mesh: Open3D triangle mesh.
        
    Returns:
        Mesh with UV coordinates.
    """
    # Note: Open3D does not have built-in UV unwrapping
    # This is a simplified implementation that uses a spherical projection
    # For complex meshes, consider using external libraries like xatlas
    
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # Calculate mesh centroid
    centroid = np.mean(vertices, axis=0)
    
    # Shift vertices to center at origin
    vertices_centered = vertices - centroid
    
    # Calculate spherical coordinates
    r = np.sqrt(np.sum(vertices_centered**2, axis=1))
    theta = np.arccos(vertices_centered[:, 2] / np.clip(r, 1e-10, None))
    phi = np.arctan2(vertices_centered[:, 1], vertices_centered[:, 0])
    
    # Map to UV space (normalized to [0, 1])
    u = (phi + math.pi) / (2 * math.pi)
    v = theta / math.pi
    
    # Create UV array
    uv = np.column_stack((u, v))
    
    # Set mesh texture coordinates
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.repeat(uv, 3, axis=0)[np.asarray(mesh.triangles).flatten()])
    
    return mesh

def generate_texture_atlas(mesh, images, camera_poses, K, atlas_size=(2048, 2048)):
    """
    Generate texture atlas for mesh using multiple images.
    
    Args:
        mesh: Open3D triangle mesh with vertex normals.
        images: List of (image, filename) tuples.
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        K: Camera intrinsic matrix.
        atlas_size: Size of texture atlas (width, height).
        
    Returns:
        Mesh with texture atlas and texture image.
    """
    # Ensure mesh has UV coordinates
    if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
        mesh = generate_uv_mapping(mesh)
    
    # Create empty texture atlas
    atlas = np.zeros((atlas_size[1], atlas_size[0], 3), dtype=np.uint8)
    
    # Get mesh data
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
    
    # For each triangle, determine best image to use for texturing
    for i, triangle in enumerate(triangles):
        # Get triangle vertices and normals
        tri_vertices = vertices[triangle]
        tri_center = np.mean(tri_vertices, axis=0)
        tri_normal = np.mean(normals[triangle], axis=0)
        tri_normal = tri_normal / np.linalg.norm(tri_normal)
        
        # Find best image for this triangle
        best_image = None
        best_score = -1
        
        for img, filename in images:
            if filename not in camera_poses:
                continue
            
            # Get camera pose
            R, t = camera_poses[filename]
            
            # Camera center in world coordinates
            camera_center = -R.T @ t
            
            # Vector from triangle center to camera
            view_vector = camera_center - tri_center
            view_vector = view_vector / np.linalg.norm(view_vector)
            
            # Calculate angle between triangle normal and view vector
            angle = np.arccos(np.clip(np.dot(tri_normal, view_vector), -1.0, 1.0))
            
            # Check if triangle is visible from camera (front-facing)
            if angle < math.pi / 2:
                # Project triangle to image
                points_3d = tri_vertices.reshape(-1, 3)
                points_2d = []
                
                for pt in points_3d:
                    # Transform to camera coordinates
                    pt_cam = R @ (pt - camera_center) - t
                    
                    # Project to image
                    x = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
                    y = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
                    
                    points_2d.append([x, y])
                
                points_2d = np.array(points_2d)
                
                # Check if all points are within image bounds
                h, w = img.shape[:2]
                if (points_2d[:, 0] >= 0).all() and (points_2d[:, 0] < w).all() and \
                   (points_2d[:, 1] >= 0).all() and (points_2d[:, 1] < h).all():
                    # Calculate score based on angle (closer to 0 is better)
                    score = math.pi / 2 - angle
                    
                    if score > best_score:
                        best_score = score
                        best_image = (img, filename, points_2d)
        
        # If a suitable image was found, project texture to atlas
        if best_image is not None:
            img, filename, points_2d = best_image
            
            # Get UV coordinates for this triangle
            uv_coords = triangle_uvs[i]
            
            # Scale UV to atlas size
            uv_pixels = uv_coords * np.array([atlas_size[0], atlas_size[1]])
            
            # Create affine transform from image to atlas
            src_points = points_2d.astype(np.float32)
            dst_points = uv_pixels.astype(np.float32)
            
            transform = cv2.getAffineTransform(src_points, dst_points)
            
            # Apply transform to copy image region to atlas
            cv2.warpAffine(img, transform, atlas_size, atlas, cv2.INTER_LINEAR)
    
    # Create mesh material and texture
    mesh.textures = [o3d.geometry.Image(atlas)]
    
    return mesh, atlas

def project_image_to_mesh(mesh, image, camera_pose, K):
    """
    Project a single image onto a mesh.
    
    Args:
        mesh: Open3D triangle mesh.
        image: Image to project.
        camera_pose: (R, t) of the camera.
        K: Camera intrinsic matrix.
        
    Returns:
        Mesh with projected vertex colors.
    """
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # Get camera parameters
    R, t = camera_pose
    
    # Camera center in world coordinates
    camera_center = -R.T @ t
    
    # For each vertex, project to image and get color
    colors = np.zeros((len(vertices), 3))
    visibility = np.zeros(len(vertices), dtype=bool)
    
    for i, vertex in enumerate(vertices):
        # Vector from vertex to camera
        view_vector = camera_center - vertex
        view_vector = view_vector / np.linalg.norm(view_vector)
        
        # Transform to camera coordinates
        vertex_cam = R @ (vertex - t)
        
        # Check if vertex is in front of camera
        if vertex_cam[2] > 0:
            # Project to image
            x = K[0, 0] * vertex_cam[0] / vertex_cam[2] + K[0, 2]
            y = K[1, 1] * vertex_cam[1] / vertex_cam[2] + K[1, 2]
            
            # Check if projection is within image bounds
            h, w = image.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # Get color from image
                color = image[int(y), int(x)] / 255.0  # Normalize to [0, 1]
                colors[i] = color
                visibility[i] = True
    
    # Only update vertices that are visible
    if np.any(visibility):
        # If mesh already has colors, blend with existing colors
        if mesh.has_vertex_colors():
            existing_colors = np.asarray(mesh.vertex_colors)
            colors[~visibility] = existing_colors[~visibility]
            
            # For visible vertices, blend with existing colors
            alpha = 0.5  # Blend factor
            colors[visibility] = alpha * colors[visibility] + (1 - alpha) * existing_colors[visibility]
        
        # Set mesh colors
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh

def blend_vertex_colors(mesh, images, camera_poses, K):
    """
    Blend vertex colors from multiple images.
    
    Args:
        mesh: Open3D triangle mesh.
        images: List of (image, filename) tuples.
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        K: Camera intrinsic matrix.
        
    Returns:
        Mesh with blended vertex colors.
    """
    # Initialize mesh with no colors
    vertices = np.asarray(mesh.vertices)
    colors = np.zeros((len(vertices), 3))
    weights = np.zeros(len(vertices))
    
    # For each image, project colors to mesh
    for img, filename in images:
        if filename not in camera_poses:
            continue
        
        # Get camera pose
        pose = camera_poses[filename]
        
        # Project image to mesh
        temp_mesh = o3d.geometry.TriangleMesh(mesh)
        temp_mesh = project_image_to_mesh(temp_mesh, img, pose, K)
        
        # Get projected colors
        if temp_mesh.has_vertex_colors():
            projected_colors = np.asarray(temp_mesh.vertex_colors)
            
            # For each vertex, update color if visible
            for i, vertex in enumerate(vertices):
                # Check if color was updated (non-zero)
                if np.any(projected_colors[i] > 0):
                    # Weighted average based on number of images that see this vertex
                    weights[i] += 1
                    colors[i] += (projected_colors[i] - colors[i]) / weights[i]
    
    # Set final colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh

def save_textured_mesh(mesh, filename):
    """
    Save textured mesh to file.
    
    Args:
        mesh: Open3D triangle mesh with textures.
        filename: Output filename (OBJ or PLY).
    """
    # Save mesh
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Saved textured mesh to {filename}")

def create_textured_mesh_from_point_cloud(points, colors, images, camera_poses, K, 
                                         reconstruction_method='poisson'):
    """
    Create textured mesh from point cloud.
    
    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors (values in [0, 1]).
        images: List of (image, filename) tuples.
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        K: Camera intrinsic matrix.
        reconstruction_method: 'poisson', 'ball_pivoting', or 'alpha_shape'.
        
    Returns:
        Textured triangle mesh.
    """
    # Generate mesh from point cloud
    mesh, _ = process_point_cloud_to_mesh(points, colors, method=reconstruction_method)
    
    # Create texture mapping (two options)
    use_vertex_colors = True  # Simpler, but lower quality
    
    if use_vertex_colors:
        # Option 1: Use vertex colors (simpler, but lower quality)
        textured_mesh = blend_vertex_colors(mesh, images, camera_poses, K)
    else:
        # Option 2: Create texture atlas (higher quality, but more complex)
        textured_mesh, _ = generate_texture_atlas(mesh, images, camera_poses, K)
    
    return textured_mesh