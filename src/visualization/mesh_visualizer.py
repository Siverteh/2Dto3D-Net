import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import open3d as o3d

def visualize_mesh_o3d(mesh, window_name="Mesh Visualization"):
    """
    Visualize mesh using Open3D.
    
    Args:
        mesh: Open3D triangle mesh.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Ensure mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Open visualization window
    o3d.visualization.draw_geometries([mesh, coord_frame], window_name=window_name)

def visualize_mesh_with_texture_o3d(mesh, window_name="Textured Mesh Visualization"):
    """
    Visualize textured mesh using Open3D.
    
    Args:
        mesh: Open3D triangle mesh with texture.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Ensure mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Open visualization window with different settings for textured mesh
    o3d.visualization.draw_geometries(
        [mesh, coord_frame],
        window_name=window_name,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )

def plot_mesh_matplotlib(vertices, faces, figsize=(10, 10), title="Mesh Visualization"):
    """
    Plot mesh using matplotlib.
    
    Args:
        vertices: Nx3 array of vertices.
        faces: Mx3 array of face indices.
        figsize: Figure size as (width, height).
        title: Plot title.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create list of vertex coordinates for each face
    mesh = Poly3DCollection([vertices[face] for face in faces], alpha=0.5)
    mesh.set_edgecolor('k')
    
    # Add mesh to plot
    ax.add_collection3d(mesh)
    
    # Set axis limits
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_textured_mesh_matplotlib(vertices, faces, vertex_colors=None, figsize=(10, 10), title="Textured Mesh"):
    """
    Plot textured mesh using matplotlib with vertex colors.
    
    Args:
        vertices: Nx3 array of vertices.
        faces: Mx3 array of face indices.
        vertex_colors: Nx3 array of vertex colors (RGB values in [0, 1]).
        figsize: Figure size as (width, height).
        title: Plot title.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create list of vertex coordinates and colors for each face
    mesh_faces = []
    face_colors = []
    
    for face in faces:
        # Get vertices for this face
        verts = vertices[face]
        mesh_faces.append(verts)
        
        if vertex_colors is not None:
            # Average color of this face
            colors = vertex_colors[face]
            face_colors.append(np.mean(colors, axis=0))
        else:
            face_colors.append([0.7, 0.7, 0.7])  # Default gray
    
    # Create mesh collection
    mesh = Poly3DCollection(mesh_faces, alpha=0.8)
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor('k')
    
    # Add mesh to plot
    ax.add_collection3d(mesh)
    
    # Set axis limits
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_interactive_mesh(vertices, faces, vertex_colors=None, title="Interactive Mesh"):
    """
    Plot an interactive 3D mesh using Plotly.
    
    Args:
        vertices: Nx3 array of vertices.
        faces: Mx3 array of face indices.
        vertex_colors: Nx3 array of vertex colors (RGB values in [0, 1]).
        title: Plot title.
    """
    # Prepare face indices for Plotly (0-based to 1-based indexing)
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    
    # Prepare vertex coordinates
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # Prepare color data
    if vertex_colors is not None:
        colorscale = [
            [i/(len(vertex_colors)-1), f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'] 
            for i, (r, g, b) in enumerate(vertex_colors)
        ]
        color_values = list(range(len(vertex_colors)))
    else:
        colorscale = [[0, 'rgb(200, 200, 200)'], [1, 'rgb(200, 200, 200)']]
        color_values = [0] * len(vertices)
    
    # Create figure
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            vertexcolor=vertex_colors if vertex_colors is not None else None,
            opacity=0.8
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Show the figure
    fig.show()

def visualize_mesh_wireframe_o3d(mesh, window_name="Mesh Wireframe"):
    """
    Visualize mesh wireframe using Open3D.
    
    Args:
        mesh: Open3D triangle mesh.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Ensure mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Open visualization window with wireframe
    o3d.visualization.draw_geometries(
        [mesh, coord_frame],
        window_name=window_name,
        mesh_show_wireframe=True,
        mesh_show_back_face=True
    )

def visualize_mesh_with_camera_poses(mesh, camera_poses, window_name="Mesh with Cameras"):
    """
    Visualize mesh with camera poses using Open3D.
    
    Args:
        mesh: Open3D triangle mesh.
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Ensure mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Create geometries list
    geometries = [mesh, coord_frame]
    
    # Add camera frustums
    for name, (R, t) in camera_poses.items():
        # Create camera frustum
        frustum = create_camera_frustum_mesh(R, t, size=0.2, color=[1, 0, 0])
        geometries.append(frustum)
    
    # Open visualization window
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def create_camera_frustum_mesh(R, t, size=0.2, color=[1, 0, 0]):
    """
    Create a mesh representing a camera frustum.
    
    Args:
        R: Camera rotation matrix.
        t: Camera translation vector.
        size: Size of the frustum.
        color: Color of the frustum.
        
    Returns:
        Open3D TriangleMesh object.
    """
    # Camera center in world coordinates is -R^T * t
    center = -R.T @ t
    
    # Camera coordinate system
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = -R[:, 2]  # Camera looks along negative Z
    
    # Define frustum vertices in camera coordinates
    vertices_cam = np.array([
        [0, 0, 0],  # Camera center
        [size, size, size * 1.5],  # Front-top-right
        [size, -size, size * 1.5],  # Front-bottom-right
        [-size, -size, size * 1.5],  # Front-bottom-left
        [-size, size, size * 1.5]   # Front-top-left
    ])
    
    # Transform to world coordinates
    vertices_world = []
    for v in vertices_cam:
        # Apply camera rotation and translation
        point = center + v[0] * x_axis + v[1] * y_axis + v[2] * z_axis
        vertices_world.append(point)
    
    vertices_world = np.array(vertices_world)
    
    # Define triangles
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Set mesh color
    mesh.paint_uniform_color(color)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    return mesh

def compare_meshes(mesh1, mesh2, window_name="Mesh Comparison"):
    """
    Compare two meshes using Open3D.
    
    Args:
        mesh1: First Open3D triangle mesh.
        mesh2: Second Open3D triangle mesh.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Ensure meshes have normals
    if not mesh1.has_vertex_normals():
        mesh1.compute_vertex_normals()
    
    if not mesh2.has_vertex_normals():
        mesh2.compute_vertex_normals()
    
    # Set different colors for each mesh
    mesh1.paint_uniform_color([1, 0, 0])  # Red
    mesh2.paint_uniform_color([0, 0, 1])  # Blue
    
    # Open visualization window
    o3d.visualization.draw_geometries(
        [mesh1, mesh2, coord_frame],
        window_name=window_name
    )

def visualize_mesh_cross_section(mesh, axis=2, position=0, thickness=0.1, window_name="Mesh Cross Section"):
    """
    Visualize a cross-section of a mesh using Open3D.
    
    Args:
        mesh: Open3D triangle mesh.
        axis: Axis perpendicular to cross-section (0=X, 1=Y, 2=Z).
        position: Position along the axis for cross-section.
        thickness: Thickness of the cross-section.
        window_name: Name of the visualization window.
    """
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # Create a cropping box
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    
    # Modify bounds for the specific axis
    min_bound[axis] = position - thickness/2
    max_bound[axis] = position + thickness/2
    
    # Crop the mesh
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_mesh = mesh.crop(bbox)
    
    # Visualize
    visualize_mesh_o3d(cropped_mesh, window_name=window_name)

def create_mesh_animation(mesh, n_frames=36, output_file=None):
    """
    Create a rotating animation of a mesh.
    
    Args:
        mesh: Open3D triangle mesh.
        n_frames: Number of frames in the animation.
        output_file: Output filename (HTML file).
        
    Returns:
        Plotly figure with animation.
    """
    # Extract mesh data
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Prepare faces for Plotly (0-based to 1-based indexing)
    i = triangles[:, 0]
    j = triangles[:, 1]
    k = triangles[:, 2]
    
    # Extract vertex colors if available
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        vertexcolor = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                      for r, g, b in vertex_colors]
    else:
        vertexcolor = None
    
    # Calculate center of mesh
    center = np.mean(vertices, axis=0)
    
    # Calculate distance from center to furthest vertex
    distances = np.sqrt(np.sum((vertices - center) ** 2, axis=1))
    max_distance = np.max(distances)
    
    # Create figure
    fig = go.Figure()
    
    # Add mesh trace
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        vertexcolor=vertexcolor,
        opacity=0.8
    ))
    
    # Create frames for rotation
    frames = []
    for i in range(n_frames):
        angle = i * 2 * np.pi / n_frames
        camera = dict(
            eye=dict(
                x=center[0] + max_distance * 2 * np.sin(angle),
                y=center[1] + max_distance * 2 * np.cos(angle),
                z=center[2] + max_distance * 0.5
            ),
            center=dict(x=center[0], y=center[1], z=center[2]),
            up=dict(x=0, y=0, z=1)
        )
        
        frames.append(go.Frame(layout=dict(scene_camera=camera)))
    
    fig.frames = frames
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
            )]
        )]
    )
    
    # Save animation if output file specified
    if output_file:
        fig.write_html(output_file)
    
    return fig