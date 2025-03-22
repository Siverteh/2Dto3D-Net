import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import open3d as o3d
from IPython.display import display

def plot_3d_points(points, color=None, size=2, title="3D Point Cloud"):
    """
    Plot 3D points using matplotlib.
    
    Args:
        points: Nx3 array of 3D points.
        color: Optional point colors or colormap.
        size: Point size.
        title: Plot title.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if color is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=size)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def plot_interactive_point_cloud(points, colors=None, size=2, title="Interactive 3D Point Cloud"):
    """
    Plot interactive 3D point cloud using Plotly.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Optional point colors (Nx3 RGB array).
        size: Point size.
        title: Plot title.
    """
    if colors is None:
        colors = np.ones((len(points), 3)) * 0.5  # Default gray
    
    # Convert RGB to hex for Plotly
    color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                for r, g, b in colors]
    
    # Create scatter3d trace
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color_str,
            opacity=0.8
        )
    )
    
    # Calculate data bounds
    x_range = [np.min(points[:, 0]), np.max(points[:, 0])]
    y_range = [np.min(points[:, 1]), np.max(points[:, 1])]
    z_range = [np.min(points[:, 2]), np.max(points[:, 2])]
    
    # Ensure equal aspect ratio
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    ) / 2
    
    mid_x = sum(x_range) / 2
    mid_y = sum(y_range) / 2
    mid_z = sum(z_range) / 2
    
    # Create figure
    fig = go.Figure(data=[scatter])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
            aspectmode='cube'  # Equal aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Show the figure
    fig.show()

def create_open3d_point_cloud(points, colors=None):
    """
    Create an Open3D point cloud from points and optional colors.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Optional Nx3 array of RGB colors (values in [0, 1]).
        
    Returns:
        Open3D point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_point_cloud_o3d(pcd, window_name="Open3D Point Cloud"):
    """
    Visualize point cloud using Open3D.
    
    Args:
        pcd: Open3D point cloud object.
        window_name: Name of the visualization window.
    """
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    # Open visualization window
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)

def visualize_point_cloud_with_cameras(points, colors, camera_positions, window_name="Point Cloud with Cameras"):
    """
    Visualize point cloud with camera positions using Open3D.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Nx3 array of RGB colors.
        camera_positions: Dictionary of camera positions {image_name: position}.
        window_name: Name of the visualization window.
    """
    # Create point cloud
    pcd = create_open3d_point_cloud(points, colors)
    
    # Create geometries list
    geometries = [pcd]
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Add camera frustums
    for name, pos in camera_positions.items():
        # Create camera frustum
        frustum = create_camera_frustum(pos, size=0.2)
        geometries.append(frustum)
    
    # Open visualization window
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def create_camera_frustum(position, size=0.2):
    """
    Create a camera frustum visualization.
    
    Args:
        position: 3D position of the camera.
        size: Size of the frustum.
        
    Returns:
        LineSet object representing the camera frustum.
    """
    # Define frustum vertices
    vertices = np.array([
        [0, 0, 0],  # Camera center
        [size, size, size],  # Front-top-right
        [size, -size, size],  # Front-bottom-right
        [-size, -size, size],  # Front-bottom-left
        [-size, size, size]   # Front-top-left
    ])
    
    # Add position offset
    vertices += np.array(position)
    
    # Define lines
    lines = np.array([
        [0, 1],  # Center to front-top-right
        [0, 2],  # Center to front-bottom-right
        [0, 3],  # Center to front-bottom-left
        [0, 4],  # Center to front-top-left
        [1, 2],  # Front face
        [2, 3],
        [3, 4],
        [4, 1]
    ])
    
    # Create LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines)
    )
    
    # Set line colors
    line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(lines))]))
    
    return line_set

def visualize_point_cloud_with_normals(points, normals, color=[0.8, 0.8, 0.8], window_name="Point Cloud with Normals"):
    """
    Visualize point cloud with normals using Open3D.
    
    Args:
        points: Nx3 array of 3D points.
        normals: Nx3 array of normal vectors.
        color: Color for all points.
        window_name: Name of the visualization window.
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Set colors
    if isinstance(color, list):
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    else:  # Assume it's an array of colors
        pcd.colors = o3d.utility.Vector3dVector(color)
    
    # Open visualization window
    o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=True)

def plot_point_cloud_cross_section(points, colors=None, axis=2, position=0, thickness=0.1, title="Point Cloud Cross Section"):
    """
    Plot a cross-section of a point cloud.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Optional Nx3 array of RGB colors.
        axis: Axis perpendicular to cross-section (0=X, 1=Y, 2=Z).
        position: Position along the axis for cross-section.
        thickness: Thickness of the cross-section.
        title: Plot title.
    """
    # Select points within cross-section
    mask = np.abs(points[:, axis] - position) < thickness
    
    if np.sum(mask) == 0:
        print(f"No points in cross-section at position {position} with thickness {thickness}")
        return
    
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    
    # Plot in 2D
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine which axes to use
    axes = [i for i in range(3) if i != axis]
    x, y = axes
    
    # Plot points
    if filtered_colors is not None:
        ax.scatter(filtered_points[:, x], filtered_points[:, y], 
                  c=filtered_colors, s=2)
    else:
        ax.scatter(filtered_points[:, x], filtered_points[:, y], s=2)
    
    # Set labels
    axes_labels = ['X', 'Y', 'Z']
    ax.set_xlabel(axes_labels[x])
    ax.set_ylabel(axes_labels[y])
    ax.set_title(f"{title} ({axes_labels[axis]}={position}±{thickness})")
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def create_point_cloud_animation(points, colors=None, n_frames=36, output_file=None):
    """
    Create a rotating animation of a point cloud.
    
    Args:
        points: Nx3 array of 3D points.
        colors: Optional Nx3 array of RGB colors.
        n_frames: Number of frames in the animation.
        output_file: Output filename (HTML file).
        
    Returns:
        Plotly figure with animation.
    """
    # Calculate center of point cloud
    center = np.mean(points, axis=0)
    
    # Calculate distance from center to furthest point
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    max_distance = np.max(distances)
    
    # Convert RGB to hex for Plotly if colors provided
    if colors is not None:
        color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                    for r, g, b in colors]
    else:
        color_str = 'rgb(100, 100, 100)'
    
    # Create figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_str,
            opacity=0.8
        )
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


def visualize_dense_point_cloud(point_cloud, window_title="3D Point Cloud Viewer", 
                                width=1280, height=720, show_coordinate_frame=True,
                                background_color=[1, 1, 1]):
    """
    Visualize a point cloud in a new window using Open3D's native visualizer.
    This opens an external window and not inline in the notebook.
    
    Args:
        point_cloud: Open3D point cloud object or NumPy array of points
        window_title: Title for the visualization window
        width: Width of the window in pixels
        height: Height of the window in pixels
        show_coordinate_frame: Whether to show the coordinate axes
        background_color: RGB background color (each value 0-1)
        
    Returns:
        None (opens visualization window)
    """
    import open3d as o3d
    import numpy as np
    
    # Convert numpy array to Open3D point cloud if needed
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        # Assume it's a NumPy array
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = temp_pcd
    
    # Check if the point cloud has colors
    if not point_cloud.has_colors():
        print("Point cloud has no colors. Generating random colors...")
        # Generate random colors
        colors = np.random.rand(len(point_cloud.points), 3)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Print stats
    print(f"Visualizing point cloud with {len(point_cloud.points):,} points")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=width, height=height)
    
    # Add point cloud
    vis.add_geometry(point_cloud)
    
    # Add coordinate frame if requested
    if show_coordinate_frame:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
    
    # Set background color
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    
    # Set better point cloud visualization options
    opt.point_size = 1.5
    
    # Set view control
    ctrl = vis.get_view_control()
    
    # Set a good default viewpoint (can be adjusted in the window)
    ctrl.set_zoom(0.8)
    ctrl.set_front([1, 1, 1])  # Isometric view
    ctrl.set_lookat([0, 0, 0])
    ctrl.set_up([0, 0, 1])
    
    # Instructions
    print("\nVisualization window opened!")
    print("Controls:")
    print("  Left-click + drag: Rotate")
    print("  Right-click + drag: Pan")
    print("  Mouse wheel/middle-click + drag: Zoom")
    print("  'h': Show help message with all controls")
    print("  '-/+': Decrease/increase point size")
    print("  'r': Reset view")
    print("  'c': Change background color")
    print("  'q': Close window")
    
    # Run visualization
    vis.run()
    vis.destroy_window()
