import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import open3d as o3d

def plot_camera_poses(camera_poses, figsize=(10, 8), title="Camera Poses"):
    """
    Plot camera poses in 3D using matplotlib.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        figsize: Figure size as (width, height).
        title: Plot title.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera centers
    camera_centers = {}
    for name, (R, t) in camera_poses.items():
        # Camera center in world coordinates is -R^T * t
        center = -R.T @ t
        camera_centers[name] = center
    
    # Plot camera centers
    centers = np.array(list(camera_centers.values()))
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', marker='o', s=50)
    
    # Add camera labels
    for name, center in camera_centers.items():
        ax.text(center[0], center[1], center[2], name, fontsize=8)
    
    # Calculate camera coordinate axes
    for name, (R, t) in camera_poses.items():
        center = camera_centers[name]
        
        # Camera axes (columns of R)
        axis_length = 0.5  # Length of axis arrows
        
        # X axis (red)
        ax.quiver(center[0], center[1], center[2], 
                 R[0, 0], R[1, 0], R[2, 0], 
                 color='r', length=axis_length)
        
        # Y axis (green)
        ax.quiver(center[0], center[1], center[2], 
                 R[0, 1], R[1, 1], R[2, 1], 
                 color='g', length=axis_length)
        
        # Z axis (blue) - camera looks along negative Z
        ax.quiver(center[0], center[1], center[2], 
                 -R[0, 2], -R[1, 2], -R[2, 2], 
                 color='b', length=axis_length)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.max([
        np.max(centers[:, 0]) - np.min(centers[:, 0]),
        np.max(centers[:, 1]) - np.min(centers[:, 1]),
        np.max(centers[:, 2]) - np.min(centers[:, 2])
    ])
    
    mid_x = (np.max(centers[:, 0]) + np.min(centers[:, 0])) / 2
    mid_y = (np.max(centers[:, 1]) + np.min(centers[:, 1])) / 2
    mid_z = (np.max(centers[:, 2]) + np.min(centers[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def plot_camera_trajectory(camera_poses, figsize=(10, 8), title="Camera Trajectory"):
    """
    Plot camera trajectory in 3D using matplotlib.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        figsize: Figure size as (width, height).
        title: Plot title.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera centers
    centers = []
    names = []
    for name, (R, t) in camera_poses.items():
        # Camera center in world coordinates is -R^T * t
        center = -R.T @ t
        centers.append(center)
        names.append(name)
    
    # Convert to numpy array
    centers = np.array(centers)
    
    # Plot camera centers
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', marker='o')
    
    # Add camera labels
    for i, name in enumerate(names):
        ax.text(centers[i, 0], centers[i, 1], centers[i, 2], name, fontsize=8)
    
    # Plot trajectory
    # Sort based on image index if filenames are like "image001.png"
    try:
        # Extract indices from filenames
        indices = [int(''.join(filter(str.isdigit, name))) for name in names]
        # Sort centers based on indices
        centers_sorted = centers[np.argsort(indices)]
        ax.plot(centers_sorted[:, 0], centers_sorted[:, 1], centers_sorted[:, 2], 'b-')
    except:
        # If sorting fails, just connect in sequence
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 'b-')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.max([
        np.max(centers[:, 0]) - np.min(centers[:, 0]),
        np.max(centers[:, 1]) - np.min(centers[:, 1]),
        np.max(centers[:, 2]) - np.min(centers[:, 2])
    ])
    
    mid_x = (np.max(centers[:, 0]) + np.min(centers[:, 0])) / 2
    mid_y = (np.max(centers[:, 1]) + np.min(centers[:, 1])) / 2
    mid_z = (np.max(centers[:, 2]) + np.min(centers[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def plot_interactive_camera_poses(camera_poses, points=None, colors=None, title="Interactive Camera Poses"):
    """
    Plot interactive camera poses in 3D using Plotly.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        points: Optional Nx3 array of 3D points.
        colors: Optional point colors (Nx3 RGB array).
        title: Plot title.
    """
    # Extract camera centers
    camera_centers = {}
    for name, (R, t) in camera_poses.items():
        # Camera center in world coordinates is -R^T * t
        center = -R.T @ t
        camera_centers[name] = center
    
    # Create figure
    fig = go.Figure()
    
    # Add camera centers
    centers = np.array(list(camera_centers.values()))
    names = list(camera_centers.keys())
    
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color='red',
            symbol='square'
        ),
        text=names,
        textposition="top center",
        name='Camera Positions'
    ))
    
    # Add camera trajectory
    # Sort based on image index if filenames are like "image001.png"
    try:
        # Extract indices from filenames
        indices = [int(''.join(filter(str.isdigit, name))) for name in names]
        # Sort centers based on indices
        sort_indices = np.argsort(indices)
        centers_sorted = centers[sort_indices]
        names_sorted = [names[i] for i in sort_indices]
        
        fig.add_trace(go.Scatter3d(
            x=centers_sorted[:, 0],
            y=centers_sorted[:, 1],
            z=centers_sorted[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Camera Trajectory'
        ))
    except:
        # If sorting fails, just connect in sequence
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Camera Trajectory'
        ))
    
    # Add 3D points if provided
    if points is not None:
        if colors is None:
            colors = np.ones((len(points), 3)) * 0.5  # Default gray
        
        # Convert RGB to hex for Plotly
        color_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                    for r, g, b in colors]
        
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_str,
                opacity=0.8
            ),
            name='3D Points'
        ))
    
    # Calculate data bounds
    if points is not None:
        # Use both camera centers and points
        all_points = np.vstack([centers, points])
        x_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        z_range = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    else:
        # Use only camera centers
        x_range = [np.min(centers[:, 0]), np.max(centers[:, 0])]
        y_range = [np.min(centers[:, 1]), np.max(centers[:, 1])]
        z_range = [np.min(centers[:, 2]), np.max(centers[:, 2])]
    
    # Ensure equal aspect ratio
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    ) / 2
    
    mid_x = sum(x_range) / 2
    mid_y = sum(y_range) / 2
    mid_z = sum(z_range) / 2
    
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

def visualize_cameras_and_points_o3d(camera_poses, points=None, colors=None, window_name="Cameras and Points"):
    """
    Visualize cameras and points using Open3D.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        points: Optional Nx3 array of 3D points.
        colors: Optional Nx3 array of RGB colors.
        window_name: Name of the visualization window.
    """
    # Create geometries list
    geometries = []
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Add camera frustums
    for name, (R, t) in camera_poses.items():
        # Create camera frustum
        frustum = create_camera_frustum_mesh(R, t)
        geometries.append(frustum)
    
    # Add point cloud if provided
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries.append(pcd)
    
    # Open visualization window
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def plot_camera_coverage(camera_poses, points, point_observations, figsize=(10, 8), title="Camera Coverage"):
    """
    Plot number of cameras that observe each point.
    
    Args:
        camera_poses: Dictionary of camera poses {image_name: (R, t)}.
        points: Nx3 array of 3D points.
        point_observations: List of dictionaries mapping image names to observations.
        figsize: Figure size as (width, height).
        title: Plot title.
    """
    # Calculate number of observations per point
    num_observations = [len(obs) for obs in point_observations]
    
    # Define colormap
    max_obs = max(num_observations) if num_observations else 1
    cmap = plt.cm.jet
    norm = plt.Normalize(1, max_obs)
    
    # Plot points with color based on observation count
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=num_observations, cmap=cmap, norm=norm,
        s=2, alpha=0.8
    )
    
    # Add camera centers
    camera_centers = []
    for name, (R, t) in camera_poses.items():
        # Camera center in world coordinates is -R^T * t
        center = -R.T @ t
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    ax.scatter(
        camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
        c='r', marker='o', s=50
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Number of Observations')
    
    # Equal aspect ratio
    all_points = np.vstack([points, camera_centers])
    max_range = np.max([
        np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
        np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
        np.max(all_points[:, 2]) - np.min(all_points[:, 2])
    ])
    
    mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
    mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
    mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()