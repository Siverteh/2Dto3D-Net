import matplotlib.pyplot as plt
import numpy as np

def visualize_depth_maps(depth_maps, num_samples=4, colormap='viridis', figsize=(16, 8)):
    """
    Visualize depth maps inline in a Jupyter notebook.
    
    Args:
        depth_maps: Dictionary mapping image names to depth maps
        num_samples: Number of depth maps to visualize (default: 4)
        colormap: Matplotlib colormap to use for depth visualization
        figsize: Figure size for the plot
        
    Returns:
        None (displays the visualization inline)
    """
    
    # Get a subset of depth maps to display
    if len(depth_maps) <= num_samples:
        selected_items = list(depth_maps.items())
    else:
        step = max(1, len(depth_maps) // num_samples)
        selected_items = list(depth_maps.items())[::step][:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, len(selected_items)//2 + len(selected_items)%2, figsize=figsize)
    if len(selected_items) <= 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each depth map
    for i, (name, depth_map) in enumerate(selected_items):
        if i >= len(axes):
            break
            
        # Skip invalid depth maps
        if depth_map is None or not np.any(depth_map > 0):
            axes[i].text(0.5, 0.5, f"No valid depth\n{name}", 
                        horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')
            continue
        
        # Normalize depth for better visualization
        valid_mask = depth_map > 0
        if np.any(valid_mask):
            vmin = np.percentile(depth_map[valid_mask], 5)
            vmax = np.percentile(depth_map[valid_mask], 95)
        else:
            vmin, vmax = 0, 1
            
        # Plot depth map
        im = axes[i].imshow(depth_map, cmap=colormap, vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{name}\nDepth range: {vmin:.2f}-{vmax:.2f}")
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(selected_items), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    valid_maps = sum(1 for dm in depth_maps.values() if dm is not None and np.any(dm > 0))
    print(f"Showing {len(selected_items)} of {valid_maps} valid depth maps (from {len(depth_maps)} total)")

    