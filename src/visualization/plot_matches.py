import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def plot_keypoints(img, keypoints, title=None, figsize=(10, 8)):
    """
    Plot keypoints on an image.
    
    Args:
        img: Input image.
        keypoints: List of keypoints.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).
    """
    # Draw keypoints
    img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Plot using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(img_kp)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_matches(img1, kp1, img2, kp2, matches, mask=None, title=None, figsize=(15, 8)):
    """
    Plot matches between two images.
    
    Args:
        img1, img2: Two input images.
        kp1, kp2: Keypoints from two images.
        matches: List of matches.
        mask: Optional mask for inlier matches.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).
    """
    # Create a new image showing the matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                               matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                               matchesMask=mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Plot using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(match_img)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_matches_side_by_side(img1, kp1, img2, kp2, matches, mask=None, title=None, 
                             max_matches=100, figsize=(15, 10)):
    """
    Plot matches between two images side-by-side with lines connecting matches.
    
    Args:
        img1, img2: Two input images.
        kp1, kp2: Keypoints from two images.
        matches: List of matches.
        mask: Optional mask for inlier matches.
        title: Optional title for the plot.
        max_matches: Maximum number of matches to display.
        figsize: Figure size as (width, height).
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display images
    ax1.imshow(img1)
    ax2.imshow(img2)
    
    # Disable axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Select subset of matches to display
    if mask is not None:
        inlier_indices = [i for i, val in enumerate(mask) if val]
        if len(inlier_indices) > max_matches:
            inlier_indices = inlier_indices[:max_matches]
        display_matches = [matches[i] for i in inlier_indices]
    else:
        if len(matches) > max_matches:
            display_matches = matches[:max_matches]
        else:
            display_matches = matches
    
    # Draw matches
    for match in display_matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        # Draw points
        ax1.plot(pt1[0], pt1[1], 'ro', markersize=4)
        ax2.plot(pt2[0], pt2[1], 'ro', markersize=4)
        
        # Draw connecting line
        con = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                             axesA=ax2, axesB=ax1, color="yellow", linewidth=0.5)
        ax2.add_artist(con)
    
    # Set title
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()

def plot_epipolar_lines(img1, img2, pts1, pts2, F, title=None, figsize=(15, 8)):
    """
    Plot epipolar lines on a pair of images.
    
    Args:
        img1, img2: Two input images.
        pts1, pts2: Matched points in the two images.
        F: Fundamental matrix.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).
    """
    def draw_epilines(img, pts, epi_lines, color=(0, 255, 0)):
        """Draw epipolar lines on an image."""
        h, w = img.shape[:2]
        result = img.copy()
        
        for line, pt in zip(epi_lines, pts):
            a, b, c = line
            
            # Draw line
            if abs(b) < 1e-9:  # Avoid division by zero
                continue
                
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(a * w + c) / b)
            
            cv2.line(result, (x0, y0), (x1, y1), color, 1)
            
            # Draw point
            cv2.circle(result, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        return result
    
    # Sample a subset of points
    n_points = min(len(pts1), 20)  # Limit to 20 points
    indices = np.random.choice(len(pts1), n_points, replace=False)
    
    pts1_subset = pts1[indices]
    pts2_subset = pts2[indices]
    
    # Calculate epipolar lines in both images
    lines1 = cv2.computeCorrespondEpilines(pts2_subset.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines = draw_epilines(img1, pts1_subset, lines1, color=(0, 255, 0))
    
    lines2 = cv2.computeCorrespondEpilines(pts1_subset.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines = draw_epilines(img2, pts2_subset, lines2, color=(0, 255, 0))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display images with epipolar lines
    ax1.imshow(img1_with_lines)
    ax2.imshow(img2_with_lines)
    
    # Set titles
    ax1.set_title('Image 1 with epipolar lines')
    ax2.set_title('Image 2 with epipolar lines')
    
    # Disable axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Set main title
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(img, keypoints, title=None, figsize=(10, 8), bin_size=30):
    """
    Plot spatial distribution of features.
    
    Args:
        img: Input image.
        keypoints: List of keypoints.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).
        bin_size: Size of histogram bins.
    """
    # Extract keypoint positions
    kp_positions = np.array([kp.pt for kp in keypoints])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display image with keypoints
    ax1.imshow(img)
    ax1.scatter(kp_positions[:, 0], kp_positions[:, 1], s=10, c='r', alpha=0.5)
    ax1.set_title('Feature Locations')
    ax1.axis('off')
    
    # Create histogram
    h, w = img.shape[:2]
    x_bins = np.arange(0, w, bin_size)
    y_bins = np.arange(0, h, bin_size)
    
    hist, _, _ = np.histogram2d(kp_positions[:, 1], kp_positions[:, 0], bins=[y_bins, x_bins])
    
    # Display histogram
    im = ax2.imshow(hist, interpolation='nearest', origin='lower',
                  extent=[0, w, 0, h], cmap='jet')
    ax2.set_title('Feature Density')
    fig.colorbar(im, ax=ax2, label='Number of features')
    
    # Set main title
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()

def plot_feature_matching_analysis(matches_dict, figsize=(12, 8)):
    """
    Plot analysis of feature matching across image pairs.
    
    Args:
        matches_dict: Dictionary mapping image pairs to matches.
        figsize: Figure size as (width, height).
    """
    # Extract data
    pairs = list(matches_dict.keys())
    num_matches = [len(matches_dict[pair][2]) for pair in pairs]
    
    # Label pairs
    pair_labels = [f"{pair[0]}-{pair[1]}" for pair in pairs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    ax.bar(range(len(pairs)), num_matches)
    
    # Set labels
    ax.set_xlabel('Image Pair')
    ax.set_ylabel('Number of Matches')
    ax.set_title('Feature Matches Across Image Pairs')
    
    # Set x-tick labels
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, rotation=90)
    
    # Add numbers on top of bars
    for i, v in enumerate(num_matches):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()