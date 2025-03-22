import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path

def load_image_sequence(directory, pattern="viff.*.png", grayscale=False):
    """
    Load an image sequence from a directory.
    
    Args:
        directory: Path to the directory containing images.
        pattern: Glob pattern to match image files.
        grayscale: Whether to load images in grayscale.
        
    Returns:
        A list of loaded images and their filenames.
    """
    image_paths = sorted(glob(os.path.join(directory, pattern)))
    images = []
    
    for path in image_paths:
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
            
        images.append((img, os.path.basename(path)))
    
    return images

def load_calibrated_image_set(directory):
    """
    Load images with calibration information if available.
    
    Args:
        directory: Path to the directory containing images and calibration.
        
    Returns:
        Tuple of (images, camera_intrinsics) where images is a list of (image, filename)
        and camera_intrinsics is the camera matrix K or None if not available.
    """
    # Load images
    images = load_image_sequence(directory)
    
    # Check for calibration file
    calib_path = os.path.join(directory, "calibration.txt")
    K = None
    
    if os.path.exists(calib_path):
        try:
            with open(calib_path, 'r') as f:
                lines = f.readlines()
                # Parse focal length and principal point
                fx = float(lines[0].strip())
                fy = float(lines[1].strip())
                cx = float(lines[2].strip())
                cy = float(lines[3].strip())
                
                # Create camera matrix
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                print(f"Loaded calibration from {calib_path}")
        except Exception as e:
            print(f"Error loading calibration: {e}")
    
    return images, K

def resize_images(images, max_dimension=1000):
    """
    Resize images while maintaining aspect ratio.
    
    Args:
        images: List of (image, filename) tuples.
        max_dimension: Maximum dimension (width or height).
        
    Returns:
        List of (resized_image, filename) tuples.
    """
    resized_images = []
    
    for img, filename in images:
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_dimension / w, max_dimension / h)
        
        if scale < 1:  # Only resize if the image is larger than max_dimension
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized_images.append((resized, filename))
        else:
            resized_images.append((img, filename))
    
    return resized_images

def check_image_quality(images, blur_threshold=100):
    """
    Check image quality (blur detection, exposure).
    
    Args:
        images: List of (image, filename) tuples.
        blur_threshold: Threshold for Laplacian variance (lower values indicate blur).
        
    Returns:
        Dictionary mapping filenames to quality metrics.
    """
    quality_metrics = {}
    
    for img, filename in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Detect blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        # Check exposure (mean brightness)
        mean_brightness = np.mean(gray)
        over_exposed = mean_brightness > 220
        under_exposed = mean_brightness < 35
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Store metrics
        quality_metrics[filename] = {
            'blur_score': blur_score,
            'is_blurry': blur_score < blur_threshold,
            'mean_brightness': mean_brightness,
            'over_exposed': over_exposed,
            'under_exposed': under_exposed,
            'contrast': contrast
        }
    
    return quality_metrics

def create_image_masks(images, method='threshold', threshold=127, kernel_size=5):
    """
    Create binary masks for images to segment objects from background.
    
    Args:
        images: List of (image, filename) tuples.
        method: 'threshold', 'adaptive', or 'otsu'.
        threshold: Threshold value for basic thresholding.
        kernel_size: Size of morphological kernel.
        
    Returns:
        List of (image, mask, filename) tuples.
    """
    masked_images = []
    
    for img, filename in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        if method == 'threshold':
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'adaptive':
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(f"Unsupported masking method: {method}")
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        masked_images.append((img, mask, filename))
    
    return masked_images

def load_images_with_mask(directory, background_color=(0, 0, 0)):
    """
    Load images with background masking (assuming images with black or other specified background).
    
    Args:
        directory: Path to the directory containing images.
        background_color: Color to be treated as background (RGB).
        
    Returns:
        List of (image, mask, filename) tuples.
    """
    images = load_image_sequence(directory)
    masked_images = []
    
    for img, filename in images:
        # Create mask by finding pixels that don't match background color
        if len(img.shape) == 3:
            # Convert background color to numpy array
            bg_color = np.array(background_color, dtype=np.uint8)
            
            # Create mask where pixels differ from background color
            diff = np.abs(img - bg_color).sum(axis=2)
            mask = (diff > 30).astype(np.uint8) * 255
        else:
            # For grayscale images
            mask = (img > background_color[0] + 30).astype(np.uint8) * 255
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        masked_images.append((img, mask, filename))
    
    return masked_images