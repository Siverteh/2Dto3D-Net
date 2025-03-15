import os
import cv2
import numpy as np
from glob import glob

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

def enhance_contrast(images):
    """
    Enhance image contrast using CLAHE.
    
    Args:
        images: List of (image, filename) tuples.
        
    Returns:
        List of (enhanced_image, filename) tuples.
    """
    enhanced_images = []
    
    for img, filename in images:
        if len(img.shape) == 3:  # Color image
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
        
        enhanced_images.append((enhanced, filename))
    
    return enhanced_images