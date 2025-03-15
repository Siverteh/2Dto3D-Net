import cv2
import numpy as np

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

def denoise_images(images, method='gaussian', params=None):
    """
    Apply denoising to images.
    
    Args:
        images: List of (image, filename) tuples.
        method: 'gaussian', 'median', 'bilateral', or 'nlmeans'.
        params: Dictionary of parameters for the denoise method.
        
    Returns:
        List of (denoised_image, filename) tuples.
    """
    if params is None:
        params = {}
    
    denoised_images = []
    
    for img, filename in images:
        if method == 'gaussian':
            ksize = params.get('ksize', 5)
            sigma = params.get('sigma', 0)
            denoised = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
        elif method == 'median':
            ksize = params.get('ksize', 5)
            denoised = cv2.medianBlur(img, ksize)
        
        elif method == 'bilateral':
            d = params.get('d', 9)
            sigma_color = params.get('sigma_color', 75)
            sigma_space = params.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        elif method == 'nlmeans':
            h = params.get('h', 10)
            template_window_size = params.get('template_window_size', 7)
            search_window_size = params.get('search_window_size', 21)
            
            if len(img.shape) == 3:  # Color image
                denoised = cv2.fastNlMeansDenoisingColored(
                    img, None, h, h, template_window_size, search_window_size)
            else:  # Grayscale image
                denoised = cv2.fastNlMeansDenoising(
                    img, None, h, template_window_size, search_window_size)
        
        else:
            raise ValueError(f"Unsupported denoising method: {method}")
        
        denoised_images.append((denoised, filename))
    
    return denoised_images

def equalize_illumination(images):
    """
    Equalize illumination across images for more consistent feature matching.
    
    Args:
        images: List of (image, filename) tuples.
        
    Returns:
        List of (equalized_image, filename) tuples.
    """
    equalized_images = []
    
    # Compute average brightness across all images
    avg_brightness = 0
    for img, _ in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        avg_brightness += np.mean(gray)
    
    avg_brightness /= len(images)
    
    # Adjust brightness of each image
    for img, filename in images:
        if len(img.shape) == 3:
            # Convert to HSV for brightness adjustment
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Compute current brightness
            current_brightness = np.mean(v)
            
            # Calculate adjustment factor
            adjustment = avg_brightness / current_brightness if current_brightness > 0 else 1
            
            # Adjust V channel
            v_adjusted = np.clip(v * adjustment, 0, 255).astype(np.uint8)
            
            # Merge channels and convert back to RGB
            hsv_adjusted = cv2.merge((h, s, v_adjusted))
            equalized = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale images
            current_brightness = np.mean(img)
            adjustment = avg_brightness / current_brightness if current_brightness > 0 else 1
            equalized = np.clip(img * adjustment, 0, 255).astype(np.uint8)
        
        equalized_images.append((equalized, filename))
    
    return equalized_images

def sharpen_images(images, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    """
    Sharpen images using unsharp masking.
    
    Args:
        images: List of (image, filename) tuples.
        kernel_size: Size of Gaussian blur kernel.
        sigma: Standard deviation for Gaussian blur.
        amount: Strength of sharpening.
        threshold: Minimum brightness difference to apply sharpening.
        
    Returns:
        List of (sharpened_image, filename) tuples.
    """
    sharpened_images = []
    
    for img, filename in images:
        # Create blurred version of the image
        blurred = cv2.GaussianBlur(img.astype(np.float32), (kernel_size, kernel_size), sigma)
        
        # Calculate unsharp mask
        sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
        
        # Apply threshold to reduce noise amplification
        if threshold > 0:
            low_contrast_mask = np.abs(img.astype(np.float32) - blurred) < threshold
            sharpened[low_contrast_mask] = img.astype(np.float32)[low_contrast_mask]
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        sharpened_images.append((sharpened, filename))
    
    return sharpened_images

def apply_gamma_correction(images, gamma=1.0):
    """
    Apply gamma correction to adjust image brightness.
    
    Args:
        images: List of (image, filename) tuples.
        gamma: Gamma value (< 1 brightens, > 1 darkens).
        
    Returns:
        List of (corrected_image, filename) tuples.
    """
    corrected_images = []
    
    # Build a lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    for img, filename in images:
        if len(img.shape) == 3:  # Color image
            # Apply gamma correction to each channel
            corrected = cv2.LUT(img, table)
        else:  # Grayscale image
            corrected = cv2.LUT(img, table)
        
        corrected_images.append((corrected, filename))
    
    return corrected_images

def normalize_images(images):
    """
    Normalize images to have consistent brightness and contrast.
    
    Args:
        images: List of (image, filename) tuples.
        
    Returns:
        List of (normalized_image, filename) tuples.
    """
    normalized_images = []
    
    for img, filename in images:
        if len(img.shape) == 3:  # Color image
            # Convert to YUV color space
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)
            
            # Normalize Y channel (luminance)
            y_norm = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
            
            # Merge channels and convert back to RGB
            yuv_norm = cv2.merge((y_norm, u, v))
            normalized = cv2.cvtColor(yuv_norm, cv2.COLOR_YUV2RGB)
        else:  # Grayscale image
            normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        normalized_images.append((normalized, filename))
    
    return normalized_images

def undistort_images(images, camera_matrix, dist_coeffs):
    """
    Undistort images using camera calibration parameters.
    
    Args:
        images: List of (image, filename) tuples.
        camera_matrix: Camera intrinsic matrix K.
        dist_coeffs: Distortion coefficients.
        
    Returns:
        List of (undistorted_image, filename) tuples.
    """
    undistorted_images = []
    
    for img, filename in images:
        h, w = img.shape[:2]
        
        # Calculate optimal camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        
        # Undistort image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        undistorted_images.append((undistorted, filename))
    
    return undistorted_images

def extract_foreground(images, method='grabcut'):
    """
    Extract foreground object from images.
    
    Args:
        images: List of (image, filename) tuples.
        method: 'grabcut' or 'threshold'.
        
    Returns:
        List of (image, mask, filename) tuples where mask is the foreground mask.
    """
    foreground_results = []
    
    for img, filename in images:
        if len(img.shape) != 3:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if method == 'grabcut':
            # Initialize mask for GrabCut
            mask = np.zeros(img.shape[:2], np.uint8)
            
            # Rectangle containing the object
            h, w = img.shape[:2]
            rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
            
            # Temporary arrays for GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create foreground mask
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255
            
        elif method == 'threshold':
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Apply Otsu's thresholding
            _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
        else:
            raise ValueError(f"Unsupported foreground extraction method: {method}")
        
        foreground_results.append((img, fg_mask, filename))
    
    return foreground_results