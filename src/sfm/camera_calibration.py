import numpy as np
import cv2

def estimate_camera_matrix(image_shape, focal_length=None):
    """
    Estimate camera intrinsic matrix.
    
    Args:
        image_shape: (height, width) of the image.
        focal_length: Focal length in pixels. If None, estimate from image width.
        
    Returns:
        3x3 camera intrinsic matrix.
    """
    height, width = image_shape[:2]
    
    if focal_length is None:
        # Estimate focal length based on image width (common heuristic)
        focal_length = 1.2 * max(width, height)
    
    # Principal point (usually at image center)
    cx, cy = width / 2, height / 2
    
    # Create camera matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    return K

def calibrate_camera_from_chessboard(images, pattern_size=(9, 6), square_size=1.0):
    """
    Calibrate camera using a chessboard pattern.
    
    Args:
        images: List of (image, filename) tuples.
        pattern_size: Number of inner corners in the chessboard (width, height).
        square_size: Size of chessboard squares in your chosen units.
        
    Returns:
        Camera matrix, distortion coefficients, rotation vectors, translation vectors.
    """
    # Prepare object points (3D points in real world space)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    success_count = 0
    
    for img, filename in images:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add points
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            success_count += 1
            print(f"Found corners in {filename}")
        else:
            print(f"Could not find corners in {filename}")
    
    if success_count < 3:
        print("Warning: Too few successful calibration images.")
        return None, None, None, None
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    print(f"Calibration successful. Average reprojection error: {total_error/len(objpoints)}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def estimate_focal_length_from_exif(image_path):
    """
    Estimate focal length from EXIF data.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Estimated focal length in pixels or None if EXIF data is not available.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        # Open image
        img = Image.open(image_path)
        
        # Extract EXIF data
        exif_data = img._getexif()
        if exif_data is None:
            return None
        
        # Map EXIF tags to readable names
        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
        
        # Check if focal length and sensor information are available
        if 'FocalLength' in exif and 'FocalLengthIn35mmFilm' in exif:
            # Get focal length in mm
            focal_length_mm = float(exif['FocalLength'][0]) / float(exif['FocalLength'][1])
            
            # Get focal length in 35mm equivalent
            focal_35mm = float(exif['FocalLengthIn35mmFilm'])
            
            # Estimate crop factor
            crop_factor = focal_35mm / focal_length_mm
            
            # Estimate sensor width (35mm film is 36mm wide)
            sensor_width_mm = 36.0 / crop_factor
            
            # Get image width in pixels
            width, _ = img.size
            
            # Estimate focal length in pixels
            focal_length_pixels = (focal_length_mm / sensor_width_mm) * width
            
            return focal_length_pixels
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
    
    return None

def validate_camera_matrix(K, image_shape):
    """
    Validate camera matrix by checking if principal point is within image bounds
    and focal length is reasonable.
    
    Args:
        K: Camera intrinsic matrix.
        image_shape: (height, width) of the image.
        
    Returns:
        True if camera matrix is valid, False otherwise.
    """
    height, width = image_shape[:2]
    
    # Extract parameters
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    
    # Check if principal point is within image
    if cx < 0 or cx > width or cy < 0 or cy > height:
        print("Warning: Principal point outside image bounds.")
        return False
    
    # Check if focal length is reasonable (usually between 0.5 and 3 times the image dimension)
    max_dim = max(width, height)
    if fx < 0.5 * max_dim or fx > 3 * max_dim or fy < 0.5 * max_dim or fy > 3 * max_dim:
        print("Warning: Focal length seems unreasonable.")
        return False
    
    # Check if fx and fy are similar (most cameras have square pixels)
    if abs(fx - fy) > 0.1 * max(fx, fy):
        print("Warning: Significant difference between horizontal and vertical focal lengths.")
        # Still valid, but suspicious
    
    return True