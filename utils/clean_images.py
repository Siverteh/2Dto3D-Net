import os
import cv2
import numpy as np
from pathlib import Path

def replace_blue_and_gray_with_black(input_dir, output_dir):
    """
    Replaces blue background and gray vertical lines with black in dinosaur images.
    
    Args:
        input_dir: Path to directory with original images
        output_dir: Path to directory for processed images
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        # Read image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read {filename}, skipping...")
            continue
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue pixels
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Handle vertical lines on both left and right sides
        height, width = img.shape[:2]
        border_width = int(width * 0.05)  # 5% of width on each side
        
        # Create masks for both left and right borders
        border_mask = np.zeros((height, width), dtype=np.uint8)
        border_mask[:, :border_width] = 255  # Left border
        border_mask[:, width-border_width:] = 255  # Right border
        
        # Convert to grayscale for gray detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create a mask for gray-ish pixels
        _, gray_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        gray_mask = cv2.bitwise_not(gray_thresh)
        
        # Combine with position to focus on border areas only
        gray_line_mask = cv2.bitwise_and(gray_mask, border_mask)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, gray_line_mask)
        
        # Additional step: remove black background around dinosaur (for the right image)
        # This helps with images that have already been partially processed
        # and have the dinosaur on a pure black background
        _, black_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        black_mask = cv2.bitwise_not(black_mask)
        
        # Combine with existing mask
        combined_mask = cv2.bitwise_or(combined_mask, black_mask)
        
        # Dilate the mask to ensure complete coverage
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Create output image with black background
        result = img.copy()
        result[combined_mask > 0] = [0, 0, 0]  # Set masked pixels to black
        
        # Save the result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result)
        print(f"Processed {filename}")
    
    print(f"Completed processing {len(image_files)} images")

if __name__ == "__main__":
    input_directory = "data/dinosaur_cropped"
    output_directory = "data/dinosaur_cropped_black"
    
    replace_blue_and_gray_with_black(input_directory, output_directory)