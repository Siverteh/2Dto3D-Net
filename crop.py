#!/usr/bin/env python3
import os
from PIL import Image
import glob

# Define directories
input_dir = "data/dinosaur"
output_dir = "data/dinosaur_cropped"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Amount to crop from edges in pixels
# Adjust these values if needed based on where the artifacts appear
CROP_TOP = 3
CROP_RIGHT = 25

# Find all image files in input directory
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))

print(f"Found {len(image_files)} images to process")

# Process each image
for img_path in image_files:
    try:
        # Get filename without directory
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Open image
        img = Image.open(img_path)
        
        # Get dimensions
        width, height = img.size
        
        # Crop image (left, top, right, bottom)
        cropped_img = img.crop((0, CROP_TOP, width - CROP_RIGHT, height))
        
        # Save to output directory
        output_path = os.path.join(output_dir, filename)
        cropped_img.save(output_path)
        
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error processing {img_path}: {e}")

print("Cropping complete!")