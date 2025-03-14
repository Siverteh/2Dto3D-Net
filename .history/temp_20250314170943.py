import os
from PIL import Image

input_dir = "./extracted_images"  # Path where .ppm files were extracted
output_dir = "./converted_images"
os.makedirs(output_dir, exist_ok=True)

# Loop over files in input_dir
for filename in os.listdir(input_dir):
    if filename.endswith(".ppm"):
        ppm_path = os.path.join(input_dir, filename)
        png_path = os.path.join(output_dir, filename.replace(".ppm", ".png"))
        
        # Load and save as PNG
        with Image.open(ppm_path) as img:
            img.convert("RGB").save(png_path)
        print(f"Converted {filename} to PNG.")

print("All .ppm files converted to .png.")
