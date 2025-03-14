import tarfile
import os

file_name = "images.tar.gz"
extract_dir = "./extracted_images"  # Change this if you want a different directory

# Check if file exists
if os.path.exists(file_name):
    print(f"Extracting '{file_name}' to '{extract_dir}'...")

    # Create output directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Open and extract the tar.gz file
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print("Extraction complete.")
else:
    print(f"File '{file_name}' does not exist!")
