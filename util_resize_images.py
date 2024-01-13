"""
Created on Tue Nov 21 16:59:45 2023
@author: Mateo HAMEL
"""

try:
    import os
    import cv2

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Define your image directory and output directory
image_directory = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\Raw_Images_tokeep"
output_directory = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\Raw_Images_tokeep\\Cropped"

# Check if output directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the coordinates of the rectangle you want to crop
# (x, y, width, height)
x = 20
y = 0
width = 800
height = 580
crop_rectangle = (x, y, width, height)

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image file
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        # Crop the image using the coordinates defined
        cropped_image = image[y:y+height, x:x+width]

        # Save the cropped image to the output directory
        cropped_image_path = os.path.join(output_directory, image_file)
        cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print(f"Error loading image {image_file}")
