"""
Created on Wed Oct 25 17:13:01 2023

@author: Mateo HAMEL
"""

try:
    # Standard Library Imports
    import os
    import json
    from typing import List, Tuple

    # Third-party Library Imports
    import cv2
    import numpy as np

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")



def compute_temporal_average(images: List[np.ndarray], window_size: int) -> List[np.ndarray]:
    """
    Compute the temporal average of a list of images over a specified window size.
    
    Args:
        images (List[np.ndarray]): List of images to average.
        window_size (int): Number of images to average over.

    Returns:
        List[np.ndarray]: List of temporally averaged images.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if window_size <= 0:
        raise ValueError("Window size should be a positive integer.")
    if len(images) < window_size:
        raise ValueError(f"The number of images should be at least {window_size}")
    
    number_chunk = len(images) // window_size
    array_imgs = np.array(images)

    imgs_average = []

    for i in range(number_chunk):
        sum_img = np.zeros(images[0].shape, dtype=np.float16)
        for j in range(window_size):
            sum_img += array_imgs[i * window_size + j]
        imgs_average.append(np.divide(sum_img, window_size).astype(np.uint8))

    for idx, avg_img in enumerate(imgs_average):
        avg_img_path = os.path.join(image_directory, f"average_{idx}.png")
        cv2.imwrite(avg_img_path, avg_img)

    return imgs_average



# Specify the directory containing the images and window size
image_directory = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\00_images_background" # Replace with your actual directory path
window_size = 4

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
images = []

# Process each image file
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)
    images.append(image)

# Call the function to compute the temporal average
compute_temporal_average(images, window_size)


