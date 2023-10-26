"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

# Standard Library Imports
import os
import time

# Third-party Library Imports
import cv2

# Local Module Imports
import image_processing


# Configuration Constants
RAW_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\raw_images'
TEMPORAL_AVERAGE_WINDOW_SIZE = 10
MEDIAN_FILTER_KERNEL_SIZE = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def load_grayscale_images(directory_path):
    """
    Load images from the specified directory and convert them to grayscale.

    Args:
        directory_path (str): The file path of the directory containing the images.

    Returns:
        tuple: A tuple containing two elements:
            - A list of grayscale images.
            - A list of filenames corresponding to the loaded images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no images are found in the directory.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The specified directory does not exist: {directory_path}")

    image_filenames = [filename for filename in os.listdir(directory_path) if filename.endswith(('.png', '.JPG', '.jpeg'))]
    if not image_filenames:
        raise ValueError("No images found in the specified directory.")

    print(f"Attempting to load {len(image_filenames)} images...")
    grayscale_images = [cv2.imread(os.path.join(directory_path, filename), cv2.IMREAD_GRAYSCALE) for filename in image_filenames]

    if not grayscale_images:
        raise ValueError("One or more images could not be loaded. Please check the file paths and formats.")

    return grayscale_images, image_filenames


def main():
    """
    Main execution function.

    This function orchestrates the loading of images, their processing, and the subsequent display
    of results. Execution time for each major step is printed to the console.
    """
    try:
        start_time = time.time()
        raw_images, image_filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")

        processed_images = image_processing.process_images(raw_images, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
