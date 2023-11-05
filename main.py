"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

# Standard Library Imports
import os
import time
import json
from typing import List, Tuple

# Third-party Library Imports
import cv2
import numpy as np
import pandas as pd

# Local Module Imports
import image_processing
import nanoparticles_counting

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Config variables
RAW_IMAGES_DIRECTORY = config['RAW_IMAGES_DIRECTORY']
PROCESSED_IMAGES_DIRECTORY = config['PROCESSED_IMAGES_DIRECTORY']
MASKED_IMAGES_DIRECTORY = config['MASKED_IMAGES_DIRECTORY']
SEGMENTED_IMAGES_DIRECTORY = config['SEGMENTED_IMAGES_DIRECTORY']
RESULTS_DIRECTORY = config['RESULTS_DIRECTORY']
TEMPORAL_AVERAGE_WINDOW_SIZE = config['TEMPORAL_AVERAGE_WINDOW_SIZE']
MEDIAN_FILTER_KERNEL_SIZE = config['MEDIAN_FILTER_KERNEL_SIZE']
CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']
ADAPTIVE_THRESHOLDING_BLOCK_SIZE = config['ADAPTIVE_THRESHOLDING_BLOCK_SIZE']
ADAPTIVE_THRESHOLDING_CONSTANT = config['ADAPTIVE_THRESHOLDING_CONSTANT']
ROI_RADIUS = config['ROI_RADIUS']


def load_grayscale_images(directory_path: str) -> Tuple[List[np.ndarray], List[str]]:
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


def save_results_to_excel(data: List[Tuple[str, int]], output_folder: str) -> None:
    df_counts = pd.DataFrame(data, columns=['Filename', 'White Pixel Count'])
    excel_file_path = os.path.join(output_folder, 'white_pixel_counts.xlsx')
    try:
        df_counts.to_excel(excel_file_path, index=False)
    except Exception as e:
        print(f"Failed to save to Excel: {e}")


def main() -> None:
    """
    Main execution function.

    This function orchestrates the loading of images, their processing, and the subsequent display
    of results. Execution time for each major step is printed to the console.
    """
    try:
        # Load images
        start_time = time.time()
        raw_images, filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")

        # Process images
        #processed_images = image_processing.process_images(raw_images, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, 
        #                                                   CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

        # Segment Au-NPS
        segmented_images = nanoparticles_counting.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, 
                                                                                   ADAPTIVE_THRESHOLDING_BLOCK_SIZE, ADAPTIVE_THRESHOLDING_CONSTANT)
        masked_images = nanoparticles_counting.apply_masking_roi(segmented_images, raw_images, filenames, MASKED_IMAGES_DIRECTORY, ROI_RADIUS)
        
        # Count white pixels and store results in a DataFrame
        counts = [(filename, nanoparticles_counting.count_white_pixels(img)) for img, filename in masked_images]
        save_results_to_excel(counts, RESULTS_DIRECTORY)
        print(f"White pixel counts have been written to Excel.")

    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
