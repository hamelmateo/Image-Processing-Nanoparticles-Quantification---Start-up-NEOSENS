"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

try:
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
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Config variables
MASKS_DIRECTORY = config['MASKS_DIRECTORY']
RAW_IMAGES_DIRECTORY = config['RAW_IMAGES_DIRECTORY']
PROCESSED_IMAGES_DIRECTORY = config['PROCESSED_IMAGES_DIRECTORY']
MASKED_IMAGES_DIRECTORY = config['MASKED_IMAGES_DIRECTORY']
SEGMENTED_IMAGES_DIRECTORY = config['SEGMENTED_IMAGES_DIRECTORY']
RESULTS_DIRECTORY = config['RESULTS_DIRECTORY']
TEMPORAL_AVERAGE_WINDOW_SIZE = config['TEMPORAL_AVERAGE_WINDOW_SIZE']
MEDIAN_FILTER_KERNEL_SIZE = config['MEDIAN_FILTER_KERNEL_SIZE']
CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']

ROI_RADIUS = config['ROI_RADIUS']
EXCEL_FILE_NAME = config['EXCEL_FILE_NAME']
ALLOWED_IMAGE_EXTENSIONS = tuple(config['ALLOWED_IMAGE_EXTENSIONS'])


# Helper Functions
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

    image_filenames = [filename for filename in os.listdir(directory_path) if filename.endswith(ALLOWED_IMAGE_EXTENSIONS)]
    if not image_filenames:
        raise ValueError("No images found in the specified directory.")

    print(f"Attempting to load {len(image_filenames)} images...")
    grayscale_images = [cv2.imread(os.path.join(directory_path, filename), cv2.IMREAD_GRAYSCALE) for filename in image_filenames]

    if not grayscale_images:
        raise ValueError("One or more images could not be loaded. Please check the file paths and formats.")

    return grayscale_images, image_filenames



def save_results_to_excel(data: List[Tuple[str, int]], output_folder: str) -> None:
    df_counts = pd.DataFrame(data, columns=['Filename', 'White Pixel Count'])
    excel_file_path = os.path.join(output_folder, EXCEL_FILE_NAME)
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
        #processed_images = image_processing.process_images(raw_images, filenames, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, 
        #                                                   CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, PROCESSED_IMAGES_DIRECTORY)
        #print('Processing done')

        # Load or create masks
        masks = nanoparticles_counting.load_or_create_masks(filenames, MASKS_DIRECTORY, RAW_IMAGES_DIRECTORY, ROI_RADIUS)
        print('Masks loaded/created')

        """
        # To finetune Methods' parameters
        # Dictionary to hold the counts for different parameters
        results = {}
        
        # For adaptive threshold:
        #block_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]  # Example values for block size
        #constants = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Example values for constant
        #
        #for block_size in block_sizes:
        #    for constant in constants:
            
        # For fixed threshold:
        threshold_values = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 
                            160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250]
        max_values = [255]  # Typically fixed at 255

        for threshold_value in threshold_values:
            for max_value in max_values:
                # Segment images with current block size and constant
                segmented_images = nanoparticles_counting.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, config, threshold_value)

                # Apply masking and count white pixels
                white_pixel_counts = []
                for i, seg_image in enumerate(segmented_images):
                    masked_img = cv2.bitwise_and(seg_image, seg_image, mask=masks[i])

                    # Save the final segmented image
                    masked_filename = f"masked_{filenames[i]}"
                    cv2.imwrite(os.path.join(MASKED_IMAGES_DIRECTORY, masked_filename), masked_img)
                    count = nanoparticles_counting.count_white_pixels(masked_img)
                    white_pixel_counts.append(count)

                # Store the results
                results[f'thresh{threshold_value}'] = white_pixel_counts

        # Convert the results to a DataFrame
        df_counts = pd.DataFrame(results)
        df_counts.insert(0, 'Filename', filenames)

        # Save to Excel file
        excel_file_path = os.path.join(RESULTS_DIRECTORY, 'white_pixel_counts_comparison.xlsx')
        df_counts.to_excel(excel_file_path, index=False)
        print(f"White pixel counts comparison has been written to {excel_file_path}")
        """

        
        # Methods implementation:
        # Segment images
        segmented_images = nanoparticles_counting.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, config)
        print('Segmentation done')

        # Apply masks and count white pixels
        masked_images = nanoparticles_counting.apply_masking(segmented_images, masks, filenames, MASKED_IMAGES_DIRECTORY)
        print('Masking done')

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
