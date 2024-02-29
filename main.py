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
    import shutil

    # Third-party Library Imports
    import cv2
    import numpy as np
    import pandas as pd

    # Local Module Imports
    import image_processing as ip
    import nanoparticles_counting as np_count
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Load the configuration file
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    raise FileNotFoundError("Config file 'config.json' not found.")


# Config variables
MASKS_DIRECTORY = config['MASKS_DIRECTORY']
MASKS_BG_DIRECTORY = config['MASKS_BG_DIRECTORY']
RAW_IMAGES_DIRECTORY = config['RAW_IMAGES_DIRECTORY']
PROCESSED_IMAGES_DIRECTORY = config['PROCESSED_IMAGES_DIRECTORY']
MASKED_IMAGES_DIRECTORY = config['MASKED_IMAGES_DIRECTORY']
SEGMENTED_IMAGES_DIRECTORY = config['SEGMENTED_IMAGES_DIRECTORY']
RESULTS_DIRECTORY = config['RESULTS_DIRECTORY']

TEMPORAL_AVERAGE_WINDOW_SIZE = config['TEMPORAL_AVERAGE_WINDOW_SIZE']
MEDIAN_FILTER_KERNEL_SIZE = config['MEDIAN_FILTER_KERNEL_SIZE']
GAUSSIAN_FILTER_KERNEL_SIZE = config['GAUSSIAN_FILTER_KERNEL_SIZE']
GAUSSIAN_FILTER_SIGMA = config['GAUSSIAN_FILTER_SIGMA']
KSPACE_FILTER_CUTOFF_FREQ = config['KSPACE_FILTER_CUTOFF_FREQ']
CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']

THRESHOLD_METHOD = config['THRESHOLDING_METHOD']
ENABLE_FINE_TUNING = config["ENABLE_FINE_TUNING"]

ROI_RADIUS = config['ROI_RADIUS']
EXCEL_FILE_NAME = config['EXCEL_FILE_NAME']
ALLOWED_IMAGE_EXTENSIONS = tuple(config['ALLOWED_IMAGE_EXTENSIONS'])



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
    """
    Save the provided data as an Excel file in the specified output folder.

    Args:
        data (List[Tuple[str, int]]): A list of tuples, each containing a filename and an associated white pixel count.
        output_folder (str): The path to the directory where the Excel file will be saved.

    Raises:
        FileNotFoundError: If the specified output directory does not exist.
        Exception: For any other issues that occur during the saving process.
    """
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"The specified output directory does not exist: {output_folder}")

    df_counts = pd.DataFrame(data, columns=['Filename', 'White Pixel Count'])
    excel_file_path = os.path.join(output_folder, EXCEL_FILE_NAME)

    try:
        df_counts.to_excel(excel_file_path, index=False)
    except Exception as e:
        print(f"Failed to save to Excel: {e}")
        raise


def clean_directory(folder_path):
    """
    Deletes all files and folders in the specified directory.

    Args:
        folder_path (str): Path to the directory to clean.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def fine_tune_parameters(raw_images: List[np.ndarray], filenames: List[str], config: dict, mask: np.ndarray) -> dict:
    """
    Fine-tune parameters for image segmentation based on the thresholding method.

    Args:
        raw_images (List[np.ndarray]): List of raw images.
        filenames (List[str]): List of filenames corresponding to the images.
        config (dict): Configuration settings.
        mask (np.ndarray): Mask used for image segmentation.

    Returns:
        dict: A dictionary containing the results of fine-tuning for each set of parameters.

    Raises:
        Exception: If an error occurs during the segmentation process.
    """
    results = {}
    try:
        if THRESHOLD_METHOD == 'adaptive':
            block_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] 
            constants = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

            for block_size in block_sizes:
                for constant in constants:
                    # NEED TO MODIFY 'APPLY_NANOPARTICLES_SEGMENTATION' DEFINITION
                    segmented_images = np_count.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, config, block_size, constant)

                    white_pixel_counts = []
                    for i, seg_image in enumerate(segmented_images):
                        masked_img = cv2.bitwise_and(seg_image, seg_image, mask)

                        masked_filename = f"masked_{filenames[i]}"
                        cv2.imwrite(os.path.join(MASKED_IMAGES_DIRECTORY, masked_filename), masked_img)
                        count = np_count.count_white_pixels(masked_img)
                        white_pixel_counts.append(count)

                    results[f'adaptive_{block_size}_{constant}'] = white_pixel_counts

        elif THRESHOLD_METHOD == 'fixed':
            threshold_values = [100, 115, 125, 135, 145, 155, 165, 175, 185, 195, 210]

            for threshold_value in threshold_values:

                # Apply segmentation with the current threshold value
                segmented_images = np_count.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, config, threshold_value)
                
                # Apply masks using the apply_masking function
                masked_images = np_count.apply_masking(segmented_images, mask, filenames, MASKED_IMAGES_DIRECTORY)
                
                white_pixel_counts = []
                for img in masked_images:
                    count = np_count.count_white_pixels(img)
                    white_pixel_counts.append(count)

                results[f'fixed_thresh{threshold_value}'] = white_pixel_counts
                
    except Exception as e:
        raise Exception(f"Error during fine-tuning: {e}")

    return results


def main() -> None:
    """
    Main execution function.

    This function orchestrates the loading of images, their processing, and the subsequent display
    of results. Execution time for each major step is printed to the console.
    """
    try:
        # Clean directories
        clean_directory(PROCESSED_IMAGES_DIRECTORY)
        clean_directory(MASKED_IMAGES_DIRECTORY)
        clean_directory(SEGMENTED_IMAGES_DIRECTORY)
        
        # Load images
        start_time = time.time()
        raw_images, filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")
    
        """
        # 1. Raw images processing
        start_time = time.time()
        processed_images = ip.process_images(raw_images, filenames, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, GAUSSIAN_FILTER_KERNEL_SIZE, 
                                                           GAUSSIAN_FILTER_SIGMA, KSPACE_FILTER_CUTOFF_FREQ, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, PROCESSED_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for processing images: {end_time - start_time:.4f} seconds")
        """

        # 2. Nanoparticle identification
        start_time = time.time()
        # Load or create signal masks
        mask = np_count.load_or_create_single_mask(filenames, MASKS_DIRECTORY, RAW_IMAGES_DIRECTORY, ROI_RADIUS)
        print('Masks loaded/created')

        if ENABLE_FINE_TUNING:
            fine_tuning_results = fine_tune_parameters(raw_images, filenames, config, mask)
            # Convert the results to a DataFrame
            df_counts = pd.DataFrame(fine_tuning_results)
            df_counts.insert(0, 'Filename', filenames)

            # Save to Excel file
            excel_file_path = os.path.join(RESULTS_DIRECTORY, 'Finetuning_Results.xlsx')
            df_counts.to_excel(excel_file_path, index=False)
            print(f"Fine tuning completed. Results are available in {RESULTS_DIRECTORY}")
            end_time = time.time()
            print(f"Execution time for fine tuning: {end_time - start_time:.4f} seconds")

        else:
            # Segment images
            segmented_images = np_count.apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY, config)
            print('Features segmented')
            
            # Apply masks
            masked_images = np_count.apply_masking(segmented_images, mask, filenames, MASKED_IMAGES_DIRECTORY)
            print('Masking Applied')
            end_time = time.time()
            print(f"Execution time for identifying nanoparticles: {end_time - start_time:.4f} seconds")

            # 3. Nanoparticle quantification
            start_time = time.time()
            counts = [(filename, np_count.count_white_pixels(img)) for img, filename in zip(masked_images, filenames)]
            save_results_to_excel(counts, RESULTS_DIRECTORY)
            print(f"Analysis completed. Results are available in {RESULTS_DIRECTORY}")
            end_time = time.time()
            print(f"Execution time for quantifying nanoparticles: {end_time - start_time:.4f} seconds")
        

    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
