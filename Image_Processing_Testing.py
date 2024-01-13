"""
Created on Wed Oct 25 17:13:01 2023
@author: Mateo HAMEL
"""

try:
    # Standard Library Imports
    import time
    import os
    import json
    from typing import List, Tuple

    # Third-party Library Imports
    import cv2
    import numpy as np
    import pandas as pd

    # Local Module Imports
    import image_metrics as im
    import image_processing as ip
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Config variables
MASKS_DIRECTORY = config['MASKS_DIRECTORY']
MASKS_BG_DIRECTORY = config['MASKS_BG_DIRECTORY']
RAW_IMAGES_DIRECTORY = config['RAW_IMAGES_DIRECTORY']
BACKGROUND_IMAGE_DIRECTORY = config['BACKGROUND_IMAGE_DIRECTORY']
PROCESSED_IMAGES_DIRECTORY = config['PROCESSED_IMAGES_DIRECTORY']
RESULTS_DIRECTORY = config['RESULTS_DIRECTORY']
MEDIAN_FILTER_KERNEL_SIZE = config['MEDIAN_FILTER_KERNEL_SIZE']
CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']

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


def test_clahe_parameters_on_list(images: List[np.ndarray], clip_limits: List[float], tile_grid_sizes: List[Tuple[int, int]]) -> List[dict]:
    """
    Apply CLAHE with different parameters to a list of images and return the processed images.

    Args:
        images (List[np.ndarray]): The list of input images.
        clip_limits (List[float]): List of CLAHE clip limits to test.
        tile_grid_sizes (List[Tuple[int, int]]): List of tile grid sizes to test.

    Returns:
        List[dict]: A list of dictionaries, each containing processed images with parameter settings as keys.

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the clip limits or tile grid sizes lists are empty.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if not clip_limits:
        raise ValueError("Clip limits list cannot be empty.")
    if not tile_grid_sizes:
        raise ValueError("Tile grid sizes list cannot be empty.")
    
    processed_images_list = []

    for clip_limit in clip_limits:
        for tile_grid_size in tile_grid_sizes:
            # Applying CLAHE to all images with the current set of parameters
            processed_images = ip.apply_clahe(images, clip_limit, tile_grid_size)
            # Storing the processed images with their respective parameters
            param_key = (clip_limit, tile_grid_size)
            processed_images_list.append({param_key: processed_images})

    return processed_images_list


def test_gaussian_filter_parameters_on_list(images: List[np.ndarray], kernel_sizes: List[int], sigma_values: List[float]) -> List[dict]:
    """
    Apply Gaussian filter with different parameters to a list of images and return the processed images.

    Args:
        images (List[np.ndarray]): The list of input images.
        kernel_sizes (List[int]): List of Gaussian kernel sizes to test.
        sigma_values (List[float]): List of sigma values to test.

    Returns:
        List[dict]: A list of dictionaries, each containing processed images with parameter settings as keys.

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the kernel sizes or sigma values lists are empty.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if not kernel_sizes:
        raise ValueError("Kernel sizes list cannot be empty.")
    if not sigma_values:
        raise ValueError("Sigma values list cannot be empty.")

    processed_images_list = []

    for kernel_size in kernel_sizes:
        for sigma in sigma_values:
            # Applying Gaussian filter to all images with the current set of parameters
            processed_images = ip.apply_gaussian_filter(images, kernel_size, sigma)
            # Storing the processed images with their respective parameters
            param_key = (kernel_size, sigma)
            processed_images_list.append({param_key: processed_images})

    return processed_images_list


def calculate_metrics(proc_images: List[np.ndarray], raw_images: List[np.ndarray], bg_image: np.ndarray, filenames: List[str], proc_filenames: List[str]) -> pd.DataFrame:
    """
    Calculate metrics for a list of raw and processed images.

    Args:
        raw_imgs (List[np.ndarray]): List of raw images.
        proc_imgs (List[np.ndarray]): List of processed images.
        bg_image (np.ndarray): Image of the background.
        filenames (List[str]): List of filenames corresponding to the raw images.
        proc_filenames (List[str]): List of filenames corresponding to the processed images.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics for each image.

    Raises:
        ValueError: If processed images list, raw images list, filenames list, or processed filenames list is empty.
        ValueError: If background image is not provided.
        FileNotFoundError: If any mask file is not found.
    """    
    if not proc_images or not raw_images:
        raise ValueError("Processed images list and raw images list cannot be empty.")
    if bg_image is None:
        raise ValueError("Background image is required.")
    if not filenames or not proc_filenames:
        raise ValueError("Filenames list and processed filenames list cannot be empty.")

    signal_masks = [cv2.imread(os.path.join(MASKS_DIRECTORY, f'mask_{filename}'), cv2.IMREAD_GRAYSCALE) for filename in filenames]
    bg_mask = cv2.imread(os.path.join(MASKS_BG_DIRECTORY, f'mask_{filenames[0]}'), cv2.IMREAD_GRAYSCALE)

    metrics = []
    for raw_image, filename, signal_mask in zip(raw_images, filenames, signal_masks):
        
        # Metrics for raw images
        raw_metrics = {
            'Filename': filename,
            'SNR': im.calculate_snr(raw_image, raw_images[0], signal_mask, bg_mask), # blank sample (raw_images[0]) is used as background image
            'CNR': im.calculate_cnr(raw_image, raw_images[0], signal_mask, bg_mask),
            'Weber Contrast': im.calculate_weber_contrast(raw_image, raw_images[0], signal_mask, bg_mask),
            'SSIM': 1.0  # SSIM with itself is 1
        }
        metrics.append(raw_metrics)

    batch_size = 9
    for i in range(0, len(proc_images), batch_size):
        batch_proc_images = proc_images[i:i + batch_size]
        batch_proc_filenames = proc_filenames[i:i + batch_size]
        for proc_image, proc_filename, signal_mask in zip(batch_proc_images, batch_proc_filenames, signal_masks):
            # Metrics for processed images
            proc_metrics = {
                'Filename': proc_filename,
                'SNR': im.calculate_snr(proc_image, batch_proc_images[0], signal_mask, bg_mask),
                'CNR': im.calculate_cnr(proc_image, batch_proc_images[0], signal_mask, bg_mask),
                'Weber Contrast': im.calculate_weber_contrast(proc_image, batch_proc_images[0], signal_mask, bg_mask),
                'SSIM': im.calculate_ssim(raw_image, proc_image)
            }
            metrics.append(proc_metrics)

    return pd.DataFrame(metrics)


def main():
    """
    Main function to process a database from the video.
    """
    # Load grayscale images
    raw_images, filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)
    bg_image,_ = load_grayscale_images(BACKGROUND_IMAGE_DIRECTORY)


    # K-space filtering fine tuning
    """
    start_time = time.time()
    cutoff_freqs = list(range(10, 300, 5))

    # Apply K-space filtering
    for freq in cutoff_freqs:
        proc_imgs = apply_kspace_filtering(raw_images, freq)

        # Save the filtered image
        for i, img in enumerate(proc_imgs):
            processed_filename = f"{filenames[i]}_freq_{freq:04d}.png"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    
    end_time = time.time()
    print(f"Execution time for K-space Filtering fine tuning: {end_time - start_time:.4f} seconds")
    """


    # K-space Filtering
    """
    start_time = time.time()
    cutoff_freq = 160
    
    # Apply K-space filter
    proc_images = ip.apply_kspace_filtering(raw_images, cutoff_freq)

    # Save processed images
    for i, img in enumerate(proc_images):
        processed_filename = f"{filenames[i]}_freq_{cutoff_freq:04d}.png"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    end_time = time.time()
    print(f"Execution time for K-space Filtering: {end_time - start_time:.4f} seconds")
    """


    # Gaussian Filtering
    """
    start_time = time.time()
    kernel_size = 5 
    sigma = 0.5

    # Apply Gaussian filter 
    proc_images = ip.apply_gaussian_filter(raw_images, kernel_size, sigma)

    # Save processed images
    for i, img in enumerate(proc_images):
        processed_filename = f"{filenames[i]}_kernel_{kernel_size:02d}_sigma_{sigma:.1f}.png"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    
    end_time = time.time()
    print(f"Execution time for Gaussian filter: {end_time - start_time:.4f} seconds")
    """


    # Gaussian Filtering fine tuning
    """
    start_time = time.time()
    kernel_sizes = [5, 11, 17, 23, 29]  # Example odd kernel sizes
    sigma_values = [0.5, 1, 1.5, 2, 2.5]  # Example sigma values

    # Test Gaussian filter with different parameters
    smooth_imgs = test_gaussian_filter_parameters_on_list(raw_images, kernel_sizes, sigma_values)

    # Save each processed image with different Gaussian filter parameters
    for i, processed_images_dict in enumerate(smooth_imgs):
        for (kernel_size, sigma), processed_images in processed_images_dict.items():
            # Iterate through each processed image in the list
            for j, processed_image in enumerate(processed_images):
                # Construct a filename that includes the Gaussian parameters and the index of the image
                processed_filename = f"{filenames[i]}_kernel_{kernel_size:02d}_sigma_{sigma:.1f}_{j}.png"
                cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), processed_image)

    print("Gaussian Filtering Fine tuning completed.")
    end_time = time.time()
    print(f"Execution time for Gaussian filter fine-tuning: {end_time - start_time:.4f} seconds")
    """


    # Method from acssensors 0c01681 - Gradient-Based Rapid Digital Immunoassay for High-Sensitivity Cardiac Troponin T (hs-cTnT) Detection in 1 μL Plasma 
    """
    # Subtract the background from each raw image
    start_time = time.time()
    subtracted_images = [cv2.subtract(bg_image[0], raw_image) for raw_image in raw_images]

    # Median filter background substraction

    median_kernel_sizes = range(3, 30, 2)

    for kernel_size in median_kernel_sizes:
        
        proc_imgs = ip.apply_median_filter(subtracted_images, kernel_size)

        # Save the filtered image
        for i, img in enumerate(proc_imgs):
            processed_filename = f"substract_kernel_{kernel_size:02d}_{filenames[i]}"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), subtracted_images[i].astype(np.int16) - img.astype(np.int16))
            #processed_filename = f"{filenames[i]}_kernel_{kernel_size}.png"
            #cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('Median filter background substraction completed.')
    end_time = time.time()
    print(f"Execution time for Background subtraction Step 1 & 2: {end_time - start_time:.4f} seconds")
    """
    

    # CLAHE fine tuning
    """
    start_time = time.time()
    clip_limits = [0.5, 1.0, 2.0, 4.0, 8.0]
    tile_grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]

    # Test CLAHE with different parameters
    proc_imgs = test_clahe_parameters_on_list(raw_images, clip_limits, tile_grid_sizes)

    # Save each processed image with different CLAHE parameters
    for i, processed_images_dict in enumerate(proc_imgs):
        for (clip_limit, tile_grid_size), processed_images in processed_images_dict.items():
            # Iterate through each processed image in the list
            for j, processed_image in enumerate(processed_images):
                # Construct a filename that includes the CLAHE parameters and the index of the image
                processed_filename = f"clip_{clip_limit}_grid_{tile_grid_size[0]:03d}x{tile_grid_size[1]:03d}_{filenames[i]}_{j}.png"
                cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), processed_image)
    print("CLAHE Fine tuning completed.")
    end_time = time.time()
    print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")
    """
    


    # Calculate metrics for the processed images
    proc_images, proc_filenames = load_grayscale_images(PROCESSED_IMAGES_DIRECTORY)
    results_df = calculate_metrics(proc_images, raw_images, bg_image[0], filenames, proc_filenames)

    # Save the results to an Excel file
    results_excel_path = os.path.join(RESULTS_DIRECTORY, "image_processing_metrics.xlsx")
    
    # Assuming `results_df` is your DataFrame
    results_df['SNR'] = results_df['SNR'].map(lambda x: f"{x:.1f}")
    results_df['CNR'] = results_df['CNR'].map(lambda x: f"{x:.2f}")
    results_df['Weber Contrast'] = results_df['Weber Contrast'].map(lambda x: f"{x:.3f}")
    results_df['SSIM'] = results_df['SSIM'].map(lambda x: f"{x:.2f}")

    
    results_df.to_excel(results_excel_path, index=False)

    print(f"Metrics results saved to {results_excel_path}")



if __name__ == "__main__":
    main()