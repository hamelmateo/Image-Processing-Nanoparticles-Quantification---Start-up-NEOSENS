"""
Created on Wed Oct 25 17:13:01 2023

@author: Mateo HAMEL
"""

try:
    # Standard Library Imports
    import os
    import json
    from typing import List

    # Third-party Library Imports
    import cv2
    import numpy as np
    import pandas as pd
    from skimage.metrics import structural_similarity as ssim

    # Local Module Imports
    import image_metrics as im
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

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
CLAHE_CLIP_LIMIT = config['CLAHE_CLIP_LIMIT']
CLAHE_TILE_GRID_SIZE = config['CLAHE_TILE_GRID_SIZE']

ROI_RADIUS = config['ROI_RADIUS']
EXCEL_FILE_NAME = config['EXCEL_FILE_NAME']
ALLOWED_IMAGE_EXTENSIONS = tuple(config['ALLOWED_IMAGE_EXTENSIONS'])

BACKGROUND_IMAGE_DIRECTORY = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\00_images_background"


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

    image_filenames = [filename for filename in os.listdir(directory_path) if filename.endswith(('.png', '.JPG', '.jpeg', '.jpg'))]
    if not image_filenames:
        raise ValueError("No images found in the specified directory.")

    print(f"Attempting to load {len(image_filenames)} images...")
    grayscale_images = [cv2.imread(os.path.join(directory_path, filename), cv2.IMREAD_GRAYSCALE) for filename in image_filenames]

    if not grayscale_images:
        raise ValueError("One or more images could not be loaded. Please check the file paths and formats.")

    return grayscale_images, image_filenames


def test_clahe_parameters_on_list(images, clip_limits, tile_grid_sizes):
    """
    Apply CLAHE with different parameters to a list of images and return the processed images.

    Args:
        images (List[np.ndarray]): The list of input images.
        clip_limits (List[float]): List of CLAHE clip limits to test.
        tile_grid_sizes (List[Tuple[int, int]]): List of tile grid sizes to test.

    Returns:
        List[Dict]: A list of dictionaries, each containing processed images with parameter settings as keys.
    """
    processed_images_list = []
    for image in images:
        processed_images = {}
        for clip_limit in clip_limits:
            for tile_grid_size in tile_grid_sizes:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                processed_image = clahe.apply(image)
                processed_images[(clip_limit, tile_grid_size)] = processed_image
        processed_images_list.append(processed_images)
    return processed_images_list


def apply_kspace_filtering(images, cutoff_freq):
    def to_frequency_domain(image):
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        return f_shift

    def apply_low_pass_filter(k_space, cutoff_frequency):
        rows, cols = k_space.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1
        filtered_k_space = k_space * mask
        return filtered_k_space

    def to_spatial_domain(f_shift):
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back
    
    filtered_images = []
    for i, img in enumerate(images):
        # Convert to frequency domain
        f_shift = to_frequency_domain(img)

        # Apply a low-pass filter
        filtered_k_space = apply_low_pass_filter(f_shift, cutoff_freq)  # Example cutoff

        # Convert back to spatial domain
        filtered_img = to_spatial_domain(filtered_k_space)
        filtered_images.append(filtered_img)
    
    return filtered_images


def test_gaussian_filter_parameters_on_list(images, kernel_sizes, sigma_values):
    """
    Apply Gaussian filter with different parameters to a list of images and return the processed images.

    Args:
        images (List[np.ndarray]): The list of input images.
        kernel_sizes (List[int]): List of Gaussian kernel sizes to test.
        sigma_values (List[float]): List of sigma values to test.

    Returns:
        List[Dict]: A list of dictionaries, each containing processed images with parameter settings as keys.
    """
    processed_images_list = []
    for image in images:
        processed_images = {}
        for kernel_size in kernel_sizes:
            for sigma in sigma_values:
                # Ensure kernel size is odd
                if kernel_size % 2 == 0:
                    kernel_size += 1

                processed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
                processed_images[(kernel_size, sigma)] = processed_image
        processed_images_list.append(processed_images)
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
    """    

    signal_masks = [cv2.imread(os.path.join(MASKS_DIRECTORY, f'mask_{filename}'), cv2.IMREAD_GRAYSCALE) for filename in filenames]
    bg_mask = cv2.imread(os.path.join(MASKS_BG_DIRECTORY, f'mask_{filenames[0]}'), cv2.IMREAD_GRAYSCALE)

    metrics = []
    for raw_image, filename, signal_mask in zip(raw_images, filenames, signal_masks):
        
        # Metrics for raw images
        raw_metrics = {
            'Filename': filename,
            'SNR': im.calculate_snr(raw_image, raw_images[0], signal_mask, bg_mask),
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
                'SSIM': ssim(raw_image, proc_image)
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
    
    # Principal thread
    """  
    # Call image processing functions and save them after each process
    filtered_imgs = image_processing.apply_median_filter(images, MEDIAN_FILTER_KERNEL_SIZE)

    # Save the image
    for i, img in enumerate(filtered_imgs):
        processed_filename = f"filtered_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)

        
    # Increase contrast
    proc_imgs = image_processing.apply_clahe(images, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

    # Save the contrasted image
    for i, img in enumerate(proc_imgs):
        processed_filename = f"processed_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
        
    cv2.imshow('Original Image', cv2.resize(images[5], (0, 0), fx=1, fy=1))
    #cv2.imshow('Filtered Image', cv2.resize(filtered_imgs[5], (0, 0), fx=1, fy=1))
    cv2.imshow('CLAHE Image', cv2.resize(proc_imgs[5], (0, 0), fx=1, fy=1))
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Destroy all the windows when a key is pressed
    """ 


    # K-space filtering fine tuning
    """
    cutoff_freqs = list(range(10, 300, 5))  # Extended range of values

    # Apply K-space filtering
    for freq in cutoff_freqs:
        proc_imgs = apply_kspace_filtering(raw_images, freq)

        # Save the filtered image
        for i, img in enumerate(proc_imgs):
            processed_filename = f"{filenames[i]}_freq_{freq:04d}.png"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    """


    # K-space filtering
    """
    cutoff_freq = 160
    proc_images = apply_kspace_filtering(raw_images, cutoff_freq)
    #for i, img in enumerate(proc_images):
    #    processed_filename = f"{filenames[i]}_freq_{cutoff_freq:04d}.png"
    #    cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('K-space filtering done')
    """


    # Gaussian Filter
    """
    # Define ranges of parameters to test
    kernel_size = 5  # Example odd kernel sizes
    sigma = 0.5  # Example sigma values

    # Test Gaussian filter with different parameters
    proc_images =  [cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma) for image in proc_images]
    for i, img in enumerate(proc_images):
        processed_filename = f"{filenames[i]}_kernel_{kernel_size:02d}_sigma_{sigma:.1f}.png"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('Gaussian filter done')
    """


    # Gaussian Filter fine tuning
    """
    # Define ranges of parameters to test
    kernel_sizes = [5, 11, 17, 23, 29]  # Example odd kernel sizes
    sigma_values = [0.5, 1, 1.5, 2, 2.5]  # Example sigma values


    # Test Gaussian filter with different parameters
    smooth_imgs = test_gaussian_filter_parameters_on_list(raw_images, kernel_sizes, sigma_values)

    # Save each processed image with different Gaussian filter parameters
    for i, processed_images_dict in enumerate(smooth_imgs):
        for (kernel_size, sigma), processed_image in processed_images_dict.items():
            processed_filename = f"{filenames[i]}_kernel_{kernel_size:02d}_sigma_{sigma:.1f}.png"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), processed_image)
    print('Gaussian filter fine-tuning done')
    """


    # Method from acssensors 0c01681 - Gradient-Based Rapid Digital Immunoassay for High-Sensitivity Cardiac Troponin T (hs-cTnT) Detection in 1 μL Plasma 
    # Subtract the background from each raw image
    """
    subtracted_images = [cv2.subtract(bg_image[0], raw_image) for raw_image in raw_images]
    """
    """
    # Save the result image
    for i, img in enumerate(subtracted_images):
        processed_filename = f"{filenames[i]}_substract.png"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('Background substraction')
    """
    # Median filter background substraction
    """
    median_kernel_sizes = range(3, 30, 2)

    for kernel_size in median_kernel_sizes:
        
        proc_imgs = image_processing.apply_median_filter(subtracted_images, kernel_size)

        # Save the filtered image
        for i, img in enumerate(proc_imgs):
            processed_filename = f"substract_kernel_{kernel_size:02d}_{filenames[i]}"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), subtracted_images[i].astype(np.int16) - img.astype(np.int16))
            #processed_filename = f"{filenames[i]}_kernel_{kernel_size}.png"
            #cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('Median filter background substraction')
    """
    

    # Median filtering
    """
    kernel_size = 3

    proc_imgs = image_processing.apply_median_filter(subtracted_images, kernel_size)
    result_img = []
    # Save the filtered image
    for i, img in enumerate(proc_imgs):
        result_img.append(subtracted_images[i].astype(np.int16) - img.astype(np.int16))
        processed_filename = f"substract_kernel_{kernel_size:02d}_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), subtracted_images[i].astype(np.int16) - img.astype(np.int16))
        #processed_filename = f"{filenames[i]}_kernel_{kernel_size}.png"
        #cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
    print('Median filter background substraction')
    """


    # From dict to list of images
    """
    smooth_images = []
    for image_dict in smooth_imgs:
        for image in image_dict.values():
            smooth_images.append(image)
    """


    # CLAHE parameter fine tuning
    """
    # Define ranges of parameters to test
    #clip_limits = [1.0]
    clip_limits = [0.5, 1.0, 2.0, 4.0, 8.0]  # Example values
    #tile_grid_sizes = [(8,8)]
    tile_grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]  # Trade off Contrast & SNR --> clip = 2.0, grid = 64x64

    # Test CLAHE with different parameters
    proc_imgs = test_clahe_parameters_on_list(raw_images, clip_limits, tile_grid_sizes)

    # Save each processed image with different CLAHE parameters
    for i, processed_images_dict in enumerate(proc_imgs):
        for (clip_limit, tile_grid_size), processed_image in processed_images_dict.items():
            # Construct a filename that includes the CLAHE parameters
            processed_filename = f"clip_{clip_limit}_grid_{tile_grid_size[0]:03d}x{tile_grid_size[1]:03d}_{filenames[i]}"
            cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), processed_image)
    print('CLAHE done')
    """



    # Calculate metrics for the processed images
    proc_images, proc_filenames = load_grayscale_images(PROCESSED_IMAGES_DIRECTORY)
    results_df = calculate_metrics(proc_images, raw_images, bg_image, filenames, proc_filenames)

    # Save the results to an Excel file
    results_excel_path = os.path.join(RESULTS_DIRECTORY, "image_processing_metrics_results.xlsx")
    
    # Assuming `results_df` is your DataFrame
    results_df['SNR'] = results_df['SNR'].map(lambda x: f"{x:.1f}")
    results_df['CNR'] = results_df['CNR'].map(lambda x: f"{x:.2f}")
    results_df['Weber Contrast'] = results_df['Weber Contrast'].map(lambda x: f"{x:.3f}")
    results_df['SSIM'] = results_df['SSIM'].map(lambda x: f"{x:.2f}")

    
    results_df.to_excel(results_excel_path, index=False)

    print(f"Metrics results saved to {results_excel_path}")



if __name__ == "__main__":
    main()