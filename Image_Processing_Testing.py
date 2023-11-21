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
    from skimage.measure import shannon_entropy
    from skimage.metrics import structural_similarity as ssim

    # Local Module Imports
    import image_processing
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


def calculate_metrics(proc_imgs, original_images):

    # Define the metric functions
    def measure_contrast(image):
        return np.std(image)

    def measure_entropy(image):
        return shannon_entropy(image)

    def measure_edge_density(image, threshold_ratio=0.1):
        """
        Measure the edge density of an image using Sobel operator.
        Args:
            image (np.ndarray): The input image.
            threshold_ratio (float): Proportion of the maximum edge value to be used as threshold.
        Returns:
            float: The edge density of the image.
        """
        # Apply Sobel operator
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

        # Calculate threshold as a proportion of the maximum edge value
        edge_threshold = threshold_ratio * np.max(edges)

        # Calculate edge density
        edge_density = np.sum(edges > edge_threshold) / (image.shape[0] * image.shape[1])
        return edge_density

    def measure_ssim(original, processed):
        # Ensure the images are the same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Convert images to the same data type if necessary
        if original.dtype != processed.dtype:
            processed = processed.astype(original.dtype)

        # Calculate SSIM
        return ssim(original, processed)

    # Initialize a list to store the results
    results_data = []

    # Loop through each processed image and calculate metrics
    for i, processed_image in enumerate(proc_imgs):
        original_image = original_images[i]
        contrast = measure_contrast(processed_image)
        entropy = measure_entropy(processed_image)
        edge_density = measure_edge_density(processed_image)
        ssim_value = measure_ssim(original_image, processed_image)

        results_data.append({
            "Filename": filenames[i],
            "Contrast": contrast,
            "Entropy": entropy,
            "Edge Density": edge_density,
            "SSIM": ssim_value
        })
    
    return results_data




# Load grayscale images
images, filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)

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


# CLAHE parameter fine tuning
""" 
# Define ranges of parameters to test
clip_limits = [2.0]  # Example values
tile_grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]  # Extended range of values

# Test CLAHE with different parameters
proc_imgs = test_clahe_parameters_on_list(images, clip_limits, tile_grid_sizes)

# Save each processed image with different CLAHE parameters
for i, processed_images_dict in enumerate(proc_imgs):
    for (clip_limit, tile_grid_size), processed_image in processed_images_dict.items():
        # Construct a filename that includes the CLAHE parameters
        processed_filename = f"processed_{filenames[i]}_clip_{clip_limit}_grid_{tile_grid_size[0]}x{tile_grid_size[1]}.png"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), processed_image)
""" 


# K-space filtering & fine tuning
"""
cutoff_freqs = list(range(500, 1000, 10))  # Extended range of values

# Apply K-space filtering
for freq in cutoff_freqs:
    proc_imgs = apply_kspace_filtering(images, freq)

    # Save the filtered image
    for i, img in enumerate(proc_imgs):
        processed_filename = f"processed_freq_{freq}_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
"""


# Median filter background substraction
"""
median_kernel_sizes = range(99, 441, 2)

for kernel_size in median_kernel_sizes:
    proc_imgs = image_processing.apply_median_filter(images, kernel_size)

    # Save the filtered image
    for i, img in enumerate(proc_imgs):
        processed_filename = f"processed_kernel_{kernel_size}_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img - images[i])
        processed_filename = f"median_{kernel_size}_{filenames[i]}"
        cv2.imwrite(os.path.join(PROCESSED_IMAGES_DIRECTORY, processed_filename), img)
"""


# Calculate metric of images
"""
results_df = pd.DataFrame(calculate_metrics(proc_imgs, images))

# Save the results to an Excel file
results_df.to_excel(os.path.join(RESULTS_DIRECTORY, "Kspace_metrics_results.xlsx"), index=False)
"""