"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

try:
    import os
    import cv2
    import numpy as np
    from typing import List, Tuple
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


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

    return imgs_average


def apply_median_filter(images: List[np.ndarray], kernel_size: int) -> List[np.ndarray]:
    """
    Apply a median filter to a list of images.
    
    Args:
        images (List[np.ndarray]): List of images to apply the median filter to.
        kernel_size (int): Size of the kernel for the median filter.

    Returns:
        List[np.ndarray]: List of images after median filtering.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size should be a positive odd integer.")
    
    filtered_imgs = [cv2.medianBlur(img, kernel_size) for img in images]
    return filtered_imgs


def apply_clahe(images: List[np.ndarray], clip_limit: float, tile_grid_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a list of images.
    
    Args:
        images (List[np.ndarray]): List of images to enhance.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization.

    Returns:
        List[np.ndarray]: List of images after applying CLAHE.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if clip_limit <= 0:
        raise ValueError("Clip limit should be a positive float.")
    if len(tile_grid_size) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in tile_grid_size):
        raise ValueError("Tile grid size should be a tuple of two positive integers.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    proc_imgs = [clahe.apply(img) for img in images]
    return proc_imgs


def process_images(images: List[np.ndarray], filenames: List[str], temporal_average_window_size: int, median_kernel_size: int, kspace_cutoff_freq: int
                   , clip_limit: float, tile_grid_size: Tuple[int, int], output_folder: str) -> List[np.ndarray]:
    """
    Process a list of images for further analysis.
    
    Args:
        images (List[np.ndarray]): List of images to process.
        temporal_average_window_size (int): Number of images over which to compute the temporal average.
        median_kernel_size (int): Size of the kernel used for median filtering.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization in CLAHE.
        output folder (str): Folder to save the output images.

    Returns:
        List[np.ndarray]: List of processed images.
    """
    if not images:
        raise ValueError("The list of images is empty.")


    # 1. Noise reduction
    # Temporal average
    """
    #averaged_imgs = compute_temporal_average(images, temporal_average_window_size)
    #cv2.imshow("Averaged Image", cv2.resize(averaged_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    """
    # Median filter
    """
    filtered_imgs = apply_median_filter(images, median_kernel_size)
    cv2.imshow("Filtered Image", cv2.resize(filtered_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    """
    # K-space filtering
    """"""
    filtered_imgs = apply_kspace_filtering(images, kspace_cutoff_freq)
    #cv2.imshow("Filtered Image", cv2.resize(filtered_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    

    # 2. Contrast enhancement
    """
    proc_imgs = apply_clahe(images, clip_limit, tile_grid_size)
    #cv2.imshow("CLAHE Image", cv2.resize(proc_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    #cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    """

    # 3. Save the final segmented image
    for i, img in enumerate(filtered_imgs):
        processed_filename = f"processed_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, processed_filename), img)
        
    return filtered_imgs