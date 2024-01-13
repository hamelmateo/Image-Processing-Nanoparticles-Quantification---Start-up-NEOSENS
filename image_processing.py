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


def apply_kspace_filtering(images: List[np.ndarray], cutoff_freq: int) -> List[np.ndarray]:
    """
    Apply k-space filtering to a list of images.

    Args:
        images: List of images to filter.
        cutoff_freq: Cutoff frequency for the k-space filter.

    Returns:
        List: List of filtered images.
        
    Raises:
        ValueError: If the images list is empty.
        ValueError: If cutoff_freq is not a non-negative integer.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    
    if not isinstance(cutoff_freq, int) or cutoff_freq < 0:
        raise ValueError("Cutoff frequency should be a non-negative integer.")
    
    def to_frequency_domain(image):
        try:
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            return f_shift
        except Exception as e:
            raise ValueError("Error during frequency domain transformation.") from e

    def apply_low_pass_filter(k_space, cutoff_frequency):
        try:
            rows, cols = k_space.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1
            filtered_k_space = k_space * mask
            return filtered_k_space
        except Exception as e:
            raise ValueError("Error during low-pass filter application.") from e

    def to_spatial_domain(f_shift):
        try:
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            return img_back
        except Exception as e:
            raise ValueError("Error during spatial domain transformation.") from e
    
    filtered_images = []
    for img in images:
        try:
            # Convert to frequency domain
            f_shift = to_frequency_domain(img)

            # Apply a low-pass filter
            filtered_k_space = apply_low_pass_filter(f_shift, cutoff_freq)  # Example cutoff

            # Convert back to spatial domain
            filtered_img = to_spatial_domain(filtered_k_space)
            filtered_images.append(filtered_img.astype(np.uint8))
        except Exception as e:
            raise ValueError("Error during k-space filtering.") from e
    
    return filtered_images


def compute_temporal_average(images: List[np.ndarray], window_size: int) -> List[np.ndarray]:
    """
    Compute the temporal average of a list of images over a specified window size.
    
    Args:
        images (List[np.ndarray]): List of images to average.
        window_size (int): Number of images to average over.

    Returns:
        List[np.ndarray]: List of temporally averaged images.

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the window size is not a positive integer.
        ValueError: If the number of images is less than the window size.
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

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the kernel size is not a positive odd integer.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size should be a positive odd integer.")
    
    filtered_imgs = [cv2.medianBlur(img, kernel_size) for img in images]
    return filtered_imgs


def apply_gaussian_filter(images: List[np.ndarray], kernel_size: int, sigma: int) -> List[np.ndarray]:
    """
    Apply a gaussian filter to a list of images.
    
    Args:
        images (List[np.ndarray]): List of images to apply the gaussian filter to.
        kernel_size (int): Size of the kernel for the gaussian filter.
        sigma (int): Degree of blurring.

    Returns:
        List[np.ndarray]: List of images after gaussian filtering.

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the kernel size is not a positive odd integer.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size should be a positive odd integer.")
    
    filtered_imgs = [cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma) for img in images]
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

    Raises:
        ValueError: If the list of images is empty.
        TypeError: If not all elements in the images list are NumPy arrays.
        ValueError: If the clip limit is not a positive float.
        ValueError: If the tile grid size is not a tuple of two positive integers.
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


def compute_differential_images(images: List[np.ndarray], original_filenames: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Compute differential images from a list of images and generate new filenames for them.

    Args:
        images (List[np.ndarray]): List of images.
        original_filenames (List[str]): List of original filenames corresponding to the images.

    Returns:
        Tuple[List[np.ndarray], List[str]]: List of differential images and their new filenames.

    Raises:
        ValueError: If fewer than two images are provided for differential imaging.
        ValueError: If the number of images and filenames do not match.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required for differential imaging.")
    if len(images) != len(original_filenames):
        raise ValueError("The number of images and filenames should be equal.")

    differential_images = []
    new_filenames = []

    # assuming each set of three images produces one differential image
    for i in range(len(images) // 3):
        # Compute the absolute difference between consecutive sets of images
        diff = cv2.bitwise_not(cv2.absdiff(images[3*i], images[3*i + 3]))

        differential_images.append(diff)
        new_filename = f"differential_{original_filenames[3*i]}"
        new_filenames.append(new_filename)

    return differential_images, new_filenames


def process_images(images: List[np.ndarray], filenames: List[str], temporal_average_window_size: int, median_kernel_size: int, gaussian_kernel_size: int, 
                   gaussian_sigma: int, kspace_cutoff_freq: int, clip_limit: float, tile_grid_size: Tuple[int, int], output_folder: str) -> List[np.ndarray]:
    """
    Process a list of images for further analysis.
    
    Args:
        images (List[np.ndarray]): List of images to process.
        temporal_average_window_size (int): Number of images over which to compute the temporal average.
        median_kernel_size (int): Size of the kernel used for median filtering.
        gaussian_kernel_size (int): Size of the kernel used for gaussian filtering.
        gaussian_sigma (int): Degree of blurring used for gaussian filtering.
        kspace_cutoff_freq (int): Cutoff frequency for k-space filtering.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization in CLAHE.
        output_folder (str): Folder to save the output images.

    Returns:
        List[np.ndarray]: List of processed images.

    Raises:
        ValueError: If the list of images is empty.
    """
    if not images:
        raise ValueError("The list of images is empty.")

    # 1. Noise reduction
    # Temporal average
    """
    #averaged_imgs = compute_temporal_average(images, temporal_average_window_size)
    """
    # Median filter
    """
    filtered_imgs = apply_median_filter(images, median_kernel_size)
    """
    # K-space filtering
    """"""
    images = apply_kspace_filtering(images, kspace_cutoff_freq)
    
    # Gaussian filtering
    """"""
    images = apply_gaussian_filter(images, gaussian_kernel_size, gaussian_sigma)
    
    # Differential Imaging
    """
    images, filenames = compute_differential_images(images, filenames)
    """
    # 2. Contrast enhancement
    """
    images = apply_clahe(images, clip_limit, tile_grid_size)
    """

    # 3. Save the final processed image
    for i, img in enumerate(images):
        processed_filename = f"processed_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, processed_filename), img)
        
    return images