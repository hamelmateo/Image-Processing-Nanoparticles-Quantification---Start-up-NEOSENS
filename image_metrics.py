"""
Created on Fri Nov 24 08:53:22 2023
@author: Mateo HAMEL
"""

try: 
    # Standard Library Imports
    from typing import Tuple

    # Third-party Library Imports
    import numpy as np
    import cv2
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Define functions to calculate various metrics
def calculate_snr(image: np.ndarray, bg_image: np.ndarray, signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of an image using defined signal and background ROIs.
    
    Args:
        image (np.ndarray): The image to analyze.
        bg_image (np.ndarray): The reference background image.
        signal_mask (np.ndarray): Mask to isolate the signal region.
        background_mask (np.ndarray): Mask to isolate the background region.
    
    Returns:
        float: The calculated SNR.

    Raises:
        ValueError: If any input is not a numpy ndarray.
        ValueError: If the shapes of image, bg_image, signal_mask, or background_mask do not match.
        ValueError: If signal_mask or background_mask is not binary (i.e., not containing only 0 and 255).
    """
    if not all(isinstance(arr, np.ndarray) for arr in [image, bg_image, signal_mask, background_mask]):
        raise ValueError("All inputs must be numpy ndarrays.")

    if not (image.shape == bg_image.shape == signal_mask.shape == background_mask.shape):
        raise ValueError("All inputs must have the same shape.")

    if not (np.all(np.isin(signal_mask, [0, 255])) and np.all(np.isin(background_mask, [0, 255]))):
        raise ValueError("Signal and background masks must be binary (contain only 0 and 255).")

    try:
        # Extract signal and background regions
        signal_region = cv2.bitwise_not(image[signal_mask == 255]) # bitwise_not because our signal are dark spots with bright field imaging
        background_region = cv2.bitwise_not(bg_image[background_mask == 255])

        # Calculate SNR
        mean_signal = np.mean(signal_region)
        std_noise = np.std(background_region)
        return mean_signal / std_noise if std_noise != 0 else float('inf')

    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating SNR: {e}")


def calculate_cnr(image: np.ndarray, bg_image: np.ndarray, signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """
    Calculate the Contrast-to-Noise Ratio (CNR) of an image.

    Args:
        image (np.ndarray): The image to analyze.
        bg_image (np.ndarray): The reference background image.
        signal_mask (np.ndarray): Mask to isolate the signal region.
        background_mask (np.ndarray): Mask to isolate the background region.

    Returns:
        float: The calculated CNR.

    Raises:
        ValueError: If any input is not a numpy ndarray.
        ValueError: If the shapes of image, bg_image, signal_mask, or background_mask do not match.
        ValueError: If signal_mask or background_mask is not binary (i.e., not containing only 0 and 255).
    """
    if not all(isinstance(arr, np.ndarray) for arr in [image, bg_image, signal_mask, background_mask]):
        raise ValueError("All inputs must be numpy ndarrays.")

    if not (image.shape == bg_image.shape == signal_mask.shape == background_mask.shape):
        raise ValueError("All inputs must have the same shape.")

    if not (np.all(np.isin(signal_mask, [0, 255])) and np.all(np.isin(background_mask, [0, 255]))):
        raise ValueError("Signal and background masks must be binary (contain only 0 and 255).")

    try:
        # Extract signal and background regions
        signal_region = cv2.bitwise_not(image[signal_mask == 255])
        background_region = cv2.bitwise_not(bg_image[background_mask == 255])

        # Calculate CNR
        mean_signal = np.mean(signal_region)
        mean_background = np.mean(background_region)
        std_noise = np.std(background_region)

        return abs(mean_signal - mean_background) / std_noise if std_noise != 0 else float('inf')
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating CNR: {e}")


def calculate_weber_contrast(image: np.ndarray, bg_image: np.ndarray, signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """
    Calculate the Weber Contrast of an image.

    Args:
        image (np.ndarray): The image to analyze.
        bg_image (np.ndarray): The reference background image.
        signal_mask (np.ndarray): Mask to isolate the signal region.
        background_mask (np.ndarray): Mask to isolate the background region.

    Returns:
        float: The calculated Weber Contrast.

    Raises:
        ValueError: If any input is not a numpy ndarray.
        ValueError: If the shapes of image, bg_image, signal_mask, or background_mask do not match.
        ValueError: If signal_mask or background_mask is not binary (i.e., not containing only 0 and 255).
    """
    if not all(isinstance(arr, np.ndarray) for arr in [image, bg_image, signal_mask, background_mask]):
        raise ValueError("All inputs must be numpy ndarrays.")

    if not (image.shape == bg_image.shape == signal_mask.shape == background_mask.shape):
        raise ValueError("All inputs must have the same shape.")

    if not (np.all(np.isin(signal_mask, [0, 255])) and np.all(np.isin(background_mask, [0, 255]))):
        raise ValueError("Signal and background masks must be binary (contain only 0 and 255).")

    try:
        # Extract signal and background regions
        signal_region = cv2.bitwise_not(image[signal_mask == 255])
        background_region = cv2.bitwise_not(bg_image[background_mask == 255])

        mean_signal = np.mean(signal_region)
        mean_background = np.mean(background_region)

        return abs(mean_signal - mean_background) / mean_background if mean_background != 0 else float('inf')
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating Weber Contrast: {e}")


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        original (np.ndarray): The original image.
        processed (np.ndarray): The processed image to compare against the original.

    Returns:
        float: The calculated SSIM.

    Raises:
        ValueError: If any input is not a numpy ndarray or if their shapes do not match.
    """
    if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
        raise ValueError("Both inputs must be numpy ndarrays.")

    if original.shape != processed.shape:
        raise ValueError("The shapes of the original and processed images must match.")

    try:
        return ssim(original, processed)
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating SSIM: {e}")


