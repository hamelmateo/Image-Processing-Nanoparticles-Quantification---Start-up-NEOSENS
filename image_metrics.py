"""
Created on Fri Nov 24 08:53:22 2023

@author: Mateo HAMEL
"""

try: 
    # Standard Library Imports
    from typing import Tuple

    # Third-party Library Imports
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Define functions to calculate various metrics
def calculate_snr(image: np.ndarray, signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of an image using defined signal and background ROIs.
    
    Args:
        image (np.ndarray): The image to analyze.
        signal_mask (np.ndarray): Mask to isolate the signal region.
        background_mask (np.ndarray): Mask to isolate the background region.
    
    Returns:
        float: The calculated SNR.
    """
    # Extract signal and background regions
    signal_region = image[signal_mask == 255]
    background_region = image[background_mask == 255]

    # Calculate SNR
    mean_signal = np.mean(signal_region)
    std_noise = np.std(background_region)
    return mean_signal / std_noise if std_noise != 0 else float('inf')


def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate the contrast of an image.
    Contrast = (Imax - Imin) / (Imax + Imin)
    """
    Imax = np.max(image)
    Imin = np.min(image)
    return (Imax - Imin) / (Imax + Imin)


def calculate_precision_recall_fscore(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F-score for binary classification tasks.
    """
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, fscore


def calculate_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve (AUC) from prediction scores.
    """
    return roc_auc_score(y_true, y_scores)


def calculate_efficiency(start_time: float, end_time: float) -> float:
    """
    Calculate the efficiency of the algorithm based on execution time.
    """
    return end_time - start_time


def calculate_repeatability(measurements: np.ndarray) -> float:
    """
    Calculate the repeatability of a measurement.
    """
    return np.std(measurements)


def calculate_cv(measurements: np.ndarray) -> float:
    """
    Calculate the Coefficient of Variation (CV).
    CV = (Standard Deviation / Mean) * 100
    """
    return (np.std(measurements) / np.mean(measurements)) * 100


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    """
    return ssim(original, processed)


