"""
Created on Sun Oct 22 14:21:03 2023

@author: Mateo HAMEL
"""

try:
    import os
    import cv2
    import numpy as np
    from typing import List, Tuple
except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


def apply_nanoparticles_segmentation(images: List[np.ndarray], filenames: List[str], output_folder: str, block_size: int, constant: int) -> List[np.ndarray]:
    """
    Apply adaptive thresholding and save the result.

    Args:
        images (List[np.ndarray]): List of images to segment.
        filenames (List[str]): List of filenames corresponding to each image.
        output_folder (str): Folder to save the output images.
        block_size (int): The size of the neighbourhood area used for threshold calculation.
        constant (int): A constant value subtracted from mean or weighted sum of the neighbourhood pixels.

    Returns:
        List[np.ndarray]: List of segmented images.    
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if len(images) != len(filenames):
        raise ValueError("The number of images and filenames must be the same.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    segmented_images = []

    for i, img in enumerate(images):
        # Ensure the image is in grayscale
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, constant)

        # Save the thresholded image
        segmented_filename = f"segmented_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, segmented_filename), adaptive_thresh)

        segmented_images.append(adaptive_thresh)

    return segmented_images


def get_circle_roi(image: np.ndarray, roi_radius: int) -> np.ndarray:
    """
    Ask the user to define a circular ROI on the image.

    Args:
        image (np.ndarray): The image on which to define the ROI.
        roi_radius (int): The radius of the circular ROI.

    Returns:
        np.ndarray: The mask image (same size as input, 0 outside ROI, 255 inside).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("The image must be a NumPy array.")
    if roi_radius <= 0:
        raise ValueError("ROI radius must be a positive integer.")

    
    mask = np.zeros_like(image)
    
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(mask, (x, y), param['radius'], 255, -1)
            param['finished'] = True
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle, {'radius': roi_radius, 'finished': False})
    
    while True:
        combined = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        cv2.imshow('image', combined)
        if cv2.waitKey(20) & 0xFF == 27:  # exit on ESC
            break
    
    cv2.destroyAllWindows()
    
    return mask


def apply_masking_roi(segm_images: List[np.ndarray], images: List[np.ndarray], filenames: List[str], output_folder: str, roi_radius:int) -> List[Tuple[np.ndarray, str]]:
    """
    Apply a circular ROI mask to a list of images.

    Args:
        segm_images (List[np.ndarray]): List of segmented images to apply the mask to.
        images (List[np.ndarray]): Original list of images to define the ROI.
        filenames (List[str]): List of filenames corresponding to each image.
        output_folder (str): Directory where masked images will be saved.
        roi_radius (int): The radius of the ROI to be applied.

    Returns:
        List[Tuple[np.ndarray, str]]: List of tuples, each containing a masked image and its filename.
    """
    if not segm_images or not images:
        raise ValueError("The lists of segmented or original images are empty.")
    if len(segm_images) != len(images) or len(images) != len(filenames):
        raise ValueError("All lists must have the same number of elements.")
    
    masked_imgs = []
    for i, img in enumerate(images):
        # Use the image to define the ROI
        mask = get_circle_roi(img, roi_radius)
        
        # Apply the mask to the image
        masked_img = cv2.bitwise_and(segm_images[i], segm_images[i], mask=mask)
        
        # Save the final segmented image
        masked_filename = f"masked_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, masked_filename), masked_img)

        # Append the masked image and filename to the list
        masked_imgs.append((masked_img, masked_filename))
    
    return masked_imgs


def count_white_pixels(binary_image: np.ndarray) -> int:
    """
    Count the number of white pixels in a binary image.

    Args:
        binary_image (np.ndarray): The binary image to analyze.

    Returns:
        int: The number of white pixels.
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError("The input must be a NumPy array.")
    if binary_image.ndim != 2 or binary_image.dtype != np.uint8:
        raise ValueError("The image must be a 2D array with a uint8 datatype.")

    # Count white pixels
    return cv2.countNonZero(binary_image)