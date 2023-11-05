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


def apply_nanoparticles_segmentation(images: List[np.ndarray], filenames: List[str], output_folder: str, block_size: int, constant: int) -> List[np.ndarray]:
    """
    Apply Otsu thresholding followed by closing and Watershed segmentation.

    Args:
        images (List[np.ndarray]): List of images to segment.
        output_folder (str): Folder to save the output images.

    Returns:
        List[np.ndarray]: List of segmented images.    
    """
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
        img (np.ndarray): The image on which to define the ROI.

    Returns:
        np.ndarray: The mask image (same size as input, 0 outside ROI, 255 inside).
    """
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
    Ask the user to define a circular ROI on one of the images and apply a masking on everything besides the ROI to all images of the list.

    Args:
        images (List[np.ndarray]): List of images to apply the mask to.

    Returns:
        List[np.ndarray]: List of masked images.    
    """
    if not images:
        return []
    
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
        binary_image: Binary image where white pixels have the value 255.

    Returns:
        int: The number of white pixels.
    """
    # Assuming white pixels are 255, count all non-zero values
    
    return cv2.countNonZero(binary_image)