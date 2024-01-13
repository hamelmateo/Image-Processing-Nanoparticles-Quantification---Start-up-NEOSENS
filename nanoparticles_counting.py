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


def apply_nanoparticles_segmentation(images: List[np.ndarray], filenames: List[str], output_folder: str, config: dict) -> List[np.ndarray]:
    """
    Applies thresholding segmentation to a list of images based on the method specified in the configuration. 
    Supports 'adaptive', 'otsu', 'fixed', 'median' and 'all' methods. When 'all' is chosen, it applies all four 
    methods and saves separate results for each.

    Args:
        images (List[np.ndarray]): List of images to segment.
        filenames (List[str]): List of filenames corresponding to each image.
        output_folder (str): Folder to save the output images.
        config (dict): Configuration dictionary containing parameters for the thresholding methods. 
                       It should include 'THRESHOLDING_METHOD' key and other related configuration.

    Returns:
        List[np.ndarray]: List of segmented images. If 'all' methods are applied, returns a concatenated list of 
                          images processed by each method. 

    Raises:
        ValueError: If the list of images is empty or if the number of images and filenames do not match.
        ValueError: If an invalid thresholding method is specified in the config.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if len(images) != len(filenames):
        raise ValueError("The number of images and filenames must be the same.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract thresholding configurations from config
    threshold_method = config['THRESHOLDING_METHOD']
    otsu_max_value = config.get('OTSU_THRESHOLDING', {}).get('max_value', 255)
    adaptive_block_size = config.get('ADAPTIVE_THRESHOLDING', {}).get('block_size', 3)
    adaptive_constant = config.get('ADAPTIVE_THRESHOLDING', {}).get('constant', 15)
    threshold_value = config.get('FIXED_THRESHOLDING', {}).get('threshold_value', 135)
    fixed_max_value = config.get('FIXED_THRESHOLDING', {}).get('max_value', 255)
    if threshold_method.lower() == "median":
        median_threshold = calculate_median_threshold(images[0:3])

    def save_and_append(segm_img, method):
        segmented_filename = f"segmented_{filenames[i].split('.')[0]}_{method}.png"
        cv2.imwrite(os.path.join(output_folder, segmented_filename), segm_img)
        segmented_images.append(segm_img)
    
    segmented_images = []


    for i, img in enumerate(images):
        # Apply the selected thresholding technique
        if threshold_method.lower() == "adaptive":
            segm_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_constant) # BINARY_INV to have dark spot labeled as 1
            save_and_append(segm_adaptive, "adaptive")
        elif threshold_method.lower() == "otsu":
            _, segm_otsu = cv2.threshold(img, 0, otsu_max_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            save_and_append(segm_otsu, "otsu")
        elif threshold_method.lower() == "fixed":
            _, segm_fixed = cv2.threshold(img, threshold_value, fixed_max_value, cv2.THRESH_BINARY_INV) # BINARY_INV to have dark spot labeled as 1
            save_and_append(segm_fixed, "fixed")
        elif threshold_method.lower() == "median":
            _, segm_median = cv2.threshold(img, median_threshold, fixed_max_value, cv2.THRESH_BINARY_INV) # BINARY_INV to have dark spot labeled as 1
            save_and_append(segm_median, "median")
        else:
            raise ValueError("Invalid thresholding method specified in config.")
        
    return segmented_images


def calculate_median_threshold(images: List[np.ndarray]) -> int:
    """
    Calculate the median threshold value from images.

    Args:
        images (List[np.ndarray]): List of images.

    Returns:
        int: Median threshold value.

    Raises:
        ValueError: If the list of images is empty.
    """
    if not images:
        raise ValueError("The list of images is empty.")

    medians = []
    for img in images:
        median = np.median(img.flatten())
        medians.append(median)

    threshold_value = int(sum(medians) / len(medians))
    print(f"Median threshold value: {threshold_value}")
    return int(sum(medians) / len(medians))


def load_or_create_masks(filenames: List[str], masks_directory_path: str, img_directory_path: str, roi_radius:int) -> List[np.ndarray]:
    """
    Loads or creates masks for a list of images.

    Args:
        filenames (List[str]): List of filenames corresponding to each image.
        masks_directory_path (str): Directory where masks are stored or will be saved.
        img_directory_path (str): Directory where the original images are located.
        roi_radius (int): The radius of the circular ROI to apply.

    Returns:
        List[np.ndarray]: List of masks for each image.

    Raises:
        ValueError: If the list of filenames is empty.
        FileNotFoundError: If the specified image directory does not exist.
    """
    if not filenames:
        raise ValueError("The list of filenames is empty.")
    if not os.path.exists(img_directory_path):
        raise FileNotFoundError(f"The specified image directory does not exist: {img_directory_path}")
    if not os.path.exists(masks_directory_path):
        os.makedirs(masks_directory_path)
    
    masks = []

    for filename in filenames:
        mask_path = os.path.join(masks_directory_path, f"mask_{filename}")
        if os.path.exists(mask_path):
            # Load the existing mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create a new mask and save it
            image_path = os.path.join(img_directory_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = get_circle_roi(image, roi_radius)
            cv2.imwrite(mask_path, mask)

            # Save the new mask in the masks directory
            cv2.imwrite(mask_path, mask)
            
        masks.append(mask)

    return masks


def get_circle_roi(image: np.ndarray, roi_radius: int) -> np.ndarray:
    """
    Ask the user to define a circular ROI on the image.

    Args:
        image (np.ndarray): The image on which to define the ROI.
        roi_radius (int): The radius of the circular ROI.

    Returns:
        np.ndarray: The mask image (same size as input, 0 outside ROI, 255 inside).

    Raises:
        TypeError: If the image is not a NumPy array.
        ValueError: If the ROI radius is not a positive integer.
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


def apply_masking(images: List[np.ndarray], masks: List[np.ndarray], filenames: List[str], output_folder: str) -> List[Tuple[np.ndarray, str]]:
    """
    Apply pre-defined masks to a list of images.

    Args:
        images (List[np.ndarray]): List of images to apply the mask to.
        masks (List[np.ndarray]): List of masks for each image.
        filenames (List[str]): List of filenames corresponding to each image.
        output_folder (str): Directory where masked images will be saved.

    Returns:
        List[np.ndarray]: List of masked images.

    Raises:
        ValueError: If the list of segmented images or masks are empty or if the number of images and filenames do not match. 
    """
    if not images or not masks:
        raise ValueError("The lists of segmented images or masks are empty.")
    if len(images) != len(masks) or len(masks) != len(filenames):
        raise ValueError("All lists must have the same number of elements.")

    masked_imgs = []
    for i, image in enumerate(images):
        # Apply the pre-defined mask to the segmented image
        masked_img = cv2.bitwise_and(image, image, mask=masks[i])

        # Save the final masked image
        masked_filename = f"masked_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, masked_filename), masked_img)

        # Append the masked image and filename to the list
        masked_imgs.append(masked_img)

    return masked_imgs


def count_white_pixels(binary_image: np.ndarray) -> int:
    """
    Count the number of white pixels in a binary image.

    Args:
        binary_image (np.ndarray): The binary image to analyze.

    Returns:
        int: The number of white pixels.
    
    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the image is not a 2D array with a uint8 datatype.
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError("The input must be a NumPy array.")
    if binary_image.ndim != 2 or binary_image.dtype != np.uint8:
        raise ValueError("The image must be a 2D array with a uint8 datatype.")

    # Count white pixels
    return cv2.countNonZero(binary_image)