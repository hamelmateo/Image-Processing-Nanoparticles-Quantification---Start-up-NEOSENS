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


def select_and_create_mask(image_path: str, mask_path: str) -> np.ndarray:
    """
    Allows the user to select a circular ROI on the image and creates a mask based on that ROI.

    Args:
        image_path (str): Path to the image on which to select the ROI.
        mask_path (str): Path where the mask will be saved.

    Returns:
        np.ndarray: The created mask.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Display the image and wait for the user to select a circular ROI
    roi = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Image")

    # ROI returned as x, y (top left corner) and width, height
    x, y, w, h = roi
    center = (x + w // 2, y + h // 2)
    radius = min(w // 2, h // 2)

    # Create a black mask
    mask = np.zeros_like(image)

    # Draw a filled circle in the mask based on the selected ROI
    cv2.circle(mask, center, radius, (255), thickness=-1)

    # Save the mask
    cv2.imwrite(mask_path, mask)

    return mask


def select_and_create_mask(image_path: str, mask_path: str, scale_factor: float = 0.25) -> np.ndarray:
    """
    Allows the user to select an ROI on a scaled version of the image and creates a mask based on that ROI
    applied to the original high-resolution image.

    Args:
        image_path (str): Path to the image on which to select the ROI.
        mask_path (str): Path where the mask will be saved.
        scale_factor (float): Factor to scale the image by for ROI selection. Default is 0.5 (50%).

    Returns:
        np.ndarray: The created mask for the original image.
    """
    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Scale down the image for ROI selection
    height, width = original_image.shape
    scaled_image = cv2.resize(original_image, (int(width * scale_factor), int(height * scale_factor)))

    # Display the scaled image and wait for the user to select an ROI
    roi = cv2.selectROI("Image", scaled_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Image")

    # Scale the ROI coordinates back to the original image size
    x, y, w, h = roi
    x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)

    # Create a mask for the original image based on the selected ROI
    mask = np.zeros_like(original_image)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # Using a rectangle as an example; adjust as needed

    # Save the mask
    cv2.imwrite(mask_path, mask)

    return mask



def load_or_create_masks(filenames: List[str], masks_directory_path: str, img_directory_path: str) -> List[np.ndarray]:
    """
    Loads or creates individual masks for each image if TIME_RESOLVED_IMAGES is False.

    Args:
        filenames (List[str]): List of filenames corresponding to each image.
        masks_directory_path (str): Directory where masks are stored or will be saved.
        img_directory_path (str): Directory where the original images are located.

    Returns:
        List[np.ndarray]: List of masks for each image.
    """
    masks = []
    for filename in filenames:
        mask_path = os.path.join(masks_directory_path, f"mask_{filename}")
        if os.path.exists(mask_path):
            # Load the existing mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Prompt user to create a new mask for each image
            image_path = os.path.join(img_directory_path, filename)
            mask = select_and_create_mask(image_path, mask_path)  # Assumes implementation of select_and_create_mask
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


def apply_masking(images: List[np.ndarray], masks: List[np.ndarray], filenames: List[str], output_folder: str) -> List[np.ndarray]:
    """
    Apply pre-defined masks to a list of images.

    Args:
        images (List[np.ndarray]): List of images to apply the masks to.
        masks (List[np.ndarray]): List of masks to be applied to the images. Can be a list with a single mask repeated if using the same mask for all images.
        filenames (List[str]): List of filenames corresponding to each image.
        output_folder (str): Directory where masked images will be saved.

    Returns:
        List[np.ndarray]: List of masked images.

    Raises:
        ValueError: If the list of images, masks, or filenames are empty or if the numbers do not match. 
    """
    if not images or not masks:
        raise ValueError("The lists of images and masks cannot be empty.")
    if len(images) != len(masks) or len(images) != len(filenames):
        raise ValueError("The numbers of images, masks, and filenames must match.")

    masked_imgs = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        # Apply the corresponding mask to the image
        masked_img = cv2.bitwise_and(image, image, mask=mask)

        # Save the final masked image
        masked_filename = f"masked_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, masked_filename), masked_img)

        # Append the masked image to the list
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