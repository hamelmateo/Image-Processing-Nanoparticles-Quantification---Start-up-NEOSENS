import cv2
import os
import numpy as np
from typing import List

RAW_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\01_images_raw'
MASKED_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\03_images_masked'

def tune_adaptive_threshold(image_path):
    """
    Interactive tuning of adaptive thresholding parameters.

    Args:
        image_path (str): Path to the input image.
    """

    def on_trackbar_change(_):
        # Retrieve trackbar positions
        block_size = cv2.getTrackbarPos('Block Size', 'Adaptive Thresholding') * 2 + 3  # must be odd and greater than 1
        c = cv2.getTrackbarPos('C', 'Adaptive Thresholding')
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, c
        )

        # Display the result
        cv2.imshow('Adaptive Thresholding', adaptive_thresh)

    # Load the image in grayscale
    original_image = cv2.imread(image_path)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow('Adaptive Thresholding')

    # Create trackbars for block size and C
    cv2.createTrackbar('Block Size', 'Adaptive Thresholding', 1, 100, on_trackbar_change)  # starting from 1 to avoid block size of 0
    cv2.createTrackbar('C', 'Adaptive Thresholding', 0, 50, on_trackbar_change)

    # Initialize the display with the first change
    on_trackbar_change(0)

    # Wait until the user finishes the tuning
    print("Adjust the parameters as needed. Press 'q' to close the window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def get_circle_roi(img: np.ndarray) -> np.ndarray:
    """
    Ask the user to define a circular ROI on the image.

    Args:
        img (np.ndarray): The image on which to define the ROI.

    Returns:
        np.ndarray: The mask image (same size as input, 0 outside ROI, 255 inside).
    """
    mask = np.zeros_like(img)
    
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(mask, (x, y), param['radius'], 255, -1)
            param['finished'] = True
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle, {'radius': 85, 'finished': False})
    
    while True:
        combined = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        cv2.imshow('image', combined)
        if cv2.waitKey(20) & 0xFF == 27:  # exit on ESC
            break
    
    cv2.destroyAllWindows()
    
    return mask


def apply_masking_roi(images: List[np.ndarray], output_folder: str) -> List[np.ndarray]:
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
        mask = get_circle_roi(img)
        
        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        masked_imgs.append(masked_img)
        
        # Save the final segmented image
        cv2.imwrite(os.path.join(output_folder, f"image_{i}_masked.png"), masked_img)
    
    return masked_imgs

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

# Load grayscale images
grayscale_images, image_filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)

# Masking
masked_images = apply_masking_roi(grayscale_images, MASKED_IMAGES_DIRECTORY)

# Loop through images and call the tuning function
for image, filename in zip(masked_images, image_filenames):
    print(f"Tuning thresholds for {filename}...")
    image_path = os.path.join(RAW_IMAGES_DIRECTORY, filename)
    tune_adaptive_threshold(image_path)