"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

# Standard Library Imports
import os
import time

# Third-party Library Imports
import cv2
import numpy as np
from typing import List

# Local Module Imports
import image_processing


# Configuration Constants
RAW_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\raw_images'
PROCESSED_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\processed_images'
TEMPORAL_AVERAGE_WINDOW_SIZE = 10
MEDIAN_FILTER_KERNEL_SIZE = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


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


def apply_nanoparticles_segmentation(images: List[np.ndarray], output_folder: str) -> List[np.ndarray]:
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

    seg_imgs = []
    kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations

    for i, img in enumerate(images):
        # Ensure the image is in grayscale
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the original grayscale image
        cv2.imwrite(os.path.join(output_folder, f"image_{i}_original.png"), gray)

        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(output_folder, f"image_{i}_thresholded.png"), thresh)

        # Apply morphological closing to fill small holes and gaps
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imwrite(os.path.join(output_folder, f"image_{i}_closing.png"), closing)

        # Remove small white regions
        sure_bg = cv2.dilate(closing, kernel, iterations=3)

        # Finding sure foreground area using distance transform and thresholding
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply Watershed
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert to BGR for watershed
        cv2.watershed(img_color, markers)
        img_color[markers == -1] = [0, 0, 255]  # Marking the boundaries with red color

        # Save the final segmented image
        cv2.imwrite(os.path.join(output_folder, f"image_{i}_segmented.png"), img_color)

        seg_imgs.append(img_color)

    return seg_imgs

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
            cv2.circle(mask, (x, y), param['radius'], (255, 255, 255), -1)
            param['finished'] = True
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle, {'radius': 50, 'finished': False})
    
    while True:
        combined = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        cv2.imshow('image', combined)
        if cv2.waitKey(20) & 0xFF == 27:  # exit on ESC
            break
    
    cv2.destroyAllWindows()
    
    return mask

def apply_masking_roi(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Ask the user to define a circular ROI on one of the images and apply a masking on everything besides the ROI to all images of the list.

    Args:
        images (List[np.ndarray]): List of images to apply the mask to.

    Returns:
        List[np.ndarray]: List of masked images.    
    """
    if not images:
        return []
    
    # Use the first image to define the ROI
    mask = get_circle_roi(images[0])
    
    # Apply the mask to all images
    masked_imgs = [cv2.bitwise_and(img, img, mask=mask) for img in images]
    
    return masked_imgs


def main():
    """
    Main execution function.

    This function orchestrates the loading of images, their processing, and the subsequent display
    of results. Execution time for each major step is printed to the console.
    """
    try:
        # Load images
        start_time = time.time()
        raw_images, _ = load_grayscale_images(RAW_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")

        # Process images
        #processed_images = image_processing.process_images(raw_images, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

        # Count Au-NPS
        masked_images = apply_masking_roi(raw_images)
        segmented_images = apply_nanoparticles_segmentation(masked_images, PROCESSED_IMAGES_DIRECTORY)
        
    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
