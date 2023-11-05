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
import pandas as pd

# Local Module Imports
import image_processing


# Configuration Constants
RAW_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\01_images_raw'
PROCESSED_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\02_images_processed'
MASKED_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\03_images_masked'
SEGMENTED_IMAGES_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\04_images_segmented'
RESULTS_DIRECTORY = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\05_Results'
TEMPORAL_AVERAGE_WINDOW_SIZE = 10
MEDIAN_FILTER_KERNEL_SIZE = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
ADAPTIVE_THRESHOLDING_BLOCK_SIZE = 3 # Size of a pixel neighborhood that is used to calculate a threshold value.
ADAPTIVE_THRESHOLDING_CONSTANT = 45 # Constant subtracted from the mean or weighted mean.


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

def count_white_pixels(binary_image):
    """
    Count the number of white pixels in a binary image.

    Args:
        binary_image: Binary image where white pixels have the value 255.

    Returns:
        int: The number of white pixels.
    """
    # Assuming white pixels are 255, count all non-zero values
    
    return cv2.countNonZero(binary_image)



def apply_nanoparticles_segmentation(images: List[np.ndarray], filenames: List[str], output_folder: str) -> List[np.ndarray]:
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
                                                cv2.THRESH_BINARY_INV, ADAPTIVE_THRESHOLDING_BLOCK_SIZE, ADAPTIVE_THRESHOLDING_CONSTANT)

        # Save the thresholded image
        segmented_filename = f"segmented_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, segmented_filename), adaptive_thresh)

        segmented_images.append(adaptive_thresh)

    return segmented_images

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


def apply_masking_roi(segm_images: List[np.ndarray], images: List[np.ndarray], filenames: List[str], output_folder: str) -> List[np.ndarray]:
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
        masked_img = cv2.bitwise_and(segm_images[i], segm_images[i], mask=mask)
        
        # Save the final segmented image
        masked_filename = f"masked_{filenames[i]}"
        cv2.imwrite(os.path.join(output_folder, masked_filename), masked_img)

        # Append the masked image and filename to the list
        masked_imgs.append((masked_img, masked_filename))
    
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
        raw_images, filenames = load_grayscale_images(RAW_IMAGES_DIRECTORY)
        end_time = time.time()
        print(f"Execution time for loading images: {end_time - start_time:.4f} seconds")

        # Process images
        #processed_images = image_processing.process_images(raw_images, TEMPORAL_AVERAGE_WINDOW_SIZE, MEDIAN_FILTER_KERNEL_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

        # Count Au-NPS
        segmented_images = apply_nanoparticles_segmentation(raw_images, filenames, SEGMENTED_IMAGES_DIRECTORY)
        masked_images = apply_masking_roi(segmented_images, raw_images, filenames, MASKED_IMAGES_DIRECTORY)
        
        # Count white pixels and store results in a DataFrame
        counts = [(filename, count_white_pixels(img)) for img, filename in masked_images]
        df_counts = pd.DataFrame(counts, columns=['Filename', 'White Pixel Count'])

        # Save to Excel file
        excel_file_path = os.path.join(RESULTS_DIRECTORY, 'white_pixel_counts.xlsx')
        df_counts.to_excel(excel_file_path, index=False)
        print(f"White pixel counts have been written to {excel_file_path}")

    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
