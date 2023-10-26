"""
Created on Thu Sep 19 10:35:22 2023

@author: Mateo HAMEL
"""

# Standard library imports
import os
import sys

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Global Variables
RAW_IMAGES_PATH = 'C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\raw_images'
N_TEMP_AVERAGE = 10
MEDIAN_KERNEL_SIZE = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

def Open_images(path):
    """
    Open images from the folder raw_images and put them in grayscale for faster computing

    Input:
        path: path of folder raw_images
    
    Returns:
        raw_imgs: List of images
        filename: List of the name of the images 
    """

    filenames = [f for f in os.listdir(path) if f.endswith(('.png','.JPG','.jpeg'))]
    print(f"Attempting to load {len(filenames)} images...")

    raw_imgs = [cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE) for f in filenames]
    
    if not raw_imgs:
        print("No images were loaded. Please check the file paths and formats.")

    return raw_imgs, filenames


def temporal_average(imgs_tmp, n):
    """
    Function that does a temporal average over N consecutive images
    Warning: if the number of images is not a multiple of the kernel size, the last images will be lost.
    
    Inputs:
        imgs_tmp: List of images to average
        n: Number of images to average over

    Returns:
        imgs_average: List of images averaged
    """
    #for img in imgs_tmp:
    #    print(img.shape)

    if len(imgs_tmp) < n:
        raise ValueError(f"The number of images should be at least {n}")
    
    number_chunk = len(imgs_tmp) // n
    array_imgs = np.array(imgs_tmp)

    imgs_average = []

    for i in range(number_chunk):
        sum_img = np.zeros(imgs_tmp[0].shape, dtype=np.float16)
        for j in range(n):
            sum_img += array_imgs[i*n+j]
        imgs_average.append(np.divide(sum_img,n).astype(np.uint8))

    # Display averaged images
    #for idx, img in enumerate(imgs_average):
    #    cv2.imshow(f"Averaged Image {idx}", cv2.resize(img, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    #    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    return imgs_average

def median_filtering(imgs_tmp, kernel_size):
    """
    Function that apply a median filter to images

    Inputs:
        imgs_tmp: List of images to apply filters to
        kernel_size: Size of the median filter

    Returns:
        imgs_average: List of images filtered
    """
        
    filtered_imgs = [cv2.medianBlur(img, kernel_size) for img in imgs_tmp]

    # Display averaged images
    #for idx, img in enumerate(filtered_imgs):
    #    cv2.imshow(f"Averaged Image {idx}", cv2.resize(img, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    #    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    return filtered_imgs

def clahe(imgs_tmp, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """
    Function that applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the images.

    Inputs:
        imgs_tmp: List of images to enhance contrast
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        proc_imgs: List of images processed
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    proc_imgs = [clahe.apply(img) for img in imgs_tmp]

    # Display processed images
    #for idx, img in enumerate(proc_imgs):
    #    cv2.imshow(f"CLAHE Image {idx}", cv2.resize(img, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    #    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    return proc_imgs


def processing(imgs_tmp):
    """
    Process the images before NPs counting

    Input:
        imgs_tmp: List of images to process

    Returns:
        proc_imgs: List of images processed
    """    
    # Remove noise
    averaged_imgs = temporal_average(imgs_tmp, N_TEMP_AVERAGE)
    cv2.imshow(f"Averaged Image", cv2.resize(averaged_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    filtered_imgs = median_filtering(averaged_imgs, MEDIAN_KERNEL_SIZE)
    cv2.imshow(f"Filtered Image", cv2.resize(filtered_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    
    # Increase sharpness
    #TODO

    # Increase contrast
    proc_imgs = clahe(filtered_imgs)
    cv2.imshow(f"Clahe Image", cv2.resize(proc_imgs[3], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    
    return proc_imgs

def main():
    """
    Main function where the core logic of the script is placed.
    """
    
    # Open raw images
    start_time = time.time()  # Start the timer

    raw_imgs, filenames = Open_images(RAW_IMAGES_PATH)
    #print(filenames)

    end_time = time.time()  # End the timer
    print(f"Execution time Open images: {end_time - start_time:.4f} seconds")

    # Show raw images
    #cv2.imshow("First Image", cv2.resize(raw_imgs[0], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    #cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    #Processing
    proc_imgs = processing(raw_imgs)

    pass

if __name__ == "__main__":
    main()  
