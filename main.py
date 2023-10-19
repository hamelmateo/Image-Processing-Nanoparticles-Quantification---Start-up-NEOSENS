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

def Open_images(path):
    """
    Open images from the folder raw_images and put them in grayscale for faster computing

    Input:
        path: path of folder raw_images
    
    Returns:
        raw_imgs: List of images
        filename: List of the name of the images 
    """

    filenames = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    raw_imgs = [cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE) for f in filenames]
    
    return raw_imgs, filenames


def processing(imgs_tmp):
    """
    Process the images before NPs counting

    Input:
        raw_imgs: List of images to process

    Returns:
        proc_imgs: List of images processed
    """    
    # Remove noise
    proc_imgs = imgs_tmp

    # Increase sharpness


    # Increase contrast

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
    cv2.imshow("First Image", cv2.resize(raw_imgs[0], None, fx=0.20, fy=0.20, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    #Processing
    proc_imgs = processing(raw_imgs)

    pass

if __name__ == "__main__":
    main()  
