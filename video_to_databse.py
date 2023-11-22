"""
Created on Tue Nov 21 16:59:45 2023

@author: Mateo HAMEL
"""

try:
    # Standard Library Imports
    import os
    from typing import List, Tuple

    # Third-party Library Imports
    from moviepy.editor import VideoFileClip
    import numpy as np
    import cv2

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Config variables
VIDEO_DIRECTORY = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\Pictures_Database_11-21-2023\\live_acquisition_unfonctionnalized.mp4"
RAW_IMAGES_DIRECTORY = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\Pictures_Database_11-21-2023\\Raw_images"
AVERAGED_IMAGES_DIRECTORY = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\Pictures_Database_11-21-2023\\Averaged_Images"
TEMPORAL_AVERAGE_WINDOW_SIZE = 50
TEMPORAL_AVERAGE_BATCH_SIZE = 200

def cut_video(video_path: str, edited_video_path: str):
    """
    Cut the video at the desired time
    
    Args:
    video_path (str): Path to the video to edit
    edited_video_path (str): Path & name of the edited video
    """
    # Load your video
    video = VideoFileClip(video_path)

    # The time is specified in seconds
    cut_video = video.subclip(0, 36*60 + 30)

    # Write the result to a new file
    cut_video.write_videofile(edited_video_path, codec='libx264', audio_codec='aac')


def video_raw_images(video_path: str, ouput_folder: str):
    """
    Retrieve raw images from a video
    
    Args:
    video_path (str): Path to the video
    output_folder (str): Path of the folder to save retrieved images
    """

    # Load your video
    video = VideoFileClip(video_path)

    # Folder to save the frames
    if not os.path.exists(ouput_folder):
        os.makedirs(ouput_folder)

    # Extract frames and their timestamps
    fps = video.fps  # Frames per second
    for i, frame in enumerate(video.iter_frames()):
        # Calculate the time of the current frame in the video
        frame_time = i / fps
        minutes = int((frame_time % 3600) // 60)
        seconds = frame_time % 60

        # Format the time in "mm-ss" & frame number
        time_str = f"{minutes:02d}-{seconds:.2f}"
        frame_number_str = f"{i+1:05d}"

        # Save the frame with the timestamp in the filename
        frame_path = os.path.join(ouput_folder, f"frame_{frame_number_str}_{time_str}.jpg")
        video.save_frame(frame_path, t=frame_time)


def compute_temporal_average(images: List[np.ndarray], window_size: int) -> List[np.ndarray]:
    """
    Compute the temporal average of a list of images over a specified window size.
    
    Args:
        images (List[np.ndarray]): List of images to average.
        window_size (int): Number of images to average over.

    Returns:
        List[np.ndarray]: List of temporally averaged images.
    """
    if not images:
        raise ValueError("The list of images is empty.")
    if not all(isinstance(image, np.ndarray) for image in images):
        raise TypeError("All elements in the images list should be NumPy arrays.")
    if window_size <= 0:
        raise ValueError("Window size should be a positive integer.")
    if len(images) < window_size:
        raise ValueError(f"The number of images should be at least {window_size}")
    
    number_chunk = len(images) // window_size
    array_imgs = np.array(images)

    imgs_average = []

    for i in range(number_chunk):
        sum_img = np.zeros(images[0].shape, dtype=np.float16)
        for j in range(window_size):
            sum_img += array_imgs[i * window_size + j]
        imgs_average.append(np.divide(sum_img, window_size).astype(np.uint8))

    return imgs_average


def load_images_in_batches(directory: str, batch_size: int) -> List[np.ndarray]:
    """
    Generator to load images in batches.

    Args:
    directory (str): Path to the directory containing images.
    batch_size (int): Number of images to load in each batch.
    
    Returns:
    List[np.ndarray]: A batch of images.
    """
    image_files = sorted(f for f in os.listdir(directory) if f.endswith('.jpg'))
    for i in range(0, len(image_files), batch_size):
        batch = [cv2.imread(os.path.join(directory, f)) for f in image_files[i:i + batch_size]]
        yield batch


def process_video(video_path: str, raw_images_folder: str, averaged_images_folder: str, window_size: int, batch_size: int):
    """
    Process a video by extracting frames, computing temporal average, and saving results.
    
    Args:
        video_path (str): Path to the video.
        raw_images_folder (str): Folder to save raw images.
        averaged_images_folder (str): Folder to save averaged images.
        window_size (int): Window size for temporal averaging.
        batch_size (int): Batch size for processing images.
    """

    # Step 1: Extract raw images from the video
    #video_raw_images(video_path, raw_images_folder)
    #print(" Raw images extraction done")

    # Step 2: Load images in batches and compute temporal average
    counter = 0
    for batch in load_images_in_batches(raw_images_folder, batch_size):
        averaged_images = compute_temporal_average(batch, window_size)
        for i, img in enumerate(averaged_images):
            counter += 1
            cv2.imwrite(os.path.join(averaged_images_folder, f"averaged_frame_{counter}.jpg"), img)
    print("Temporal averages done")


def main():
    """
    Main function to process a database from the video.
    """
    
    # Parameters
    video_path = VIDEO_DIRECTORY
    raw_images_folder = RAW_IMAGES_DIRECTORY
    averaged_images_folder = AVERAGED_IMAGES_DIRECTORY
    window_size = TEMPORAL_AVERAGE_WINDOW_SIZE
    batch_size = TEMPORAL_AVERAGE_BATCH_SIZE

    # Ensure output directories exist
    if not os.path.exists(raw_images_folder):
        os.makedirs(raw_images_folder)
    if not os.path.exists(averaged_images_folder):
        os.makedirs(averaged_images_folder)

    # Process the video
    process_video(video_path, raw_images_folder, averaged_images_folder, window_size, batch_size)


if __name__ == "__main__":
    main()
