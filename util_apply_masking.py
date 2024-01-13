"""
Created on Wed Oct 25 17:13:01 2023
@author: Mateo HAMEL
"""

try:
    import cv2
    import numpy as np

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


def create_and_save_circle_roi(image_path: str, mask_save_path: str, masked_image_save_path, roi_radius: int):
    """
    Create a circular ROI on an image, save the mask, and apply the mask to the image.

    Args:
        image_path (str): Path to the input image.
        mask_save_path (str): Path to save the mask.
        roi_radius (int): Radius of the circular ROI.
    
    Returns:
        masked_image (np.ndarray): Masked image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(mask, (x, y), roi_radius, 255, -1)
            param['finished'] = True

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_circle, {'finished': False})
    
    while True:
        combined = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.imshow('Image', combined)
        if cv2.waitKey(20) & 0xFF == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()

    # Save the mask
    cv2.imwrite(mask_save_path, mask)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(masked_image_save_path, masked_image)

    return masked_image

# Example usage
image_path = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\01_images_raw\\05000_A3.png"  # Replace with your image path
mask_save_path = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\mask_05000_A3.png"
masked_image_save_path = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\masked_05000_A3.png" # Replace with your masked image save path
roi_radius = 85  # Set the radius of the ROI

masked_image = create_and_save_circle_roi(image_path, mask_save_path, masked_image_save_path, roi_radius)

# Optionally, save or display the masked image
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
