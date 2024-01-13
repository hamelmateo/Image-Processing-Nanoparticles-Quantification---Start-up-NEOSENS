import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load an image
image_path = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\01_images_raw\\averaged_frame_1.jpg"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# Create a square mask
f = 160
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-f:crow+f, ccol-f:ccol+f] = 1  # The size of the square is 60x60

# Apply mask to the frequency domain
f_shift_masked = f_shift #* mask
magnitude_spectrum_masked = 20 * np.log(np.abs(f_shift_masked))

# Inverse FFT to get the filtered image back in the spatial domain
f_ishift_masked = np.fft.ifftshift(f_shift_masked)
image_filtered = np.fft.ifft2(f_ishift_masked)
image_filtered = np.abs(image_filtered)


# Original Image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.savefig("original_image.png")

# Original Frequency Domain
plt.figure(figsize=(6, 6))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Original Frequency Domain')
plt.colorbar()
plt.savefig("original_frequency_domain.png")

# Masked Frequency Domain
plt.figure(figsize=(6, 6))
plt.imshow(magnitude_spectrum_masked, cmap='gray')
plt.title('Masked Frequency Domain Representation')
plt.colorbar(shrink=0.59)
plt.savefig("masked_frequency_domain.png")

# Filtered Image
plt.figure(figsize=(12, 12))
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtered Image')
plt.colorbar()
plt.savefig("filtered_image.png")

