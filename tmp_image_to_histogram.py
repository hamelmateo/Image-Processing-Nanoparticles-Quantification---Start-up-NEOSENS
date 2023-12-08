import matplotlib.pyplot as plt
import cv2

# Load an image
image_path = "C:\\Users\\hamel\\OneDrive - Neosens Diagnostics\\04_Technology_Software\\Image Processing\\NEOSENS\\01_images_raw\\00500_A3.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculate the histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure(figsize=(10, 4))
plt.title('Histogram of the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.xlim([0, 256])

# Save the histogram plot instead of showing it
plt.savefig('image_histogram.png')

# Close the plot to avoid displaying it in the notebook output
plt.close()




