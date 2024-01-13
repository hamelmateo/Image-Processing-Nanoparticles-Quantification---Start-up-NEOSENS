"""
Created on Wed Oct 25 17:13:01 2023
@author: Mateo HAMEL
"""

try:
    import matplotlib.pyplot as plt
    import numpy as np

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Data
frame_numbers = list(range(1, 27))
white_pixel_counts = [16, 7, 11, 8, 18, 12, 11, 2, 9, 8, 4, 10, 6, 9, 4, 4, 5, 0, 2, 0, 0, 0, 3, 0, 1, 0]

average_white_pixel_count = np.mean(white_pixel_counts)

# Convert frame numbers to time in seconds (1 frame = 50 seconds)
times = [30 * frame for frame in frame_numbers]


# Re-plotting with the fitted line
plt.figure(figsize=(12, 6))
plt.plot(times, white_pixel_counts, marker='o', label="Data")
plt.axhline(y=average_white_pixel_count, color='red', linestyle='-', label=f'Average count = {average_white_pixel_count:.2f}')
plt.title("White Pixel Count vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("White Pixel Count")
plt.legend()
plt.grid(True)
plt.savefig('timeresolvedplot.png', format='png')
