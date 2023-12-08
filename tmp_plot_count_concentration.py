import matplotlib.pyplot as plt
import numpy as np

# Concentrations in ng/mL (assuming a consistent order with the count data)
concentrations = np.array([0, 0, 0, 0, 50, 50, 50, 50, 50, 
                           100, 100, 100, 100, 100, 300, 300, 300, 300, 300, 
                           500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 
                           5000, 5000, 5000, 5000, 5000, 12000, 12000, 12000, 12000, 12000, 
                           20000, 20000, 20000, 20000, 20000])

# Counts for each image
counts = np.array([1019, 1145, 1084, 1023, 1087, 49, 769, 29, 465, 206, 415, 98, 298, 350, 1457, 
    1945, 1674, 1693, 347, 570, 1595, 1339, 927, 1634, 5122, 5828, 4196, 6904, 
    5445, 6067, 6647, 6865, 4146, 5964, 5833, 4403, 6315, 5299, 6017, 8838, 
    7489, 8262, 11103, 12434, 8453, 8419])

# Calculate the mean and standard deviation for each concentration
unique_concentrations = np.unique(concentrations)
mean_counts = [np.mean(counts[concentrations == uc]) for uc in unique_concentrations]
std_counts = [np.std(counts[concentrations == uc]) for uc in unique_concentrations]

# Plotting the individual counts
plt.scatter(concentrations, counts, label='Individual Counts', alpha=0.6, s=4, c='blue')

# Plotting the mean counts with error bars for the standard deviation
plt.errorbar(unique_concentrations, mean_counts, yerr=std_counts, fmt='o', color='r', 
             label='Mean Count with STD', ecolor='red', elinewidth=1, capsize=4, ms=3)

# Labels and Title
plt.xlabel('Concentration (ng/mL)')
plt.ylabel('Count')
plt.title('Global Median Threshold - Results')
plt.legend()

# Show the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('counts_vs_concentration.png', format='png')