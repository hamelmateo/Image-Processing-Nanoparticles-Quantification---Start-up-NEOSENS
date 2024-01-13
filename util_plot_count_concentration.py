"""
Created on Wed Oct 25 17:13:01 2023
@author: Mateo HAMEL
"""

try:
    import matplotlib.pyplot as plt
    import numpy as np

except ImportError as e:
    raise ImportError(f"Required modules are missing. {e}")


# Concentrations in ng/mL (assuming a consistent order with the count data)
concentrations = np.array([0, 0, 0, 0, 50, 50, 50, 50, 50, 
                           100, 100, 100, 100, 100, 300, 300, 300, 300, 300, 
                           500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 
                           5000, 5000, 5000, 5000, 5000, 12000, 12000, 12000, 12000, 12000, 
                           20000, 20000, 20000, 20000, 20000])

# Counts for each image
counts = np.array([3, 34, 28, 30, 32, 24, 39, 26, 31, 25, 47, 33, 42, 32, 95, 136, 120, 84, 72, 168, 146, 
                   279, 136, 96, 395, 467, 200, 239, 264, 370, 280, 989, 924, 1132, 755, 469, 1052, 
                   893, 828, 1554, 1166, 1674, 1559, 1970, 1485, 1935])

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