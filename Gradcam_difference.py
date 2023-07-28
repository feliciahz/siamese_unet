import cv2
import numpy as np
import os
import pandas as pd

# Get the list of heatmap files
heatmap_files = sorted([file for file in os.listdir(heatmap_dir) if file.endswith('.jpg')])

num_pairs = len(heatmap_files) // 2

# Initialize lists to store patient IDs, average intensities, differences, and percentages
patient_ids = []
average_intensities_t0 = []
average_intensities_t1 = []
differences = []
percentages = []

for i in range(num_pairs):
    # Load the heatmaps
    heatmap1 = cv2.imread(os.path.join(heatmap_dir, heatmap_files[2*i]), cv2.IMREAD_GRAYSCALE)
    heatmap2 = cv2.imread(os.path.join(heatmap_dir, heatmap_files[2*i+1]), cv2.IMREAD_GRAYSCALE)

    # Extract patient ID from filename
    patient_id = heatmap_files[2*i].split('_')[0]

    # Calculate the average intensity of each heatmap
    average_intensity1 = np.mean(heatmap1)
    average_intensity2 = np.mean(heatmap2)

    # Calculate the difference
    difference = average_intensity2 - average_intensity1  # t1 relative to t0

    # Calculate the percentage change
    percentage = (difference / average_intensity1) * 100 if average_intensity1 != 0 else 0

    # Store the results
    patient_ids.append(patient_id)
    average_intensities_t0.append(average_intensity1)
    average_intensities_t1.append(average_intensity2)
    differences.append(difference)
    percentages.append(percentage)

# Prepare a DataFrame and save to a CSV file
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Average_Intensity_T0': average_intensities_t0,
    'Average_Intensity_T1': average_intensities_t1,
    'Difference': differences,
    'Percentage': percentages
})
df.to_csv('heatmap_differences.csv', index=False)
