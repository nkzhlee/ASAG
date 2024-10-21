import pandas as pd
import os

# File paths
data_file = '/data/smk6961/general_analysis/test_set_with_predictions.csv'
output_dir = '/data/smk6961/general_analysis/low_jaccard'

# Load the data
data = pd.read_csv(data_file)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to save filtered data into CSV files
def save_csv(filtered_data, file_name):
    file_path = os.path.join(output_dir, file_name)
    filtered_data.to_csv(file_path, index=False)

# Iterate over all possible confusion matrix cells (T = {0,1,2}, P = {0,1,2})
for true_label in [0, 1, 2]:
    for asrrn_pred in [0, 1, 2]:
        for sfrn_pred in [0, 1, 2]:
            
            # Define overlap condition (both models predict the same)
            overlap = data[(data['True Label'] == true_label) & 
                           (data['AsRRN Prediction'] == asrrn_pred) & 
                           (data['SFRN Prediction'] == sfrn_pred)]
            
            # Define AsRRN disjoint condition (AsRRN makes the given prediction, but SFRN doesn't)
            asrrn_disjoint = data[(data['True Label'] == true_label) & 
                                  (data['AsRRN Prediction'] == asrrn_pred) & 
                                  (data['SFRN Prediction'] != sfrn_pred)]
            
            # Define SFRN disjoint condition (SFRN makes the given prediction, but AsRRN doesn't)
            sfrn_disjoint = data[(data['True Label'] == true_label) & 
                                 (data['SFRN Prediction'] == sfrn_pred) & 
                                 (data['AsRRN Prediction'] != asrrn_pred)]
            
            # Save overlap data
            if not overlap.empty:
                overlap_file = f'T{true_label}P{asrrn_pred}_overlap.csv'
                save_csv(overlap, overlap_file)
            
            # Save AsRRN disjoint data
            if not asrrn_disjoint.empty:
                asrrn_disjoint_file = f'T{true_label}P{asrrn_pred}_asrrn_disjoint.csv'
                save_csv(asrrn_disjoint, asrrn_disjoint_file)
            
            # Save SFRN disjoint data
            if not sfrn_disjoint.empty:
                sfrn_disjoint_file = f'T{true_label}P{sfrn_pred}_sfrn_disjoint.csv'
                save_csv(sfrn_disjoint, sfrn_disjoint_file)

print("CSV files have been generated for all confusion matrix cells.")
