'''
Randomally pick a subset of the dataset for the research
'''

import pandas as pd

# File paths
# Use the absolute path
input_file = r"C:\Users\shaha\Projects\Python Projects\Israel-Palestine-Political-Affiliation-Text-Classification\Data\full_dataset_untagged.csv"
output_file = r"C:\Users\shaha\Projects\Python Projects\Israel-Palestine-Political-Affiliation-Text-Classification\Data\research_data.csv"

# Columns to keep
columns_to_keep = ["comment_id", "self_text", "created_time", "subreddit", "post_id", "author_name"]

# Load the data
df = pd.read_csv(input_file, usecols=columns_to_keep)

# Sample 1.5% of the data randomly
sampled_df = df.sample(frac=0.015, random_state=42)  # Set random_state for reproducibility

# Save to a new CSV file
sampled_df.to_csv(output_file, index=False)

print(f"Sampled data saved to {output_file}.")