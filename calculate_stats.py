import pandas as pd
import numpy as np
import re

def parse_array_string(s):
    # Remove brackets and split by whitespace
    s = s.strip('[]')
    # Handle multiple spaces and newlines if any
    values = re.split(r'\s+', s.strip())
    return np.array([float(v) for v in values if v])

def main():
    df = pd.read_csv('seed_results.csv')
    
    # Parse the string representations of arrays
    recall_arrays = np.stack(df['recall'].apply(parse_array_string).values)
    ndcg_arrays = np.stack(df['ndcg'].apply(parse_array_string).values)
    
    # Calculate mean and std
    recall_mean = np.mean(recall_arrays, axis=0)
    recall_std = np.std(recall_arrays, axis=0)
    
    ndcg_mean = np.mean(ndcg_arrays, axis=0)
    ndcg_std = np.std(ndcg_arrays, axis=0)
    
    # Assuming k values are [5, 10, 20] based on previous context
    k_values = [5, 10, 20]
    
    print("Recall Stats:")
    for i, k in enumerate(k_values):
        if i < len(recall_mean):
            print(f"Recall@{k}: Mean = {recall_mean[i]:.6f}, Std = {recall_std[i]:.6f}")
            
    print("\nNDCG Stats:")
    for i, k in enumerate(k_values):
        if i < len(ndcg_mean):
            print(f"NDCG@{k}:   Mean = {ndcg_mean[i]:.6f}, Std = {ndcg_std[i]:.6f}")

if __name__ == "__main__":
    main()
