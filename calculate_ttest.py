import csv
import re
import numpy as np
from scipy import stats

def parse_array_string(s):
    # Remove brackets and split by whitespace
    s = s.strip('[]')
    # Handle multiple spaces and newlines if any
    values = re.split(r'\s+', s.strip())
    return np.array([float(v) for v in values if v])

def read_results(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row['seed'])
            recall = parse_array_string(row['recall'])
            ndcg = parse_array_string(row['ndcg'])
            data[seed] = {'recall': recall, 'ndcg': ndcg}
    return data

def main():
    file1 = 'seed_results.csv'
    file2 = 'seed_results_seg.csv'
    
    try:
        data1 = read_results(file1)
        data2 = read_results(file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Find common seeds
    seeds = sorted(list(set(data1.keys()) & set(data2.keys())))
    
    if not seeds:
        print("No common seeds found.")
        return
        
    print(f"Found {len(seeds)} common seeds: {seeds}")
    
    # Prepare arrays for t-test
    # Shape: (num_seeds, num_k)
    recall1 = np.array([data1[s]['recall'] for s in seeds])
    recall2 = np.array([data2[s]['recall'] for s in seeds])
    
    ndcg1 = np.array([data1[s]['ndcg'] for s in seeds])
    ndcg2 = np.array([data2[s]['ndcg'] for s in seeds])
    
    k_values = [5, 10, 20]
    
    print("\nPaired t-test Results (seed_results vs seed_results_seg):")
    print("-" * 40)
    print(f"{'Metric':<15} | {'p-value':<15}")
    print("-" * 40)
    
    # Recall
    for i, k in enumerate(k_values):
        if i < recall1.shape[1]:
            t_stat, p_val = stats.ttest_rel(recall1[:, i], recall2[:, i])
            print(f"Recall@{k:<7} | {p_val:<15.6f}")
            
    # NDCG
    for i, k in enumerate(k_values):
        if i < ndcg1.shape[1]:
            t_stat, p_val = stats.ttest_rel(ndcg1[:, i], ndcg2[:, i])
            print(f"NDCG@{k:<9} | {p_val:<15.6f}")

if __name__ == "__main__":
    main()
