import numpy as np
from scipy.stats import skew
import csv



lengths = [4, 8, 16, 32, 64]
parallels = [1, 2, 4, 8, 16]
windows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def compute_hurst(ts):
    
  
    N = len(ts)
    mean_ts = np.mean(ts)
    Y = np.cumsum(ts - mean_ts)
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = np.array([np.std(ts[:i+1]) for i in range(N)])
    RS = R[1:] / (S[1:] + 1e-8)  # avoid division by zero
    log_RS = np.log(RS)
    log_n = np.log(np.arange(2, N+1))
    H, _ = np.polyfit(log_n, log_RS, 1)
    return H

def global_feature_vector(X, normalize=True, method="zscore"):

    X = np.array(X, dtype=float)
    mean_val = np.mean(X)
    std_val = np.std(X)
    range_val = np.max(X) - np.min(X)
    skewness_val = skew(X)
    hurst_val = compute_hurst(X)
    
    fv = np.array([mean_val, std_val, range_val, skewness_val, hurst_val], dtype=float)

    if normalize:
        if method == "zscore":
            fv = (fv - np.mean(fv)) / (np.std(fv) + 1e-8)
        elif method == "minmax":
            fv = (fv - np.min(fv)) / (np.max(fv) - np.min(fv) + 1e-8)
        else:
            raise ValueError("Unknown normalization method. Use 'zscore' or 'minmax'.")
    
    return fv


fv_list = []

for length in lengths:
    for parallel in parallels:
        for window in windows:
            tcpdump_file = f"tcp_l{length}_p{parallel}_w{window}.txt"
            try:
                with open(tcpdump_file, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ')
                    X = []
                    for row in reader:
                        if len(row) >= 2:  
                            try:
                                X.append(float(row[-1]))  # take last column as numeric value
                            except ValueError:
                                continue  # skip rows that can't be converted
                    if len(X) > 0:
                        fv = global_feature_vector(X, normalize=True, method="minmax")
                        fv_list.append(fv)
            except FileNotFoundError:
                print(f"File {tcpdump_file} not found. Skipping...")


fv_norm = np.array(fv_list)
fv_reshaped = fv_norm.reshape(-1, 5)

print("fv_reshaped shape:", fv_reshaped.shape) 


# Save into CSV file
output_file = "fv_reshaped.csv"
np.savetxt(output_file, fv_reshaped, delimiter=",", fmt="%.6f",
           header="mean,std,range,skewness,hurst", comments="")
