import numpy as np
import csv
from scipy.stats import skew

lengths = [4, 8, 16, 32, 64]
parallels = [1, 2, 4, 8, 16]

def compute_hurst(ts):
    """
    Estimate the Hurst exponent using R/S method.
    Returns 0.0 if computation is not possible.
    """
    N = len(ts)
    if N < 20:  
        return 0.0
    
    mean_ts = np.mean(ts)
    Y = np.cumsum(ts - mean_ts)
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = np.array([np.std(ts[:i+1]) for i in range(N)])
    
    RS = R[1:] / (S[1:] + 1e-8)
    if np.any(RS <= 0):
        return 0.0
    
    log_RS = np.log(RS)
    log_n = np.log(np.arange(2, N+1))
    try:
        H, _ = np.polyfit(log_n, log_RS, 1)
        return float(H)
    except Exception:
        return 0.0

def udp_feature_vector(X):
    """
    UDP feature vector: mean, std, range, skewness, hurst
    """
    X = np.array(X, dtype=float)
    if len(X) < 5:
        return np.zeros(5)

    mean_val = np.mean(X)
    std_val = np.std(X)
    range_val = np.max(X) - np.min(X)
    skewness_val = skew(X) if np.std(X) > 0 else 0.0
    hurst_val = compute_hurst(X)

    fv = np.array([mean_val, std_val, range_val, skewness_val, hurst_val], dtype=float)
    fv = np.nan_to_num(fv) 
    return fv

# Collect all UDP feature vectors
fv_list_udp = []

for length in lengths:
    for parallel in parallels:
        udp_file = f"udp_l{length}_p{parallel}.txt"
        try:
            with open(udp_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                X = []
                for row in reader:
                    if len(row) >= 2:
                        try:
                            X.append(float(row[-1]))
                        except ValueError:
                            continue
                if len(X) > 0:
                    fv = udp_feature_vector(X)
                    fv_list_udp.append(fv)
        except FileNotFoundError:
            print(f"File {udp_file} not found. Skipping...")

fv_array = np.array(fv_list_udp)


fv_min = fv_array.min(axis=0)
fv_max = fv_array.max(axis=0)
fv_norm = (fv_array - fv_min) / (fv_max - fv_min + 1e-8)

print("fv_norm shape:", fv_norm.shape)

# Save to CSV
output_file = "fv_udp.csv"
np.savetxt(output_file, fv_norm, delimiter=",", fmt="%.6f",
           header="mean,std,range,skewness,hurst", comments="")
