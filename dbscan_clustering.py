import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('fv_reshaped.csv')


features = ['mean', 'std', 'skewness', 'hurst']
X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def robust_find_optimal_eps(X, k=4):
    """Robust method to find optimal epsilon"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Data Points Sorted by Distance')
    plt.ylabel(f'{k}-Distance')
    plt.title('K-Distance Graph for DBSCAN (Optimal EPS)')
    plt.grid(True, alpha=0.3)
    
    
    optimal_eps = np.percentile(distances, 85)  
    
    plt.axhline(y=optimal_eps, color='r', linestyle='--', 
                label=f'Suggested EPS: {optimal_eps:.3f} (85th percentile)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dbscan_eps_optimization.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_eps


print("Finding optimal DBSCAN parameters...")
optimal_eps = robust_find_optimal_eps(X_scaled, k=4)
print(f"Suggested EPS value: {optimal_eps:.3f}")


dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
dbscan_cluster_labels = dbscan.fit_predict(X_scaled)


df['DBSCAN_Cluster'] = dbscan_cluster_labels


n_clusters = len(set(dbscan_cluster_labels)) - (1 if -1 in dbscan_cluster_labels else 0)
n_noise = list(dbscan_cluster_labels).count(-1)

print(f"Initial DBSCAN Results: {n_clusters} clusters found, {n_noise} noise points")


if n_clusters < 2 or n_noise > len(df) * 0.7:
    print("Poor DBSCAN results detected. Trying alternative parameters...")
    
  
    eps_candidates = [optimal_eps * 0.5, optimal_eps * 0.7, optimal_eps * 1.2, optimal_eps * 1.5]
    best_eps = optimal_eps
    best_clusters = n_clusters
    best_noise = n_noise
    
    for eps_candidate in eps_candidates:
        dbscan_test = DBSCAN(eps=eps_candidate, min_samples=5)
        labels_test = dbscan_test.fit_predict(X_scaled)
        clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        noise_test = list(labels_test).count(-1)
        
        print(f"  EPS {eps_candidate:.3f}: {clusters_test} clusters, {noise_test} noise")
        
        if 2 <= clusters_test <= 10 and noise_test < best_noise:
            best_eps = eps_candidate
            best_clusters = clusters_test
            best_noise = noise_test
            dbscan_cluster_labels = labels_test
    
    if best_clusters > n_clusters or best_noise < n_noise:
        print(f"Better parameters found: EPS={best_eps:.3f}")
        optimal_eps = best_eps
        n_clusters = best_clusters
        n_noise = best_noise
        dbscan_cluster_labels = dbscan_cluster_labels
        df['DBSCAN_Cluster'] = dbscan_cluster_labels

print(f"Final DBSCAN Results: {n_clusters} clusters found, {n_noise} noise points")


dbscan_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'yellow', 'pink']
noise_color = 'gray'


plt.figure(figsize=(14, 8))


fv_ids = np.arange(len(df))


all_cluster_ids = sorted(set(dbscan_cluster_labels))


for cluster_id in all_cluster_ids:
    mask = dbscan_cluster_labels == cluster_id
    if cluster_id == -1:
       
        plt.scatter(fv_ids[mask], [cluster_id] * np.sum(mask), 
                    c=noise_color, marker='x', s=80, alpha=0.8,
                    label='Noise')
    else:
      
        color_idx = cluster_id % len(dbscan_colors)
        plt.scatter(fv_ids[mask], [cluster_id] * np.sum(mask), 
                    c=dbscan_colors[color_idx], marker='o', s=100, alpha=0.7,
                    label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title(f'DBSCAN Clustering: Cluster ID vs FV ID\n({n_clusters} clusters, {n_noise} noise points)', 
          fontsize=14, fontweight='bold')


y_ticks = all_cluster_ids
y_labels = ['Noise' if x == -1 else f'Cluster {x}' for x in all_cluster_ids]

plt.yticks(y_ticks, y_labels)
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clustering_results_dbscan.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(16, 6))


for cluster_id in all_cluster_ids:
    mask = dbscan_cluster_labels == cluster_id
    fv_ids_cluster = fv_ids[mask]
    
    
    jitter = np.random.normal(0, 0.05, len(fv_ids_cluster))
    
    if cluster_id == -1:
       
        plt.scatter(fv_ids_cluster, [-1 + jitter[i] for i in range(len(fv_ids_cluster))], 
                    c=noise_color, marker='x', s=60, alpha=0.8,
                    label='Noise')
    else:
        
        color_idx = cluster_id % len(dbscan_colors)
        plt.scatter(fv_ids_cluster, [cluster_id + jitter[i] for i in range(len(fv_ids_cluster))], 
                    c=dbscan_colors[color_idx], marker='o', s=80, alpha=0.8,
                    label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title(f'DBSCAN Clustering: 1D Visualization - Cluster ID vs FV ID\n({n_clusters} clusters, {n_noise} noise points)', 
          fontsize=14, fontweight='bold')


plt.yticks(y_ticks, y_labels)
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clustering_results_1D_dbscan.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 6))


cluster_counts = {}
for cluster_id in all_cluster_ids:
    count = np.sum(dbscan_cluster_labels == cluster_id)
    cluster_counts[cluster_id] = count


sorted_clusters = sorted(cluster_counts.keys())
cluster_names = ['Noise' if x == -1 else f'Cluster {x}' for x in sorted_clusters]
counts = [cluster_counts[x] for x in sorted_clusters]


bar_colors = []
for cluster_id in sorted_clusters:
    if cluster_id == -1:
        bar_colors.append(noise_color)
    else:
        color_idx = cluster_id % len(dbscan_colors)
        bar_colors.append(dbscan_colors[color_idx])

bars = plt.bar(range(len(sorted_clusters)), counts, 
               color=bar_colors, alpha=0.7, edgecolor='black')

plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
plt.ylabel('Number of Feature Vectors', fontsize=12, fontweight='bold')
plt.title(f'DBSCAN: Feature Vector Distribution per Cluster\n({n_clusters} clusters, {n_noise} noise points)', 
          fontsize=14, fontweight='bold')
plt.xticks(range(len(sorted_clusters)), cluster_names)


for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('cluster_distribution_dbscan.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 8))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


for cluster_id in all_cluster_ids:
    if cluster_id == -1:
        continue  
    mask = dbscan_cluster_labels == cluster_id
    color_idx = cluster_id % len(dbscan_colors)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=dbscan_colors[color_idx], marker='o', s=60, alpha=0.7,
                label=f'Cluster {cluster_id}')


noise_mask = dbscan_cluster_labels == -1
if np.sum(noise_mask) > 0:
    plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                c=noise_color, marker='x', s=50, alpha=0.8,
                label='Noise')

plt.xlabel('First Principal Component', fontsize=12, fontweight='bold')
plt.ylabel('Second Principal Component', fontsize=12, fontweight='bold')
plt.title(f'DBSCAN Clustering: PCA Visualization\n({n_clusters} clusters, {n_noise} noise points)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pca_visualization_dbscan.jpg', dpi=300, bbox_inches='tight')
plt.show()


print("DBSCAN Clustering Results Summary:")
print("=" * 50)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"EPS parameter used: {optimal_eps:.3f}")
print(f"Min samples parameter: 5")

for cluster_id in sorted(all_cluster_ids):
    if cluster_id == -1:
        cluster_data = df[df['DBSCAN_Cluster'] == cluster_id]
        print(f"\nNoise points: {len(cluster_data)} feature vectors")
        if len(cluster_data) > 0:
            print(f"  Mean range: {cluster_data['mean'].min():.3f} - {cluster_data['mean'].max():.3f}")
            print(f"  Std range:  {cluster_data['std'].min():.3f} - {cluster_data['std'].max():.3f}")
    else:
        cluster_data = df[df['DBSCAN_Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_data)} feature vectors")
        print(f"  Mean range: {cluster_data['mean'].min():.3f} - {cluster_data['mean'].max():.3f}")
        print(f"  Std range:  {cluster_data['std'].min():.3f} - {cluster_data['std'].max():.3f}")


print("\nDBSCAN Clustering Statistics:")
print("=" * 50)


dbscan_cluster_stats = []


noise_data = df[df['DBSCAN_Cluster'] == -1]
if len(noise_data) > 0:
    fv_ids_noise = noise_data.index.tolist()
    fv_min_noise = min(fv_ids_noise) if len(fv_ids_noise) > 0 else 0
    fv_max_noise = max(fv_ids_noise) if len(fv_ids_noise) > 0 else 0
    
    dbscan_cluster_stats.append({
        'Cluster': 'Noise',
        'Count': len(noise_data),
        'Mean Avg': noise_data['mean'].mean() if len(noise_data) > 0 else 0,
        'Std Avg': noise_data['std'].mean() if len(noise_data) > 0 else 0,
        'FV ID Range': f"{fv_min_noise}-{fv_max_noise}" if len(fv_ids_noise) > 0 else "N/A"
    })


for cluster_id in sorted(all_cluster_ids):
    if cluster_id == -1:
        continue
        
    cluster_data = df[df['DBSCAN_Cluster'] == cluster_id]
    
    
    count = len(cluster_data)
    mean_avg = cluster_data['mean'].mean()
    std_avg = cluster_data['std'].mean()
    
   
    fv_ids_cluster = cluster_data.index.tolist()
    fv_min = min(fv_ids_cluster)
    fv_max = max(fv_ids_cluster)
    fv_range = f"{fv_min}-{fv_max}"
    
    
    dbscan_cluster_stats.append({
        'Cluster': f'Cluster {cluster_id}',
        'Count': count,
        'Mean Avg': mean_avg,
        'Std Avg': std_avg,
        'FV ID Range': fv_range
    })
    
   
    print(f"Cluster {cluster_id}\t| {count}\t| {mean_avg:.4f}\t| {std_avg:.4f}\t| {fv_range}")


if len(noise_data) > 0:
    print(f"Noise\t\t| {len(noise_data)}\t| {noise_data['mean'].mean():.4f}\t| {noise_data['std'].mean():.4f}\t| {fv_min_noise}-{fv_max_noise}")


dbscan_stats_df = pd.DataFrame(dbscan_cluster_stats)
print("\n" + "=" * 50)
print("Formatted DBSCAN Clustering Statistics:")
print("=" * 50)
print(dbscan_stats_df.to_string(index=False))

# Save statistics to CSV
dbscan_stats_df.to_csv('dbscan_cluster_statistics.csv', index=False)
print(f"\nStatistics saved to 'dbscan_cluster_statistics.csv'")

