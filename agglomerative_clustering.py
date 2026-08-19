import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data
df = pd.read_csv('fv_reshaped.csv')


features = ['mean', 'std', 'skewness', 'hurst']
X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


optimal_k = 4
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_cluster_labels = agg_clustering.fit_predict(X_scaled)


df['Agglomerative_Cluster'] = agg_cluster_labels

plt.figure(figsize=(12, 8))


fv_ids = np.arange(len(df))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for cluster_id in range(optimal_k):
    mask = agg_cluster_labels == cluster_id
    plt.scatter(fv_ids[mask], [cluster_id] * np.sum(mask), 
                c=colors[cluster_id], marker='o', s=100, alpha=0.7,
                label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title('Agglomerative Clustering: Cluster ID vs FV ID', fontsize=14, fontweight='bold')


plt.yticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clustering_results_agglomerative.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 6))


for cluster_id in range(optimal_k):
    mask = agg_cluster_labels == cluster_id
    fv_ids_cluster = fv_ids[mask]
    
 
    jitter = np.random.normal(0, 0.1, len(fv_ids_cluster))
    
    plt.scatter(fv_ids_cluster, [cluster_id + jitter[i] for i in range(len(fv_ids_cluster))], 
                c=colors[cluster_id], marker='o', s=80, alpha=0.8,
                label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title('Agglomerative Clustering: 1D Visualization - Cluster ID vs FV ID', 
          fontsize=14, fontweight='bold')


plt.yticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clustering_results_1D_agglomerative.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
cluster_counts = df['Agglomerative_Cluster'].value_counts().sort_index()

bars = plt.bar(range(optimal_k), cluster_counts.values, 
               color=colors[:optimal_k], alpha=0.7, edgecolor='black')

plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
plt.ylabel('Number of Feature Vectors', fontsize=12, fontweight='bold')
plt.title('Agglomerative: Feature Vector Distribution per Cluster', fontsize=14, fontweight='bold')
plt.xticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])


for bar, count in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('cluster_distribution_agglomerative.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(15, 8))
linked = linkage(X_scaled, method='ward')

dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp',
           p=20)
plt.title('Agglomerative Clustering Dendrogram', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=linked[-optimal_k, 2], color='r', linestyle='--', label=f'Cut line for {optimal_k} clusters')
plt.legend()
plt.tight_layout()
plt.savefig('dendrogram_agglomerative.jpg', dpi=300, bbox_inches='tight')
plt.show()


print("Agglomerative Clustering Results Summary:")
print("=" * 50)
for cluster_id in range(optimal_k):
    cluster_data = df[df['Agglomerative_Cluster'] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_data)} feature vectors")
    print(f"  Mean range: {cluster_data['mean'].min():.3f} - {cluster_data['mean'].max():.3f}")
    print(f"  Std range:  {cluster_data['std'].min():.3f} - {cluster_data['std'].max():.3f}")
    print()


print("Agglomerative Clustering Statistics:")
print("=" * 50)


agg_cluster_stats = []

for cluster_id in range(optimal_k):
    cluster_data = df[df['Agglomerative_Cluster'] == cluster_id]
    
   
    count = len(cluster_data)
    mean_avg = cluster_data['mean'].mean()
    std_avg = cluster_data['std'].mean()
    
  
    fv_ids_cluster = cluster_data.index.tolist()
    fv_min = min(fv_ids_cluster)
    fv_max = max(fv_ids_cluster)
    fv_range = f"{fv_min}-{fv_max}"
    
    
    agg_cluster_stats.append({
        'Cluster': f'Cluster {cluster_id}',
        'Count': count,
        'Mean Avg': mean_avg,
        'Std Avg': std_avg,
        'FV ID Range': fv_range
    })
    
 
    print(f"Cluster {cluster_id}\t| {count}\t| {mean_avg:.4f}\t| {std_avg:.4f}\t| {fv_range}")


agg_stats_df = pd.DataFrame(agg_cluster_stats)
print("\n" + "=" * 50)
print("Formatted Agglomerative Clustering Statistics:")
print("=" * 50)
print(agg_stats_df.to_string(index=False))


agg_stats_df.to_csv('agglomerative_cluster_statistics.csv', index=False)
print(f"\nStatistics saved to 'agglomerative_cluster_statistics.csv'")


fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')


table = ax.table(cellText=agg_stats_df.values,
                 colLabels=agg_stats_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])


table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)


plt.title('Agglomerative Clustering Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('agglomerative_cluster_statistics.jpg', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "=" * 50)
print("Detailed Agglomerative Cluster Statistics:")
print("=" * 50)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Agglomerative_Cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
    print(f"  Mean: {cluster_data['mean'].mean():.4f} ± {cluster_data['mean'].std():.4f}")
    print(f"  Std:  {cluster_data['std'].mean():.4f} ± {cluster_data['std'].std():.4f}")
    print(f"  Skewness: {cluster_data['skewness'].mean():.4f} ± {cluster_data['skewness'].std():.4f}")
    print(f"  Hurst:    {cluster_data['hurst'].mean():.4f} ± {cluster_data['hurst'].std():.4f}")
    print(f"  FV ID Range: {min(cluster_data.index)}-{max(cluster_data.index)}")


agg_results_df = df[['mean', 'std', 'skewness', 'hurst', 'Agglomerative_Cluster']].copy()
agg_results_df['Feature_Vector_ID'] = range(len(agg_results_df))
agg_results_df.to_csv('agglomerative_clustering_results.csv', index=False)
print("\nAgglomerative clustering results saved to 'agglomerative_clustering_results.csv'")

