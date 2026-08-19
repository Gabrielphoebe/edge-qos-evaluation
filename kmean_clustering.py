import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the data
df = pd.read_csv('fv_reshaped.csv')


features = ['mean', 'std', 'skewness', 'hurst']
X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)


df['KMeans_Cluster'] = cluster_labels


plt.figure(figsize=(12, 8))


fv_ids = np.arange(len(df))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    plt.scatter(fv_ids[mask], [cluster_id] * np.sum(mask), 
                c=colors[cluster_id], marker='o', s=100, alpha=0.7,
                label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title('K-means Clustering: Cluster ID vs FV ID', fontsize=14, fontweight='bold')  


plt.yticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')  


plt.text(0.02, 0.98, 'Clustering Algorithm: K-means', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.savefig('clustering_results_kmeans.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 6))


for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    fv_ids_cluster = fv_ids[mask]
    
    
    jitter = np.random.normal(0, 0.1, len(fv_ids_cluster))
    
    plt.scatter(fv_ids_cluster, [cluster_id + jitter[i] for i in range(len(fv_ids_cluster))], 
                c=colors[cluster_id], marker='o', s=80, alpha=0.8,
                label=f'Cluster {cluster_id}')

plt.xlabel('Feature Vector ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster ID', fontsize=12, fontweight='bold')
plt.title('K-means Clustering: 1D Visualization - Cluster ID vs FV ID', 
          fontsize=14, fontweight='bold')  


plt.yticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left')  


plt.text(0.02, 0.98, 'Clustering Algorithm: K-means', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.savefig('clustering_results_1D_kmeans.jpg', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()

bars = plt.bar(range(optimal_k), cluster_counts.values, 
               color=colors[:optimal_k], alpha=0.7, edgecolor='black')

plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
plt.ylabel('Number of Feature Vectors', fontsize=12, fontweight='bold')
plt.title('K-means: Feature Vector Distribution per Cluster', fontsize=14, fontweight='bold')
plt.xticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])


for bar, count in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('cluster_distribution_kmeans.jpg', dpi=300, bbox_inches='tight')
plt.show()


print("K-means Clustering Results Summary:")
print("=" * 40)
for cluster_id in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_data)} feature vectors")
    print(f"  Mean range: {cluster_data['mean'].min():.3f} - {cluster_data['mean'].max():.3f}")
    print(f"  Std range:  {cluster_data['std'].min():.3f} - {cluster_data['std'].max():.3f}")
    print()

# Save the clustering results to CSV
results_df = df[['mean', 'std', 'skewness', 'hurst', 'KMeans_Cluster']].copy()
results_df['Feature_Vector_ID'] = range(len(results_df))
results_df.to_csv('kmeans_clustering_results.csv', index=False)
print("Clustering results saved to 'kmeans_clustering_results.csv'")
