import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('train_and_test2.csv')

# Replace 'column1' and 'column2' with the actual names of your columns
data = data[['Age', 'Fare']].values  # Convert to a NumPy array
print(data)

# Initial Plot of the Data
plt.figure(figsize=(18, 5))

# Plot the Initial Data Distribution
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o')
plt.title('Initial Data Distribution')
plt.xlabel('Age')
plt.ylabel('Fare')

# K-means Clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

error_rate_kmeans = mean_squared_error(data, kmeans.cluster_centers_[kmeans.labels_], squared=False)

print("K-means Clustering:")
print("Final Cluster Centers:\n", kmeans.cluster_centers_)
print("Final Clusters:\n", [data[kmeans.labels_ == i] for i in range(kmeans.n_clusters)])
print("Epoch Size (Iterations):", kmeans.n_iter_)
print("Error Rate:", error_rate_kmeans)

# K-means Clustering Visualization
plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical.fit(data)

final_centers_hierarchical = np.array([
    data[hierarchical.labels_ == 0].mean(axis=0),
    data[hierarchical.labels_ == 1].mean(axis=0)
])

error_rate_hierarchical = mean_squared_error(data, final_centers_hierarchical[hierarchical.labels_], squared=False)

print("\nHierarchical Clustering:")
print("Final Cluster Centers:\n", final_centers_hierarchical)
print("Final Clusters:\n", [data[hierarchical.labels_ == i] for i in range(2)])
print("Error Rate:", error_rate_hierarchical)

# Hierarchical Clustering Visualization
plt.subplot(1, 3, 3)
plt.scatter(data[:, 0], data[:, 1], c=hierarchical.labels_, cmap='viridis', marker='o')
plt.scatter(final_centers_hierarchical[:, 0], final_centers_hierarchical[:, 1], s=200, c='red', label='Centroids')
plt.title('Hierarchical Clustering')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()

plt.tight_layout()
plt.show()
