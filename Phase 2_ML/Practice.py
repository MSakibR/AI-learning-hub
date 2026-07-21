from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Create model
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Centroids
centroids = kmeans.cluster_centers_

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], color="red", marker="X", s=200)
plt.show()
