import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

# Generate synthetic data
np.random.seed(0) 
X = np.random.rand(100, 2) # 100 points in 2D space

# Define the number of clusters
num_clusters = 3

# Create KMeans model
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centers')

# Add title and labels
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()