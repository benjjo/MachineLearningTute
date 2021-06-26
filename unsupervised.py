from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
style.use("ggplot")

# x = [1, 5, 1.5, 8, 1, 9]
# y = [2, 8, 1.8, 8, 0.6, 11]
data = pd.read_csv('./xclara.csv')
x = data['Temperature Difference'].values
y = data['Pressure Difference'].values

plt.scatter(x, y)
plt.show()

# Converting our data to a NumPy array
# X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
X = list(zip(x, y))

# We initialise K-means algorithm with the required parameter and use .fit() to fit the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Getting values of the centroids and labels based on the fitment
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# print(centroids)
# print(labels)

# Plotting and visualising the output
colours = ["g.", "r.", "c.", "y."]

for i in range(len(X)):
    # print("Coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colours[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()
