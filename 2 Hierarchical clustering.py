# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Importing the dataset

dataset = pd.read_csv('/Users/nicomellein/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values # die letzten 2 sind Merkmale um eine Muster in den anderen Daten zu finden
#print(X)

# Using the dendrogram to find the optimal number of cluster

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward')) # beste Methode ist minimal punkte (Ward variance minimization algorithm)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# Training the Hierarchical Clustering model on the dataset

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = cluster.fit_predict(X)




# Visualising the clusters

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
