""" Applies Clustering algorithms to microscopy image data"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans      # For clustering
# from scipy.cluster.hierarchy import dendrogram, linkage  # For clustering and visualization
from sklearn.preprocessing import StandardScaler  # For standardizing data
# from sklearn.neighbors import NearestNeighbors    # For nearest neighbors
from sklearn.model_selection import train_test_split
import joblib

fullimageData = pd.read_csv('Full_sample_list.csv')
imageData = pd.read_csv('Combined_set_1.csv')
centre_as_cast = pd.read_csv('Centre_As_cast.csv')
centre_homogenised_560 = pd.read_csv('Centre_homogenised_at_560_degrees.csv')
edge_as_cast = pd.read_csv('Edge_As_cast.csv')
centre_homogenised_580 = pd.read_csv('Centre_homogenised_at_580_degrees.csv')
edge_homogenised_560 = pd.read_csv('Edge_homogenised_at_560_degrees.csv')
edge_homogenised_580 = pd.read_csv('Edge_homogenised_at_580_degrees.csv')

as_cast = pd.concat([edge_as_cast,centre_as_cast])
as_cast.y = ['as_cast' for i in range(len(as_cast.y))]
homogenised_560 = pd.concat([centre_homogenised_560,edge_homogenised_560])
homogenised_560.y = ['homogenised_560' for i in range(len(homogenised_560.y))]
homogenised_580 = pd.concat([centre_homogenised_580,edge_homogenised_580])
homogenised_580.y = ['homogenised_580' for i in range(len(homogenised_580.y))]
fullsetsimple = pd.concat([as_cast, homogenised_560, homogenised_580])
centres = pd.concat([centre_as_cast,centre_homogenised_560,centre_homogenised_580])

means = pd.DataFrame([centre_as_cast.mean(),centre_homogenised_560.mean(),edge_as_cast.mean(), centre_homogenised_580.mean(),
                      edge_homogenised_560.mean(),edge_homogenised_580.mean()])
class ImageAnalysis:
    def __init__(self, data):
        self.data = data

    def pairGrid(self, bgn, end):
        # Creates a list of pixel labels to plot
        t = ["P" + str(i) for i in range(bgn, end)]
        # Creates a grid using Seaborn's PairGrid()
        plt.figure(1)
        grd_plt = sns.PairGrid(
            self.data,
            vars=t,
            hue="y",
            diag_sharey=False,
            palette=["red", "green", "blue", "yellow", "brown", "black"])

        # Adds histograms on the diagonal
        grd_plt.map_diag(plt.hist)

        # Adds density plots above the diagonal
        grd_plt.map_upper(sns.kdeplot)

        # Adds scatterplots below the diagonal
        grd_plt.map_lower(sns.scatterplot)

        # Adds a legend
        grd_plt.add_legend()
        plt.show()
        return

    def split_data(self):
        X_trn, X_tst, y_trn, y_tst = train_test_split(
            self.data.filter(regex='\d'),
            self.data.y,
            test_size=0.30,
            random_state=42)
        return X_trn, X_tst, y_trn, y_tst

    def kMeans_plot(self, nClusters, features = ['P120', 'P130']):
        km = KMeans(n_clusters=nClusters)
        km.fit(self.data.filter(regex = '\d'))
        colours = ["red", "green", "blue", "yellow", "brown", "black"]
        nLabels = len(set(self.data.y))
        plt.figure(2)
        sns.scatterplot(
            x=features[0],
            y=features[1],
            data=self.data,
            hue=self.data.y,
            style=km.labels_,
            palette= colours[:nLabels])
        # Adds cluster centers to the same plot
        plt.scatter(
            km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            marker='x',
            s=200,
            c='red')
        plt.show()
        return km

    def plotPCA(self, x_data, y_data):
        # Create an instance of the PCA class
        pca = PCA()
        # # Transforms the training data ('tf' = 'transformed')
        trn_tf = pca.fit_transform(StandardScaler().fit_transform(x_data))
        # Plot the variance explained by each component
        colours = ['red', 'green', 'blue', "yellow", "brown", "black"]
        NLabels = len(set(y_data))
        plt.figure(4)
        plt.plot(pca.explained_variance_ratio_)
        plt.xlim([-5, 30])
        # Plots the projected data set on the first two principal components and colors by class
        plt.figure(5)
        sns.scatterplot(
            x=trn_tf[:, 0],
            y=trn_tf[:, 1],
            style=y_data,
            hue=y_data,
            palette=colours[:NLabels])
        print('pca score:',pca.score(StandardScaler().fit_transform(x_data)))

if __name__ == "__main__":
    main()
