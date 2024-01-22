# -*- coding: utf-8 -*-
"""
Spyder Editor

LAB Article 3 Class for machine learning parameters
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class Scores_data:
    
    def __init__(self,data,y):
        """
        Parameters
        ----------
        data : np array
            X of shape (n_observations,n_PCs)
        y : str
            array of list of labels.
        """
        self.data = data
        self.y = y
        self.labels = y
        
    def scoreplot(self,fPC_x = 1,fPC_y = 2,y = None):
        
        if not y: y = self.y
        x_,y_ = self.data[:,fPC_x-1],self.data[:,fPC_y-1]
        fig,ax = plt.subplots(figsize=(18, 16))
        sns.scatterplot(x=x_, y=y_,hue = y,ax = ax,legend = True)
        ax.set_xlabel("fPC 1 score")
        ax.set_ylabel("fPC 2 score")
        ax.set_title("FPCA scores - method: {}".format(method))
        
        ax_FPCA.tick_params(labelsize = 10)

class ML_FDA:
    
    def __init__(self):
        parameter = None

if __name__ == '__main__':
    
    #Either multidimensional by default or by species
    method = 'Default'
    
    #read in data
    directory = r'D:\VP\ARTICLE3\ML_paramaters'
    df_variance = pd.read_csv(r'{}\explained_variance.csv'.format(directory),index_col=0)
    df_scores = pd.read_csv(r'{}\fPCA_score_{}.csv'.format(directory,method),index_col=0)
    #coeffs = pd.read_csv(r'{}\B_Spline_coefficients_{}.csv'.format(directory,method),index_col=0)
    
    #Denote y values for hue
    y = [y[:-1] for y in df_scores['Repetition']]
    
    #Get associated explained variance and scores
    variance = np.array(df_variance.loc[method])
    scores = df_scores[df_scores.columns[1:]].values
    
    #Plot fPCA Score Plot
    x_,y_ = scores[:,0],scores[:,1]
    fig_FPCA,ax_FPCA = plt.subplots(figsize=(20, 18))
    sns.scatterplot(x=x_, y=y_,hue = y,ax = ax_FPCA,legend = True)
    ax_FPCA.set_xlabel("fPC 1 score")
    ax_FPCA.set_ylabel("fPC 2 score")
    ax_FPCA.set_title("FPCA scores - method: {}".format(method))
    
    ax_FPCA.tick_params(labelsize = 13)
    
    #%%perform k_means on scores dataset
    
    # Define a custom distance metric (e.g., L1 norm for demonstration purposes)
    # def custom_distance(x, y):
    #     return np.sum(np.abs(x - y))
    
    # Apply k-means clustering with the custom distance metric
    n_clusters = 5  # Replace with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(scores)
    
    # Access cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Centers:\n", cluster_centers)
    print("Labels:", labels)
    
    fig_FPCA,ax_FPCA = plt.subplots(figsize=(20, 18))
    sns.scatterplot(x=x_, y=y_,hue = labels,ax = ax_FPCA,legend = True)
    ax_FPCA.set_xlabel("fPC 1 score")
    ax_FPCA.set_ylabel("fPC 2 score")
    ax_FPCA.set_title("FPCA scores - method: {}".format(method))
    
    ax_FPCA.tick_params(labelsize = 13)
    
    #%% calculate sub cluster distances