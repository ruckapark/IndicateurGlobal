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

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def index_dictionary(y, unique_elements):
    index_dict = {}

    for elem in unique_elements:
        indices = [i for i, x in enumerate(y) if x == elem]
        index_dict[elem] = indices

    return index_dict

# def cluster_sparsity(cluster_centers, scores, index, dim = 2):
#     array = scores[:,:dim]
#     cluster_points = {s:}
#     sparsity_dict = {}

#     for cluster, center in cluster_centers.items():
#         center_np = np.array(center)
#         points_np = np.array(surrounding_points[cluster])

#         # Transpose the points if needed (to have shape (num_p, 2))
#         if points_np.shape[1] != point_dim:
#             points_np = np.transpose(points_np)

#         # Calculate distances from the cluster center
#         distances = np.linalg.norm(points_np - center_np, axis=1)

#         # Normalize distances by the absolute distance of the cluster center from 0,0
#         normalized_distances = distances / np.linalg.norm(center_np)

#         # Calculate the average normalized distance (cluster sparsity)
#         sparsity = np.mean(normalized_distances)
#         sparsity_dict[cluster] = sparsity

#     return sparsity_dict

# # Example usage
# cluster_centers = {'a': [0, 1], 'b': [2, 2]}
# surrounding_points = {'a': [[0, 1], [1, 1.3], [0, 1], [-1, 0.7]], 'b': [[2, 2], [2.5, 2.7], [3, 2.8], [2, 2.9]]}

# result_dict = cluster_sparsity(cluster_centers, surrounding_points)
# print(result_dict)


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
        self.data_ = self.scale()
        self.y = y
        self.labels = y
        
    def scale(self,method = 'log1p'):
        
        #Adapt this to only treat one column at a time
        if method == 'log1p':
            return self.log1p_scale(self.data)
        elif method == 'log':
            return self.log_scale(self.data)
        elif method == 'sqrt1p':
            return self.sqrt1p_scale(self.data)
        elif method == 'sqrt':
            return self.sqrt_scale(self.data)
        else:
            print('Method Unknown')
        
    def scoreplot(self,fPC_x = 1,fPC_y = 2,y = None,palette = None):
        
        if not y: y = self.y
        x_,y_ = self.data_[:,fPC_x-1],self.data_[:,fPC_y-1]
        fig,ax = plt.subplots(figsize=(16, 14))
        sns.scatterplot(x=x_, y=y_,hue = y,ax = ax,legend = True,palette = palette)
        ax.set_xlabel("fPC 1 score")
        ax.set_ylabel("fPC 2 score")
        ax.set_title("FPCA scores - method: {}".format(method))
        
        ax.tick_params(labelsize = 10)
        
    def log1p_scale(self,array):
        return np.sign(array) * np.log1p(np.abs(array))
    def log_scale(self,array):
        return np.sign(array) * np.log(np.abs(array))
    def sqrt1p_scale(self,array):
        return np.sign(array) * (np.sqrt(np.abs(array) + np.ones(shape = array.shape)) - np.ones(shape = array.shape))
    def sqrt_scale(self,array):
        return np.sign(array) * np.sqrt(np.abs(array))

class ML_FDA:
    
    def __init__(self):
        parameter = None

if __name__ == '__main__':
    
    plt.close('all')
    sns.set_style('white')
    
    #Either multidimensional by default or by species
    method = 'Default'
    
    #Read in data
    directory = r'D:\VP\ARTICLE3\ML_paramaters'
    df_variance = pd.read_csv(r'{}\explained_variance.csv'.format(directory),index_col=0)
    df_scores = pd.read_csv(r'{}\fPCA_score_{}.csv'.format(directory,method),index_col=0)
    #coeffs = pd.read_csv(r'{}\B_Spline_coefficients_{}.csv'.format(directory,method),index_col=0)
    
    #Denote y values for hue
    y = [y[:-1] for y in df_scores['Repetition']]
    
    #Get associated explained variance and scores
    variance = np.array(df_variance.loc[method])
    scores_class = Scores_data(df_scores[df_scores.columns[1:]].values, y)
    
    scores = scores_class.data_
    scores_ = scores*variance
    y = scores_class.y
    substances = pd.Series(y).unique()
    
    index_y = index_dictionary(y, substances)
    
    #Plot fPCA Score Plot
    x_,y_ = scores[:,0],scores[:,1]
    fig_FPCA,ax_FPCA = plt.subplots(figsize=(20, 18))
    sns.scatterplot(x=x_, y=y_,hue = y,ax = ax_FPCA,legend = True)
    ax_FPCA.set_xlabel("fPC 1 score")
    ax_FPCA.set_ylabel("fPC 2 score")
    ax_FPCA.set_title("FPCA scores - method: {}".format(method))
    ax_FPCA.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 9)
    
    ax_FPCA.tick_params(labelsize = 13)
    fig_FPCA.tight_layout()
    
    # Create a custom palette with five symbols repeated for every ten labels
    plot_params = {s:['o','#aaaaaa'] for s in substances}
    symbols = ['o', 's', 'd', 'X', '^']
    custom_palette = ['#386cb0', '#fdb462', '#7fc97f', '#d73027', '#66c2a5', '#e34a33', '#b2df8a', '#8856a7', '#fdae61']
    custom_palette = ['#386cb0', '#e3c956', '#548c54', '#d73027', '#66c2a5', '#e333a8', '#b2df8a', '#8856a7', '#fdae61']
    
    #class palette
    class_palette = ['#bcbd22', '#d62728', '#2ca02c', '#ff7f0e', '#e377c2', '#1f77b4', '#8c564b']
    classes = ['Insecticide','Metal','Other Pesticide','PAH','PPCP','Solvent','Other']
    class_colors = dict(zip(classes,class_palette))
    
    typol_palette = ['#000000','#ff0000','#00ff00','#0000ff','#ff00ff','#a3a3a3']
    typol_classes = ['Class 1','Class 2','Class 3','Class 4','Metal','Other']
    typol_colors = dict(zip(typol_classes,typol_palette))
    
    for i,s in enumerate(substances):
        marker = symbols[i//len(custom_palette)]
        color = custom_palette[i%len(custom_palette)]        
        plot_params[s] = [marker,color]
    
    #Replot with substance specific marker
    x_,y_ = scores[:,0],scores[:,1]
    fig_FPCA,ax_FPCA = plt.subplots(figsize=(16, 14))
    for i in range(scores.shape[0]):
        s = y[i]
        ax_FPCA.scatter(scores[i,0],scores[i,1],marker = plot_params[s][0],color = plot_params[s][1],s = 45)
        
    #Plot legend
    handles,labels = [],[]
    for label, (marker, color) in plot_params.items():
        # Create a Line2D object with the specified marker and color
        handle = Line2D([0], [0], marker=marker, color=color, markersize=8, label=label)
        handles.append(handle)
        labels.append(label)
        
    ax_FPCA.legend(handles=handles, labels=labels,loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 10)
    plt.tight_layout()
    
    #%%read in molecules class
    molecules = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\molecules.csv',index_col = 0)
    y_class = [molecules.loc[y[i]]['Class'] for i in range(len(y))]
    y_typol = ['Class {}'.format(int(molecules.loc[y[i]]['TyPol'])) for i in range(len(y))]
    y_typol2 = [x for x in y_typol]
    for i in range(len(y_typol2)):
        if y_typol2[i] == 'Class 4':
            y_typol2[i] = 'Class 3'
        elif y_typol2[i] == 'Class 5':
            y_typol2[i] = 'Class 4'
            
    #%% Plot y_class and 
    class_all = [y_class[index_y[s][0]] for s in substances]
    typol_all = [y_typol2[index_y[s][0]] for s in substances]
    
    df_class = pd.DataFrame(index = np.arange(len(substances)),columns = ['Name','Repetitions','Class'])
    for i,s in enumerate(substances):
        df_class.iloc[i] = [s,len(index_y[s]),y_class[index_y[s][0]]]
        
    df_class = df_class.sort_values(['Class','Name']).reset_index(drop = True)
    
    #begin bar plot
    fig_bar,ax_bar = plt.subplots(figsize=(6,9))
    sns.barplot(data = df_class,x = 'Repetitions',y = 'Name',ax = ax_bar,hue = 'Class',dodge = False)
    plt.legend(fontsize=13.8,loc = 7)
    plt.tight_layout()
                
    
    #%%
    
    #reorder custom palettes: 
    custom_class = ['#1f77b4','#8c564b','#bcbd22','#d62728','#ff7f0e','#2ca02c','#e377c2']
    custom_typol = ['#a3a3a3','#ff0000','#ff00ff','#000000','#0000ff','#00ff00']
    
    scores_class.scoreplot(y = y_class,palette = custom_class)
    scores_class.scoreplot(y = y_typol,palette = custom_typol)
    
    #Calculate molecule cluster centres
    centres = np.zeros((len(substances),scores.shape[1]))
    for i,s in enumerate(substances):
        centres[i] = np.mean(scores[index_y[s]],axis = 0)
        
    cluster_centres = {s:[centres[i][0],centres[i][1]] for i,s in enumerate(substances)}
    
    #centre sparsity
    dim = 2
    array = scores[:,:dim]
    cluster_points = {s:array[index_y[s]] for s in substances}
    sparsity_dict = {}

    for cluster, centre in cluster_centres.items():
        centre_np = np.array(centre)
        points_np = np.array(cluster_points[cluster])

        # Calculate distances from the cluster center
        distances = np.linalg.norm(points_np - centre_np, axis=1)

        # Normalize distances by the absolute distance of the cluster center from 0,0
        normalized_distances = distances / np.linalg.norm(centre_np)

        # Calculate the average normalized distance (cluster sparsity)
        sparsity = np.mean(normalized_distances)
        sparsity_dict[cluster] = sparsity
    
    # Sort the dictionary by values in ascending order
    sorted_dict = dict(sorted(sparsity_dict.items(), key=lambda item: item[1]))
    
    # Create a bar plot using Seaborn
    fig,ax = plt.subplots(figsize=(9,6))
    sns.barplot(x=list(sorted_dict.keys()), y=list(sorted_dict.values()),ax = ax,color = 'blue')
    
    # Set labels and title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cluster Sparsity')
    ax.set_title('Cluster Sparsity in Ascending Order')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels
    
    ax.set_yscale('log')
    ax.axhline(0.8,color = 'red',linestyle = '--')
    
    # Show the plot
    plt.tight_layout()
    
    
    
    #%%perform k_means on scores dataset
    
    matplot_colors = [
        '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
        '#8c654b','#e377c2','#7f7f7f','#bcbd22','#17becf'
        ]
    
    # Define a custom distance metric (e.g., L1 norm for demonstration purposes)
    # def custom_distance(x, y):
    #     return np.sum(np.abs(x - y))
    
    # Apply k-means clustering with the custom distance metric
    n_clusters = 9  # Replace with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    labels = kmeans.fit_predict(scores_)
    final_labels = labels.copy()
    
    # Access cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Centers:\n", cluster_centers)
    print("Labels:", labels)
    
    fig_FPCA,ax_FPCA = plt.subplots(figsize=(16, 14))
    sns.scatterplot(x=x_, y=y_,hue = labels,ax = ax_FPCA,legend = False,palette = matplot_colors)
    ax_FPCA.set_xlabel("fPC 1 score")
    ax_FPCA.set_ylabel("fPC 2 score")
    ax_FPCA.set_title("FPCA scores - method: {}".format(method))
    
    ax_FPCA.tick_params(labelsize = 13)
    
    cluster_index = {s:labels[index_y[s]] for s in substances}
    kmeans_scores = dict(zip(np.arange(9),[np.where(labels == n)[0] for n in range(9)]))
    kmeans_centres = [[np.mean(scores[:,0][kmeans_scores[n]]),np.mean(scores[:,1][kmeans_scores[n]])] for n in range(9)]
    
    for k,c in enumerate(kmeans_centres):
        ax_FPCA.scatter(x = c[0],y = c[1],marker = 'x',s = 80,color = matplot_colors[k])
        
    handles = [mpatches.Patch(color=color, label=legend) for legend, color in zip(['Cluster {} \n n-{} \n '.format(i,len(kmeans_scores[i-1])) for i in range(1,10)], matplot_colors)]
    ax_FPCA.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 15)
    plt.tight_layout()
    
    hex_colors_spectral = ['#d4464e', '#f47543', '#fdae60','#fecf8b', '#fefcbe', '#e6f58f','#aacd9f', '#66c2a5', '#3292bb']
    
    """
    #%% elbow method - an evaluate cluster consistency
    
    sse = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scores_)
        sse.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure()
    plt.plot(range(1, 15), sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    
    #%% silhouette method
    from sklearn.metrics import silhouette_score

    silhouette_scores = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scores_)
        silhouette_scores.append(silhouette_score(scores_, kmeans.labels_))
    
    # Plot silhouette scores
    plt.figure()
    plt.plot(range(2, 15), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.show()
    
    #%% cluster consistency
    
    limit_sparsity = 0.8 
    assignment_index = 0
    
    consistencies = []
    
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scores_)
        labels = kmeans.fit_predict(scores_)
        consistency = 0
        for s in substances:
            if sparsity_dict[s] > 0.8: 
                pass
            elif cluster_points[s].shape[0] == 1:
                pass
            else:
                y_pred = labels[index_y[s]]
                counts = np.bincount(y_pred)
                y_max = np.argmax(counts)
                y_true = np.ones_like(y_pred) * y_max
                score = np.sum(y_pred == y_true)/len(y_pred)
                consistency += score
        consistencies.append(consistency/28)
    
    #%%
    plt.figure()
    plt.plot(range(2,15), consistencies)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Cluster consistency')
    plt.title('Assessment of intra cluster consistency')
    plt.show()
    
    """
    #%% Comparison - with same number of cluster
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score #comparison
    from sklearn.metrics import confusion_matrix #ground truth vs. TyPol or Class
    
    #%% Calculate sparsity metric with all lower than Anthracene
    #from sklearn.metrics import jaccard_score
    
    #%% Visualise class attribution for Clusters by class
    cluster_lengths = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_lengths[i] = np.sum(labels == i)
    cluster_proportions_class = {n:np.zeros(len(classes)) for n in range(n_clusters)}    
    for i,k in enumerate(labels):
        c = classes.index(y_class[i])
        cluster_proportions_class[k][c] += 1
    for i,k in enumerate(cluster_proportions_class):
        cluster_proportions_class[k] = cluster_proportions_class[k]/cluster_lengths[i]
        
    #make array where row 1 is all values for corresponding clusters
    class_proportions_array = np.column_stack(list(cluster_proportions_class.values()))
    clusters = ['Cluster {}'.format(i+1) for i in np.arange(n_clusters)]
    cluster_proportions = {c:class_proportions_array[i] for i,c in enumerate(classes)}
    
    width = 0.5

    fig_class, ax_class = plt.subplots(figsize = (8,4))
    bottom = np.zeros(n_clusters)

    for cluster, proportion in cluster_proportions.items():
        color = class_colors[cluster]
        p = ax_class.bar(clusters, proportion, width, label=cluster, bottom=bottom, color=color)
        bottom += proportion

    ax_class.set_title("Micropollutant Class Associations")
    ax_class.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_class.set_xticklabels(clusters, rotation=45, ha='right')
    plt.tight_layout()
    
    #PLOT OVER ALL SUBSTANCES
    
    
    #%% Visualise class attribution for Clusters by TyPol Class
    
    #BUG THIS HAS FAILED
    for i in range(len(y_typol)):
        if y_typol[i] == 'Class 0':
            y_typol[i] = 'Other'
        elif y_typol[i] == 'Class 5':
            y_typol[i] = 'Metal'
            
    cluster_lengths = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_lengths[i] = np.sum(labels == i)
    cluster_proportions_class = {n:np.zeros(len(typol_classes)) for n in range(n_clusters)}    
    for i,k in enumerate(labels):
        c = typol_classes.index(y_typol[i])
        cluster_proportions_class[k][c] += 1
    for i,k in enumerate(cluster_proportions_class):
        cluster_proportions_class[k] = cluster_proportions_class[k]/cluster_lengths[i]
        
    #make array where row 1 is all values for corresponding clusters
    class_proportions_array = np.column_stack(list(cluster_proportions_class.values()))
    clusters = ['Cluster {}'.format(i+1) for i in np.arange(n_clusters)]
    cluster_proportions = {c:class_proportions_array[i] for i,c in enumerate(typol_classes)}
    
    width = 0.5

    fig_class, ax_class = plt.subplots(figsize = (8,4))
    bottom = np.zeros(n_clusters)

    for cluster, proportion in cluster_proportions.items():
        color = typol_colors[cluster]
        p = ax_class.bar(clusters, proportion, width, label=cluster, bottom=bottom, color=color)
        bottom += proportion

    ax_class.set_title("TyPol Cluster Associations")
    ax_class.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_class.set_xticklabels(clusters, rotation=45, ha='right')
    plt.tight_layout()
    

    #%% Hierarchical clustering to visualise tree
    
    
    #%% Clustering on B-Spline components
    
    