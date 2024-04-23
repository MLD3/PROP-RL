import numpy as np
import pandas as pd
import os
import joblib
import pickle
from Cluster_Ensembles import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

###################################################################################
#######################         Helper Functions        ###########################
###################################################################################

def compute_centroids(df, centroids):
    cluster_label = df.iloc[0, 0]
    centroid = df.iloc[:, 1:].mean(axis = 0).to_numpy()
    centroids[cluster_label] = centroid
        
def combine_results_from_ensemble(split_id, k = 100):
    
    E, D = 150, 32  # dimensionality of embedded features
    partitions = []
    
    # load trained embeddings
    tr_embed_df = pd.read_pickle('./data/embedded.p')
    train_np = tr_embed_df.to_numpy()
    save_dir = './kmeans_model/split_{0:03d}/ensemble_clusters_{1}/'.format(split_id, k)
    
    for i in range(E):
        kmeans = joblib.load(save_dir + 'kmeans_{}.joblib'.format(i))
        labels = kmeans.predict(train_np)
        partitions.append(labels)

    # stack labels to form E x N matrix 
    partitions = np.stack(partitions, axis=0)
    with open(save_dir + 'each_predictions_tr.npy', 'wb') as f:
        np.save(f, partitions)
        
    # Combine the ensemble labels
    consensus_labels = cluster_ensembles(partitions, nclass = k, solver = 'hbgf', verbose = True, random_state = 0)
    with open(save_dir + 'consensus_predictions_tr.npy', 'wb') as f:
        np.save(f, consensus_labels)
        print('Labels combined')
    
    # Reconstruct cluster centers using the consensus labels
    cluster_center = tr_embed_df.copy()
    cluster_center.insert(0, 'labels', consensus_labels)
    centroids = np.zeros((k, D))
    cluster_center.groupby('labels', as_index = False).apply(compute_centroids, centroids)
    with open(save_dir + 'centroids.npy', 'wb') as f:
        np.save(f, centroids)
        print('Cluster center calculated')
        
    # Set centers as means of new k-means object
    kmeans = KMeans(n_clusters=k, init=centroids).fit(centroids)
    assert np.allclose(kmeans.cluster_centers_, centroids)
    joblib.dump(kmeans, save_dir + 'kmeans_ensemble.joblib')
    print('Finished calculating new k-means')

###################################################################################
#######################          Main Functions         ###########################
###################################################################################
def run_ensemble_kmeans(split_id = None, k = 100, E = 150):
    '''
    Run ensemble kmeans. This samples a single window from every patient and learns a clustering solution.
    After E bootstraps, it combines the ensemble results into a single clustering solution
        split_id : id of the data partition used to learn the clustering solution from
        k : number of states
        E : number of bootstraps
    '''
    tr_embed_df = pd.read_pickle('./data/embedded.p'.format(split_id, split_id))
    for i in tqdm(range(E)): 
        # randomly sample a single window from every patient
        train_samp = tr_embed_df.groupby('hosp_id', as_index=False).apply(lambda x: x.sample(1, random_state = i))
        train_samp = train_samp.reset_index().drop(columns=['level_0']).set_index(['hosp_id', 'window_id', 'ID'])

        # kmeans clustering
        kmeans = KMeans(n_clusters = k, random_state = i).fit(train_samp)
        joblib.dump(kmeans, './kmeans_model/ensemble_clusters_{1}/kmeans_{2}.joblib'.format(k, i))

    # Combine the ensemble results into a single clustering solution
    combine_results_from_ensemble(split_id, k)
    
def predict_clusters_for_embedded_data(save_dir, kmeans, k = 100):
    ''' 
    Using the different clustering solutions, predict the clusters for the embedded data
        save_dir: path to directory where cluster predictions are saved.
        kmeans: clustering model, loaded in from `kmeans_ensemble.joblib` generated in `combine_results_from_ensemble()`
    '''
    embed_df = pd.read_pickle('./data/embedded.p')
    labels = kmeans.predict(embed_df.astype('float'))
    cluster = embed_df.copy()
    cluster.insert(0, 'labels', labels)
    cluster.to_csv(save_dir + 'ens_clusters.csv')
    with open(save_dir + 'ens_clust_pred.npy', 'wb') as f:
        np.save(f, train_labels)

def main():
    ''' Dummy example for how these functions will be utilized in practice '''
    run_ensemble_kmeans()
    predict_clusters_for_embedded_data()
    
    