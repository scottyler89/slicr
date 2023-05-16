import numpy as np
#from scipy.sparse import csr_matrix, lil_matrix
#from sklearn.neighbors import NearestNeighbors
#from sklearn.linear_model import LinearRegression
#from copy import deepcopy
#from scipy.sparse import csr_matrix
import torch
import networkx as nx
from .correction import correct_observed_distances, compute_distance_correction_simplified, compute_covariate_difference

def expand_knn_list(knn_adj_list):
    """
    Expand a k-nearest neighbors adjacency list to include both first and second neighbors.
    """
    # Convert the adjacency list to a sparse adjacency matrix
    n = len(knn_adj_list)
    indices = [[i, j] for i, row in enumerate(knn_adj_list) for j in row]
    indices = torch.tensor(indices, dtype=torch.long).t()  # Transpose to get a 2D tensor
    values = torch.ones(len(indices[0]))  # One value for each index pair
    adj_mat_indices = indices
    adj_mat_values = values
    # Compute the square of the adjacency matrix using spspmm
    adj_mat_sq_indices, adj_mat_sq_values = spspmm(adj_mat_indices, adj_mat_values, adj_mat_indices, adj_mat_values, n, n, n)
    adj_mat_sq = torch.sparse_coo_tensor(adj_mat_sq_indices, adj_mat_sq_values, (n, n))
    # Each row now contains the first and second neighbors of each node
    expanded_knn_adj_list = [torch.where(row > 0)[0].tolist() for row in adj_mat_sq.to_dense()]
    return expanded_knn_adj_list


# Function to prune kNN list
def prune_knn_list(knn_adj_list, corrected_dist, k):
    """
    Prune a k-nearest neighbors adjacency list so that only the top k neighbors with the smallest corrected distances are kept.
    """
    for node in range(len(knn_adj_list)):
        neighbor_dists = [(neighbor, corrected_dist[node, neighbor]) for neighbor in knn_adj_list[node]]
        top_k = heapq.nsmallest(k, neighbor_dists, key=lambda x: x[1])
        knn_adj_list[node] = [x[0] for x in top_k]
    return knn_adj_list


# Function to iteratively update kNN list and corrected distances
def iterative_update(knn_adj_list, corrected_dist, covar_mat, k, threshold=0.01, max_iter=10):
    """
    TODO: DO NOT USE YET. Iteratively update the k-nearest neighbors adjacency list and corrected distances until convergence (or a maximum number of iterations is reached).
    """
    knn_adj_list = [list(adj_list) for adj_list in knn_adj_list] # Convert tensor to list of lists
    for _ in range(max_iter):
        old_corrected_dist = corrected_dist.clone()
        expanded_knn_adj_list = expand_knn_list(knn_adj_list)
        knn_adj_list = prune_knn_list(expanded_knn_adj_list, corrected_dist, k)
        covar_diff = compute_covariate_difference(knn_adj_list, covar_mat)
        dist_correction = compute_distance_correction_simplified(covar_diff, corrected_dist)
        corrected_dist = correct_observed_distances(obs_knn_dist, dist_correction)
        new_sum_dist = corrected_dist.sum(dim=1)
        if torch.abs(old_corrected_dist.sum(dim=1) - new_sum_dist).max() < threshold:
            break
    return knn_adj_list, corrected_dist



def prune_only(adj, dists, new_k):
    """
    Prune a k-nearest neighbors adjacency list and the associated distances so that only the top k neighbors with the smallest distances are kept.
    """
    out_adj = np.zeros((adj.shape[0],new_k))
    out_dists = np.zeros((adj.shape[0],new_k))
    for node in range(adj.shape[0]):
        top_k_idxs = np.argsort(dists[node])[:new_k]
        out_adj[node,:] = adj[node,top_k_idxs]
        out_dists[node,:] = dists[node,top_k_idxs]
    return(out_adj, out_dists)


def inv_min_max_norm(in_vect, epsilon=1e-15):
    """
    Apply inverse min-max normalization to a vector of distances. This function is used to transform distances into weights for graph construction.
    """
    ## note that the first node is self & zero distance
    in_vect[1:] = 1/in_vect[1:]
    in_vect[1:] -= np.min(in_vect[1:]) + epsilon
    in_vect[1:] /= np.max(in_vect[1:])
    return in_vect


def G_from_adj_and_dist(knn_adj_list, corrected_dist):
    """
    Construct a graph from a k-nearest neighbors adjacency list and corrected distances. The distances are first transformed into weights using inverse min-max normalization.
    """
    n = len(knn_adj_list)
    # Create a weighted graph from the adjacency list and distances
    G = nx.Graph()
    for i in range(n):
        corrected_dist[i] = inv_min_max_norm(corrected_dist[i])
        ## 1: to skep self edges
        for j, dist in zip(knn_adj_list[i,1:], corrected_dist[i,1:]):
            G.add_edge(i, j, weight=dist)
    return(G)




