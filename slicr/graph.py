## graph.py
import numpy as np
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


def prune_only_mask(adj, dists, mask, new_k):
    """
    Prune a k-nearest neighbors adjacency list and the associated distances so that only the top k neighbors with the smallest distances are kept. 
    Also uses a mask to ignore certain values.
    """
    out_adj = np.zeros((adj.shape[0], new_k))
    out_dists = np.zeros((adj.shape[0], new_k))
    # create a new mask with the same size
    out_mask = np.zeros((mask.shape[0], new_k), dtype=bool)
    for node in range(adj.shape[0]):
        # mask the distances and adjacency array
        masked_dists = dists[node, mask[node]]
        masked_adj = adj[node, mask[node]]
        # sort only the unmasked distances
        top_k_idxs = np.argsort(masked_dists)[:new_k]
        out_adj[node, :] = masked_adj[top_k_idxs]
        out_dists[node, :] = masked_dists[top_k_idxs]
        out_mask[node, :] = mask[node, top_k_idxs]
    return (out_adj, out_dists, out_mask)


def inv_min_max_norm(in_vect, epsilon=1e-8):
    """
    Apply inverse min-max normalization to a vector of distances. This function is used to transform distances into weights for graph construction.
    """
    ## note that the first node is self & zero distance
    in_vect[1:] = 1/in_vect[1:]
    in_vect[1:] -= (np.min(in_vect[1:]) - epsilon)
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


def mask_knn(dists, cutoff_threshold=3, min_k=10):
    """
    Generate a mask for k nearest neighbors based on a cutoff threshold.
    
    Parameters:
    ----------
    obs_knn_dist : numpy ndarray
        A matrix of the distances to the k nearest neighbors in the observed data.

    k : int
        The number of nearest neighbors to consider.

    cutoff_threshold : float
        The relative gap threshold for considering a difference between sorted order distances.
        This cutoff is the multiple of the mean sort-order difference for considering the distance.
        "too big" to be kept. Getting to this point or farther away will be masked.
        
    Returns:
    -------
    mask : numpy ndarray
        A binary mask matrix indicating which distances should be considered (1) and which should be ignored (0).

    Examples:
    --------
    >>> obs_knn_dist = np.array(...)
    >>> k = 10
    >>> cutoff_threshold = 3.0
    >>> mask = mask_knn(obs_knn_dist, k, cutoff_threshold)
    """
    # Calculate the differences between adjacent distances
    dist_diffs = np.diff(dists[:, min_k:], axis=1)
    # Calculate the cutoff for each row
    diff_cutoff = np.mean(dist_diffs, axis=1) * cutoff_threshold
    print("mean cutoff")
    print(np.mean(diff_cutoff))
    # # Create a mask where the differences exceed the cutoff
    exceed_cutoff_mask = dist_diffs > diff_cutoff[:, np.newaxis]
    print("mean # exceeding cutoff")
    print(np.mean(np.sum(exceed_cutoff_mask,axis=1)))
    print(exceed_cutoff_mask)
    # see which rows need to be masked at all (total # of exceeding edge diffs)
    # see which rows need to be masked at all (any exceeding edge diffs)
    rows_to_mask = np.where(np.any(exceed_cutoff_mask, axis=1))[0]
    #rows_to_mask = np.where(np.sum(exceed_cutoff_mask,axis=1) > 0)[0]
    print("rows_to_mask")
    print(rows_to_mask)
    print("# need pruning:")
    print(rows_to_mask.shape[0])
    ## Calculate each node's first instance of exceeding the cutoff
    inclusion_mask = np.ones_like(dists, dtype=bool)
    #first_exceed_indices = np.zeros(exceed_cutoff_mask.shape[0],dtype=int)
    for node in rows_to_mask:
        temp_arg_max = np.argmax(exceed_cutoff_mask[node,:])
        #print(temp_arg_max)
        inclusion_mask[node, (min_k+temp_arg_max+1):] = False
    print(inclusion_mask)
    print(np.mean(np.sum(inclusion_mask,axis=1)))
    return inclusion_mask


def G_from_adj_and_dist_mask(knn_adj_list, corrected_dist, mask):
    """
    Construct a graph from a k-nearest neighbors adjacency list and corrected distances. The distances are first transformed into weights using inverse min-max normalization. The mask indicates the valid neighbor pairs.
    """
    n = len(knn_adj_list)
    # Create a weighted graph from the adjacency list and distances
    G = nx.Graph()
    for i in range(n):
        corrected_dist[i] = inv_min_max_norm(corrected_dist[i])
        # 1: to skip self edges
        for j, dist, m in zip(knn_adj_list[i, 1:], corrected_dist[i, 1:], mask[i, 1:]):
            if m:  # Only add edge if mask value is True
                G.add_edge(i, j, weight=dist)
    return G


def pruned_to_csr_mask(pruned_adj, pruned_dists, mask, inv_min_max=True):
    # Compute the number of nodes and the number of neighbors (k)
    n_nodes, k = pruned_adj.shape
    # Prepare the data for the csr matrix
    if inv_min_max:
        # Replace invalid distances with np.nan
        pruned_dists_masked = np.where(mask, pruned_dists, np.nan)
        # Apply the normalization function along the rows and then flatten the array
        data = np.apply_along_axis(
            inv_min_max_norm, 1, pruned_dists_masked).flatten()
    else:
        data = pruned_dists.flatten()
    # Flatten the array and convert to int
    indices = pruned_adj.flatten().astype(int)
    indptr = np.arange(0, n_nodes * k + 1, k)  # Create the index pointer array
    # Create the csr matrix
    adj_mat_csr = csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
    # to double check there are no floating point errors
    adj_mat_csr.data[adj_mat_csr.data < 0] = 0
    adj_mat_csr.data[np.isnan(adj_mat_csr.data)] = 0
    return adj_mat_csr
