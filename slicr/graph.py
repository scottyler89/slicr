## graph.py
import torch
import numpy as np
import networkx as nx
from copy import deepcopy
from scipy.sparse import csr_matrix
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
    assert adj.shape[1]>= new_k, "need to have a greater number of k connections in the original graph compared to the new_k we're trying to prune to"
    return(adj[:,:new_k], dists[:,:new_k], mask[:,:new_k])

"""     out_adj = np.zeros((adj.shape[0], new_k))
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
 """

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
        G.add_node(i)
    for i in range(n):
        corrected_dist[i] = inv_min_max_norm(corrected_dist[i])
        ## 1: to skep self edges
        for temp_j, temp_dist in zip(knn_adj_list[i,1:], corrected_dist[i,1:]):
            if temp_j in G[i]:
                    G.add_edge(deepcopy(i),
                               deepcopy(temp_j),
                               weight=min(deepcopy(temp_dist), 
                                          G[i][temp_j]['weight'])
                    )
            else:
                G.add_edge(deepcopy(i),
                            deepcopy(temp_j),
                            weight=deepcopy(temp_dist)
                            )
            G.add_edge(i, temp_j, weight=temp_dist)
    return(G)


def mask_knn_mean_dist(dists, cutoff_threshold=3, min_k=10):
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
    mask = np.ones_like(dists, dtype=bool)
    mean_dist = np.mean(dists[:,1:min_k])
    sd_dist = np.std(dists[:, 1:min_k])
    z_dist = dists[:, min_k:] - mean_dist
    z_dist /= sd_dist
    #sns.distplot(z_dist.flatten())
    #plt.show()
    mask[:,min_k:] = z_dist < cutoff_threshold
    print("mean z mask:")
    print(mask)
    print("mean number connections:")
    print(np.mean(np.sum(mask, axis=1)))
    return(mask)


def mask_knn_diff_dist(dists, cutoff_threshold=3, min_k=10):
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
    diff_mask = np.ones_like(dists, dtype=bool)
    discrete_diff = np.diff(dists[:, (min_k-1):], axis=1)
    discrete_diff -= np.mean(discrete_diff)
    discrete_diff /= np.std(discrete_diff)
    print("discrete_diff:")
    print(discrete_diff)
    # mask everything whose discrete difference is > threshold & farther
    # So everything that's less than the cutoff is good to go, so set those to true
    temp_diff_mask_cutoff = discrete_diff < cutoff_threshold
    print("temp_diff_mask_cutoff:")
    print(temp_diff_mask_cutoff)
    # finds the idxs with jumps (where there is a false)
    idxs_with_diff_mask = np.where(
        np.min(temp_diff_mask_cutoff, axis=1) == 0)[0]
    for idx in idxs_with_diff_mask:
        # then mask everything that is farther than the jump
        # where are they false, then which index is lowest that is false
        temp_gap_idx = np.min(
            np.where(temp_diff_mask_cutoff[idx, :] == False)[0])
        temp_diff_mask_cutoff[idx, temp_gap_idx:] = False
    diff_mask[:, min_k:] = temp_diff_mask_cutoff
    print("diff_mask")
    print(diff_mask)
    print("mean number connections:")
    print(np.mean(np.sum(diff_mask, axis=1)))
    return (diff_mask)



def mask_knn(dists, cutoff_threshold=3, min_k=10, skip_mean_mask = False):
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
    dists = dists.numpy()
    if not skip_mean_mask:
        mask = mask_knn_mean_dist(dists, cutoff_threshold=cutoff_threshold, min_k=min_k)
    else:
        mask = np.ones_like(dists, dtype=bool)
    ## now we'll also mask at big jumps
    if False:
        diff_mask = np.ones_like(dists, dtype=bool)
        discrete_diff = np.diff(dists[:, (min_k-1):], axis=1)
        discrete_diff -= np.mean(discrete_diff)
        discrete_diff /= np.std(discrete_diff)
        print("discrete_diff:")
        print(discrete_diff)
        # mask everything whose discrete difference is > threshold & farther
        # So everything that's less than the cutoff is good to go, so set those to true
        temp_diff_mask_cutoff = discrete_diff < cutoff_threshold
        print("temp_diff_mask_cutoff:")
        print(temp_diff_mask_cutoff)
        # finds the idxs with jumps (where there is a false)
        idxs_with_diff_mask = np.where(np.min(temp_diff_mask_cutoff, axis=1)==0)[0]
        for idx in idxs_with_diff_mask:
            # then mask everything that is farther than the jump
            # where are they false, then which index is lowest that is false
            temp_gap_idx = np.min(np.where(temp_diff_mask_cutoff[idx,:]==False)[0])
            temp_diff_mask_cutoff[idx, temp_gap_idx:]=False
        diff_mask[:, min_k:] = temp_diff_mask_cutoff
        print("diff_mask")
        print(diff_mask)
        print("mean number connections:")
        print(np.mean(np.sum(diff_mask, axis=1)))
    else:
        diff_mask = mask_knn_diff_dist(
            dists, cutoff_threshold=cutoff_threshold, min_k=min_k)
    # merge the masks
    mask = mask * diff_mask
    print("final mask:")
    print(mask)
    print("mean number connections:")
    print(np.mean(np.sum(mask, axis=1)))
    return(mask)


def G_from_adj_and_dist_mask(knn_adj_list, corrected_dist, mask):
    """
    Construct a graph from a k-nearest neighbors adjacency list and corrected distances. The distances are first transformed into weights using inverse min-max normalization. The mask indicates the valid neighbor pairs.
    """
    n = len(knn_adj_list)
    # Create a weighted graph from the adjacency list and distances
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        inv_min_max_dist = inv_min_max_norm(deepcopy(corrected_dist[i,:]))
        # 1: to skip self edges
        for temp_j, temp_dist, temp_m in zip(knn_adj_list[i, 1:], inv_min_max_dist[1:], mask[i, 1:]):
            if temp_m:  # Only add edge if mask value is True
                # if the edge is already there, update to the minimum
                if temp_j in G[i]:
                    G.add_edge(deepcopy(i),
                               deepcopy(temp_j),
                               weight=min(deepcopy(temp_dist), 
                                          G[i][temp_j]['weight'])
                    )
                else:
                    G.add_edge(deepcopy(i),
                               deepcopy(temp_j),
                               weight=deepcopy(temp_dist)
                               )
                #G.add_edge(deepcopy(i), deepcopy(temp_j), weight=deepcopy(temp_dist))
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


def get_re_expanded_adj_and_dist(pruned_adj, pruned_dists, mask, k, epsilon = 1e-8, debug=False,local_iter=0):
    # Number of nodes
    n_nodes = pruned_adj.shape[0]
    # Initialize the new adjacency list, distance list and mask
    new_adj = torch.zeros((n_nodes, k), dtype=torch.long)
    new_adj[:, 0] = torch.arange(n_nodes, dtype=torch.long)
    new_dists = torch.zeros((n_nodes, k))
    new_mask = torch.zeros((n_nodes, k), dtype=torch.bool)
    new_mask[:, 0] = True
    # Process each node
    for node in range(n_nodes):
        # Get the first neighbors and their distances
        first_neighbors = pruned_adj[node,:]
        #print("first_neighbors", first_neighbors)
        first_neighbor_dists = pruned_dists[node,:]
        #print("first_neighbor_dists", first_neighbor_dists)
        # Get the second neighbors and their distances
        second_neighbors = pruned_adj[first_neighbors,:]
        second_neighbor_dists = pruned_dists[first_neighbors,:]
        # we'll assume that the nearest non-self is 'as close as you can get'
        first_neighbor_dists_offset=first_neighbor_dists.clone()
        first_neighbor_dists_offset[1:] = first_neighbor_dists[1:] - \
            first_neighbor_dists[1]+epsilon
        # reset the self distance to zero
        first_neighbor_dists_offset[0]=0
        #print(first_neighbor_dists)
        #print(first_neighbor_dists_offset)
        # Expand first_neighbor_dists to match the shape of second_neighbor_dists
        first_neighbor_dists_expanded = first_neighbor_dists_offset.unsqueeze(1).expand_as(second_neighbor_dists)
        # Now, add them together
        combined_dists = first_neighbor_dists_expanded + second_neighbor_dists
        if not torch.all(combined_dists>=0):
            print("first_neighbor_dists_offset")
            print(first_neighbor_dists_offset)
            print("second_neighbor_dists")
            print(second_neighbor_dists)
            print("combined_dists")
            print(combined_dists)
            assert False, "Combined_dists is negative. This is a bug."
        # now reset the first neighbors to their actual distances 
        combined_dists[:,0]=pruned_dists[node,:]
        #print(pruned_dists[first_neighbors, :]-combined_dists)
        # Compute the total distances to the second neighbors by adding the first neighbors' distances
        combined_dists = first_neighbor_dists.view(
            -1, 1) + second_neighbor_dists
        # Create a mask to ignore the self-connections, but keep first neighbors
        #########################################################
        ## DONE: incorporate the prior mask as well
        previously_masked_nodes = first_neighbors[~mask[node, :]]
        #print("previously_masked_nodes.shape",previously_masked_nodes.shape)
        #print("second_neighbors.shape",second_neighbors.shape)
        #print("node:",node)
        #temp_mask = (second_neighbors != node)
        # Create a tensor from the node and expand it to match second_neighbors shape
        node_tensor = torch.tensor(node).expand(*second_neighbors.shape)
        # Handle the case when previously_masked_nodes is empty
        if len(previously_masked_nodes) == 0:
            # If there are no previously_masked_nodes, then all second_neighbors are valid (not in previously_masked_nodes)
            previously_masked_nodes_condition = torch.ones_like(second_neighbors, dtype=torch.bool)
        else:
            # If there are previously_masked_nodes, do the comparison
            second_neighbors_expanded = second_neighbors.unsqueeze(-1).expand(-1, -1, len(previously_masked_nodes))
            previously_masked_nodes_expanded = previously_masked_nodes.unsqueeze(0).unsqueeze(0).expand(*second_neighbors.shape, -1)
            previously_masked_nodes_condition = ~torch.any(second_neighbors_expanded == previously_masked_nodes_expanded, dim=-1)
        # Create temp_mask
        temp_mask = (second_neighbors != node_tensor) & previously_masked_nodes_condition
        #second_neighbors_expanded = second_neighbors.unsqueeze(2).expand(-1, -1, len(previously_masked_nodes))
        #previously_masked_nodes_expanded = previously_masked_nodes.unsqueeze(0).unsqueeze(0).expand(second_neighbors.shape[0], second_neighbors.shape[1], -1)
        #temp_mask = (second_neighbors != node.unsqueeze(1)) & (~torch.any(second_neighbors_expanded == previously_masked_nodes_expanded, dim=2))
        #######################################################
        #print("temp_mask\n", temp_mask)
        #print("temp_mask.shape:",temp_mask.shape)
        # Apply the mask to the combined_neighbors and combined_dists
        masked_neighbors = second_neighbors[temp_mask]
        masked_dists = combined_dists[temp_mask]
        # Get the unique nodes in the masked_neighbors and their indices
        unique_nodes, indices = torch.unique(
            masked_neighbors, return_inverse=True)
        # Compute the minimum distance to each unique node
        min_dists = torch.zeros(unique_nodes.shape[0])
        for i, unique_node in enumerate(unique_nodes):
            min_dists[i] = masked_dists[indices == i].min()
        # Get the indices that would sort the min_dists
        sorted_indices = torch.argsort(min_dists)
        # Get the indices of top k shortest distances
        # -1 to give the self-connection it's slot
        temp_k = min(k, sorted_indices.shape[0])-1
        top_k_indices = sorted_indices[:temp_k]
        # Select top k neighbors and their distances and store them
        #print("setting the top k neighbors:")
        new_adj[node, 1:(temp_k+1)] = unique_nodes[top_k_indices]
        #print(new_adj[node,:])
        new_dists[node, 1:(temp_k+1)] = min_dists[top_k_indices]
        #print(new_dists[node, :])
        # Update the mask
        new_mask[node, :(temp_k+1)] = True
        if debug:
            if temp_k < k-1:
                print("WARNING: found instance of insufficient k")
                print("temp_k:", temp_k, "k-1:", k-1)
                print("top_k_indices")
                print(top_k_indices)
                print("unique_node:")
                print(unique_node)
                print("setting the remainder to be masked")
                print(new_mask[node,:])
                print(new_adj[node, :])
                #print(new_adj[node,:])
                print(new_dists[node, :])
                #print(new_dists[node, :])
                # Update the mask
                print("mask:\n",new_mask[node,])
            else:
                print("\n\n full k!")
            #print(new_mask[node,:])
            ## double check that we only have unique values here
            temp_nodes, temp_idxs = torch.unique(
                new_adj[node, :], return_inverse=True)
            if temp_nodes.shape[0] < k:
                print("top_k_indices")
                print(top_k_indices)
                print("min_dists")
                print(min_dists)
                print("unique_node")
                print(unique_node)
                print("min_dists[top_k_indices]")
                print(min_dists[top_k_indices])
                assert temp_nodes.shape[0] < k, "we have duplicate nodes when trying to expand the network"
    print("avg number connections:",np.mean(np.sum(new_mask.numpy(), axis=1)))
    if local_iter>2 and debug:
        print("iter:", local_iter)
        # print(poop)
    return new_adj, new_dists, new_mask








""" 
pruned_adj = torch.tensor(
[[0,1,2],
 [1,2,3],
 [2,3,4],
 [3,4,5],
 [4,5,6],
 [5,6,1],
 [6,1,2]],
dtype=torch.long
)
pruned_dists = torch.tensor(
[[0,1,2],
 [0,1,2],
 [0,1,2],
 [0,1,2],
 [0,1,2],
 [0,1,2],
 [0, 1, 2]],
dtype=torch.float
)
n_nodes = pruned_adj.shape[0]
k=4
# Initialize the new adjacency list, distance list and mask
new_adj = torch.zeros((n_nodes, k), dtype=torch.long)
new_adj[:, 0] = torch.arange(n_nodes, dtype=torch.long)
new_dists = torch.zeros((n_nodes, k))
new_mask = torch.zeros((n_nodes, k), dtype=torch.bool)
new_mask[:, 0] = True

for node in range(n_nodes):
    # Get the first neighbors and their distances
    first_neighbors = pruned_adj[node, :]
    # print("first_neighbors", first_neighbors)
    first_neighbor_dists = pruned_dists[node, :]
    # print("first_neighbor_dists", first_neighbor_dists)
    # Get the second neighbors and their distances
    second_neighbors = pruned_adj[first_neighbors, :]
    second_neighbor_dists = pruned_dists[first_neighbors, :]
    # we'll assume that the nearest non-self is 'as close as you can get'
    first_neighbor_dists_offset = first_neighbor_dists.clone()
    first_neighbor_dists_offset[1:] = first_neighbor_dists[1:] - \
        first_neighbor_dists[1]+epsilon
    # reset the self distance to zero
    first_neighbor_dists_offset[0] = 0
    # print(first_neighbor_dists)
    # print(first_neighbor_dists_offset)
    # Expand first_neighbor_dists to match the shape of second_neighbor_dists
    first_neighbor_dists_expanded = first_neighbor_dists_offset.unsqueeze(
        1).expand_as(second_neighbor_dists)
    # Now, add them together
    combined_dists = first_neighbor_dists_expanded + second_neighbor_dists
    if not torch.all(combined_dists >= 0):
        print("first_neighbor_dists_offset")
        print(first_neighbor_dists_offset)
        print("second_neighbor_dists")
        print(second_neighbor_dists)
        print("combined_dists")
        print(combined_dists)
        assert False, "Combined_dists is negative. This is a bug."
    # now reset the first neighbors to their actual distances
    combined_dists[:, 0] = pruned_dists[node, :]
    # print(pruned_dists[first_neighbors, :]-combined_dists)
    # Compute the total distances to the second neighbors by adding the first neighbors' distances
    combined_dists = first_neighbor_dists.view(
        -1, 1) + second_neighbor_dists
    # Create a mask to ignore the self-connections, but keep first neighbors
    temp_mask = (second_neighbors != node)
    # print("temp_mask\n", temp_mask)
    # print("temp_mask.shape:",temp_mask.shape)
    # Apply the mask to the combined_neighbors and combined_dists
    masked_neighbors = second_neighbors[temp_mask]
    masked_dists = combined_dists[temp_mask]
    # Get the unique nodes in the masked_neighbors and their indices
    unique_nodes, indices = torch.unique(
        masked_neighbors, return_inverse=True)
    # Compute the minimum distance to each unique node
    min_dists = torch.zeros(unique_nodes.shape[0])
    for i, unique_node in enumerate(unique_nodes):
        min_dists[i] = masked_dists[indices == i].min()
    # Get the indices that would sort the min_dists
    sorted_indices = torch.argsort(min_dists)
    # Get the indices of top k shortest distances
    # -1 to give the self-connection it's slot
    temp_k = min(k, sorted_indices.shape[0])-1
    top_k_indices = sorted_indices[:temp_k]
    # Select top k neighbors and their distances and store them
    # print("setting the top k neighbors:")
    new_adj[node, 1:temp_k+1] = unique_nodes[top_k_indices]
    # print(new_adj[node,:])
    new_dists[node, 1:temp_k+1] = min_dists[top_k_indices]
    # print(new_dists[node, :])
    # Update the mask
    new_mask[node, :temp_k+1] = True

 """