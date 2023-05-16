#import numpy as np
#from scipy.sparse import csr_matrix, lil_matrix
#from sklearn.neighbors import NearestNeighbors
#from sklearn.linear_model import LinearRegression
#from copy import deepcopy
#from scipy.sparse import csr_matrix
import torch
#from .utils import convert_to_torch_sparse#, standardize, unstandardize


# 1. Compute Covariate Difference
def compute_covariate_difference(knn_adj_list, covar_mat):
    """
    Compute the absolute difference in covariate values between each source node and its corresponding target nodes.
    """
    # Get the covariate values for each pair of adjacent nodes
    node_covar_values = covar_mat[knn_adj_list]  # shape: (n, k, g)
    # Separate the source nodes' covariate values
    source_node_covar_values = node_covar_values[:, 0, :]  # shape: (n, g)
    # Compute the difference between each source node and its corresponding target nodes
    covar_diff = (node_covar_values - source_node_covar_values[:, None, :]).abs()  # shape: (n, k, g)
    return covar_diff


# 2. Compute the predicted distances
def compute_distance_correction_simplified(covar_diff, obs_knn_dist):
    """
    Compute the correction for the distances based on covariate differences.
    """
    # Ensure inputs are PyTorch tensors of type Float
    covar_diff = covar_diff.float()
    obs_knn_dist = obs_knn_dist.float()
    # Initialize tensor to store the corrected distances
    dist_correction = torch.zeros_like(obs_knn_dist)
    # Iterate over each row (node)
    for i in range(covar_diff.shape[0]):
        # Extract covariate differences and observed distances for this node
        covar_diff_node = covar_diff[i, 1:, :].unsqueeze(0)
        obs_knn_dist_node = obs_knn_dist[i, 1:].unsqueeze(1)  # make it a column vector
        ## Standardize
        #covar_diff_node, mean_covar, std_covar = standardize(covar_diff_node)
        #obs_knn_dist_node, mean_dist, std_dist = standardize(obs_knn_dist_node)
        # Calculate beta coefficients
        beta = torch.pinverse(covar_diff_node) @ obs_knn_dist_node
        # Calculate distance correction
        temp_correction = (covar_diff_node @ beta).squeeze()
        # Un-standardize
        #temp_correction = unstandardize(temp_correction, mean_dist, std_dist)
        dist_correction[i, 1:] = temp_correction
    return dist_correction


# 3. Correct Observed Distances
def correct_observed_distances(obs_knn_dist, dist_correction):
    """
    Correct the observed distances by subtracting the computed correction. The row-wise means and sds of the corrected distances are then adjusted to match those of the original observed distances.
    """
    # Compute row-wise means of the observed distances
    obs_knn_dist_means = obs_knn_dist[:,1:].mean(dim=1, keepdim=True)
    obs_knn_dist_sds = obs_knn_dist[:,1:].std(dim=1, keepdim=True)
    # Compute corrected distances
    corrected_obs_knn_dist = obs_knn_dist - dist_correction
    # Compute row-wise means and sds of the corrected distances
    corrected_obs_knn_dist_means = corrected_obs_knn_dist[:,1:].mean(dim=1, keepdim=True)
    corrected_obs_knn_dist_sds = corrected_obs_knn_dist[:,1:].std(dim=1, keepdim=True)
    # Adjust the corrected distances to preserve the original row-wise means and sds
    corrected_obs_knn_dist[:,1:] = (corrected_obs_knn_dist[:,1:] - corrected_obs_knn_dist_means) * (obs_knn_dist_sds/corrected_obs_knn_dist_sds) + obs_knn_dist_means
    return corrected_obs_knn_dist

