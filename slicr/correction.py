## correction.py
import numpy as np
import torch


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



def compute_distance_correction_simplified(covar_diff, obs_knn_dist, locally_weighted=False, explore=False):
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
        obs_knn_dist_node = obs_knn_dist[i, 1:].unsqueeze(
            1).unsqueeze(0)  # make it a 3D tensor
        ## here we're exploring, so ignore
        if explore:
            diff_distances = np.diff(obs_knn_dist[i, 1:])
        if locally_weighted:
            # Calculate beta coefficients
            # add small constant to avoid division by zero
            weights = 1.0 / (obs_knn_dist_node + 1e-8)
            weighted_covar_diff_node = weights * covar_diff_node
            weighted_obs_knn_dist_node = weights * obs_knn_dist_node
            # Reshape tensors for batch matrix multiplication
            beta = torch.pinverse(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_covar_diff_node)) \
                .bmm(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_obs_knn_dist_node))
        else:
            # Calculate beta coefficients
            beta = torch.pinverse(covar_diff_node) @ obs_knn_dist_node
        # Calculate distance correction
        temp_correction = (covar_diff_node @ beta).squeeze()
        dist_correction[i, 1:] = temp_correction
    return dist_correction

######################################################
## <exploring>
def compute_distance_correction_simplified_with_beta(covar_diff, obs_knn_dist, locally_weighted=False, explore=False):
    """
    Compute the correction for the distances based on covariate differences.
    Also return the beta coefficients representing the impact of the covariates.
    """
    # Ensure inputs are PyTorch tensors of type Float
    covar_diff = covar_diff.float()
    obs_knn_dist = obs_knn_dist.float()
    # Initialize tensor to store the corrected distances
    dist_correction = torch.zeros_like(obs_knn_dist)
    # Initialize tensor to store the beta coefficients
    betas = torch.zeros((covar_diff.shape[0], covar_diff.shape[2]))
    # Iterate over each row (node)
    for i in range(covar_diff.shape[0]):
        # Extract covariate differences and observed distances for this node
        covar_diff_node = covar_diff[i, 1:, :].unsqueeze(0)
        obs_knn_dist_node = obs_knn_dist[i, 1:].unsqueeze(
            1).unsqueeze(0)  # make it a 3D tensor
        # here we're exploring, so ignore
        if explore:
            diff_distances = np.diff(obs_knn_dist[i, 1:])
        if locally_weighted:
            # Calculate beta coefficients
            # add small constant to avoid division by zero
            weights = 1.0 / (obs_knn_dist_node + 1e-8)
            weighted_covar_diff_node = weights * covar_diff_node
            weighted_obs_knn_dist_node = weights * obs_knn_dist_node
            # Reshape tensors for batch matrix multiplication
            beta = torch.pinverse(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_covar_diff_node)) \
                .bmm(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_obs_knn_dist_node))
        else:
            # Calculate beta coefficients
            beta = torch.pinverse(covar_diff_node) @ obs_knn_dist_node
        # Calculate distance correction
        temp_correction = (covar_diff_node @ beta).squeeze()
        dist_correction[i, 1:] = temp_correction
        # Store beta coefficients
        betas[i] = beta.squeeze()
    return dist_correction, betas


def compute_distance_correction_simplified_with_beta_inclusion(covar_diff, obs_knn_dist, inclusion_mask=None, locally_weighted=False):
    """
    Compute the correction for the distances based on covariate differences.
    Also return the beta coefficients representing the impact of the covariates.
    """
    # Ensure inputs are PyTorch tensors of type Float
    covar_diff = covar_diff.float()
    obs_knn_dist = obs_knn_dist.float()
    # Ensure inclusion_mask is also a tensor
    if inclusion_mask is not None:
        inclusion_mask = inclusion_mask.float()
    # Initialize tensor to store the corrected distances
    dist_correction = torch.zeros_like(obs_knn_dist)
    # Initialize tensor to store the beta coefficients
    betas = torch.zeros((covar_diff.shape[0], covar_diff.shape[2]))
    # Iterate over each row (node)
    for i in range(covar_diff.shape[0]):
        # Extract covariate differences and observed distances for this node
        covar_diff_node = covar_diff[i, 1:, :].unsqueeze(0)
        obs_knn_dist_node = obs_knn_dist[i, 1:].unsqueeze(
            1).unsqueeze(0)  # make it a 3D tensor
        if inclusion_mask is not None:
            mask = inclusion_mask[i, 1:].unsqueeze(1).unsqueeze(0)
            covar_diff_node = covar_diff_node * mask
            obs_knn_dist_node = obs_knn_dist_node * mask
        if locally_weighted:
            # Calculate beta coefficients
            # add small constant to avoid division by zero
            weights = 1.0 / (obs_knn_dist_node + 1e-8)
            weighted_covar_diff_node = weights * covar_diff_node
            weighted_obs_knn_dist_node = weights * obs_knn_dist_node
            # Reshape tensors for batch matrix multiplication
            beta = torch.pinverse(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_covar_diff_node)) \
                .bmm(weighted_covar_diff_node.transpose(1, 2).bmm(weighted_obs_knn_dist_node))
        else:
            # Calculate beta coefficients
            beta = torch.pinverse(covar_diff_node) @ obs_knn_dist_node
        # Calculate distance correction
        temp_correction = (covar_diff_node @ beta).squeeze()
        # Apply correction only to the included distances
        if inclusion_mask is not None:
            temp_correction = temp_correction * mask.squeeze() + obs_knn_dist_node.squeeze() * (1 - mask.squeeze())
        dist_correction[i, 1:] = temp_correction
        # Store beta coefficients
        betas[i] = beta.squeeze()
    return dist_correction, betas

## </exploring>
######################################################


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



def correct_observed_distances_mask(obs_knn_dist, dist_correction, mask):
    """
    Correct the observed distances by subtracting the computed correction. The row-wise means and sds of the corrected distances are then adjusted to match those of the original observed distances. The correction is only applied to values indicated by the mask.
    """
    mask[:, 0] = False
    # Create an array of the same shape as obs_knn_dist for the corrected distances
    corrected_obs_knn_dist = obs_knn_dist.clone()
    # Calculate means and stds for only the unmasked (included) values
    obs_knn_dist_means = torch.zeros(obs_knn_dist.shape[0], 1)
    obs_knn_dist_sds = torch.zeros(obs_knn_dist.shape[0], 1)
    for i in range(obs_knn_dist.shape[0]):
        obs_knn_dist_means[i] = obs_knn_dist[i][mask[i]].mean()
        obs_knn_dist_sds[i] = obs_knn_dist[i][mask[i]].std()
    # Compute corrected distances for the masked elements
    corrected_obs_knn_dist[mask] = obs_knn_dist[mask] - dist_correction[mask]
    # Compute row-wise means and sds of the corrected distances for the masked elements
    corrected_obs_knn_dist_means = torch.zeros(
        corrected_obs_knn_dist.shape[0], 1)
    corrected_obs_knn_dist_sds = torch.zeros(
        corrected_obs_knn_dist.shape[0], 1)
    for i in range(corrected_obs_knn_dist.shape[0]):
        corrected_obs_knn_dist_means[i] = corrected_obs_knn_dist[i][mask[i]].mean()
        corrected_obs_knn_dist_sds[i] = corrected_obs_knn_dist[i][mask[i]].std()
    # Create adjusted means and sds tensors with the same shape as corrected_obs_knn_dist
    corrected_obs_knn_dist_means_adj = corrected_obs_knn_dist_means.repeat(
        1, corrected_obs_knn_dist.shape[1])
    corrected_obs_knn_dist_sds_adj = corrected_obs_knn_dist_sds.repeat(
        1, corrected_obs_knn_dist.shape[1])
    obs_knn_dist_means_adj = obs_knn_dist_means.repeat(
        1, obs_knn_dist.shape[1])
    obs_knn_dist_sds_adj = obs_knn_dist_sds.repeat(1, obs_knn_dist.shape[1])
    # Adjust the corrected distances to preserve the original row-wise means and sds for the masked elements
    corrected_obs_knn_dist[mask] = (corrected_obs_knn_dist[mask] - corrected_obs_knn_dist_means_adj[mask]) * (
        obs_knn_dist_sds_adj[mask]/corrected_obs_knn_dist_sds_adj[mask]) + obs_knn_dist_means_adj[mask]
    return corrected_obs_knn_dist


