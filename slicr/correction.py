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
    assert torch.all(obs_knn_dist >=
                     0), "There's a bug. The original distances have negatives"
    ## here we temporarily ignore the mask so that 
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
    # It's possible in weird circumstances, that the regression rotates something below zero (meaning)
    # that this would be even more similar to the point than itself. That's ridiculous. So no.
    # Set a large value
    large_val = torch.full_like(corrected_obs_knn_dist, float('inf'))
    # Use torch.where to apply the mask
    masked_obs_knn_dist = torch.where(mask, corrected_obs_knn_dist, large_val)
    # Calculate the row-wise minimums
    row_mins = masked_obs_knn_dist.min(dim=1)
    print("row_mins")
    print(row_mins)
    # go through the rows where the min is less than zero, then subtract the min value to that row
    # so that they're all positive + eta offset of 1e-8
    # for index where row_mins < 0 ...
    # Identify indices of rows where minimum is less than zero
    epsilon = 1e-8
    indices = torch.where(row_mins.values < epsilon)[0]
    # Define a small constant, eta
    # For each index where row_min is less than zero, subtract min value and add eta
    for i in indices:
        corrected_obs_knn_dist[i] -= row_mins.values[i]
        corrected_obs_knn_dist[i] += epsilon
        corrected_obs_knn_dist[i][0] = 0.
    row_mins = corrected_obs_knn_dist.min(1)
    print("row_mins after correction:")
    print(row_mins)
    print("obs_knn_dist")
    print(obs_knn_dist)
    print("dist_correction")
    print(dist_correction)
    print("corrected_obs_knn_dist")
    print(corrected_obs_knn_dist)
    print("all >= 0:",torch.all(corrected_obs_knn_dist >= 0))
    # set the self connection back to true
    mask[:, 0] = True
    #assert torch.all(corrected_obs_knn_dist>=0), "There's a bug. The corrected distances have negatives"
    return corrected_obs_knn_dist


def resort_order(adj, dist, mask):
    for i in range(dist.shape[0]):
        # Only consider elements that are not masked
        mask_indices = mask[i]
        # Get the elements that are not masked
        unmasked_adj = adj[i, mask_indices]
        unmasked_dist = dist[i, mask_indices]
        # Sort only the unmasked elements
        new_order = torch.argsort(unmasked_dist)
        # Reassign the unmasked, sorted elements back to their original positions
        adj[i, mask_indices] = unmasked_adj[new_order]
        dist[i, mask_indices] = unmasked_dist[new_order]
        mask[i, mask_indices] = mask[i, mask_indices][new_order]
        # Check the sorted order
        if dist[i, 2]-dist[i, 1] < 0 :
            print("dist[i, 2]-dist[i, 1] > 0")
            print("dist[i, :]")
            print(dist[i, :])
            print("adj[i, :]")
            print(adj[i, :])
            assert dist[i, 2]-dist[i, 1] > 0, "WTF?"
    assert torch.all(dist[:, 2]-dist[:, 1]>0), "Sorted in the wrong order. This is a bug."
    return (adj, dist, mask)


def remeasure_distances(adj, obs_X):
    # Index into obs_X using the adjacency list, this gives us a tensor of neighbors
    neighbor_features = obs_X[adj]
    # Calculate the difference between each observation and its neighbors
    # We use None to add an extra dimension to obs_X for broadcasting
    diffs = obs_X[:, None] - neighbor_features
    # Square the differences
    sq_diffs = diffs ** 2
    # Sum over the feature dimension and take the square root to get the Euclidean distances
    distances = np.sqrt(sq_diffs.sum(-1))
    print(distances)
    return(torch.tensor(distances))

