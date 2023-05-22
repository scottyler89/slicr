## analysis.py
import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .utils import convert_to_torch_sparse
from .correction import compute_covariate_difference, compute_distance_correction_simplified, correct_observed_distances, compute_distance_correction_simplified_with_beta_inclusion, correct_observed_distances_mask
from .graph import mask_knn#, expand_knn_list, prune_knn_list, iterative_update, prune_only, inv_min_max_norm, G_from_adj_and_dist



def perform_analysis(obs_X, obs_knn_dist, covar_mat, k, locally_weighted=False):
    """
    Perform the entire analysis pipeline, which includes nearest neighbors search, distance correction, adjacency list expansion, and iterative update.
    """
    # Check input types and formats
    assert isinstance(obs_X, csr_matrix), "obs_X must be a scipy csr_matrix"
    assert isinstance(obs_knn_dist, np.ndarray), "obs_knn_dist must be a numpy ndarray"
    assert isinstance(covar_mat, np.ndarray), "covar_mat must be a numpy ndarray"
    assert obs_X.shape[0] == obs_knn_dist.shape[0] == covar_mat.shape[0], "Inconsistent input shapes"
    assert obs_knn_dist.shape[1] == k, "Number of nearest neighbors in obs_knn_dist does not match specified k"
    # Convert to PyTorch tensors
    obs_X_torch = convert_to_torch_sparse(obs_X)
    obs_knn_dist_torch = torch.tensor(obs_knn_dist)
    covar_mat_torch = torch.tensor(covar_mat).float()
    # Perform nearest neighbors search
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(obs_X_torch)
    obs_knn_dist, obs_knn_adj_list = nbrs.kneighbors(obs_X_torch, return_distance=True)
    # Compute the covariate difference
    covar_diff = compute_covariate_difference(obs_knn_adj_list, covar_mat_torch)
    # Compute the distance correction
    dist_correction = compute_distance_correction_simplified(
        covar_diff, obs_knn_dist_torch, locally_weighted=locally_weighted)
    # Correct the observed distances
    corrected_obs_knn_dist = correct_observed_distances(obs_knn_dist_torch, dist_correction)
    # Convert corrected distances back to numpy and return
    return corrected_obs_knn_dist.numpy()


def perform_analysis_with_mask(obs_X, obs_knn_dist, covar_mat, k, cutoff_threshold, locally_weighted=False):
    """
    Perform the entire analysis pipeline, which includes nearest neighbors search, distance correction, adjacency list expansion, and iterative update, incorporating a mask matrix to indicate values to be included.

    Parameters:
    ----------
    obs_X : scipy csr_matrix
        The observed data in the form of a sparse matrix.

    obs_knn_dist : numpy ndarray
        A matrix of the distances to the k nearest neighbors in the observed data.

    covar_mat : numpy ndarray
        A matrix of covariates.

    k : int
        The number of nearest neighbors to consider.

    cutoff_threshold : float
        The threshold for considering a jump in distances significant. This parameter is used in the generation of the mask matrix.

    locally_weighted : bool, optional
        Whether to use locally weighted regression in the distance correction computation. Defaults to False.

    Returns:
    -------
    corrected_obs_knn_dist : numpy ndarray
        The corrected observed distances to the k nearest neighbors in the observed data, after applying the full pipeline of nearest neighbors search, distance correction, adjacency list expansion, and iterative update. The correction has been applied only to values indicated by the mask matrix.

    Examples:
    --------
    >>> obs_X = csr_matrix(...)
    >>> obs_knn_dist = np.array(...)
    >>> covar_mat = np.array(...)
    >>> k = 10
    >>> cutoff_threshold = 3.0
    >>> corrected_obs_knn_dist = perform_analysis_with_mask(obs_X, obs_knn_dist, covar_mat, k, cutoff_threshold)
    """
    # Check input types and formats
    # assert isinstance(obs_X, csr_matrix), "obs_X must be a scipy csr_matrix"
    assert isinstance(obs_knn_dist, np.ndarray), "obs_knn_dist must be a numpy ndarray"
    assert isinstance(covar_mat, np.ndarray), "covar_mat must be a numpy ndarray"
    assert obs_X.shape[0] == obs_knn_dist.shape[0] == covar_mat.shape[0], "Inconsistent input shapes"
    assert obs_knn_dist.shape[1] == k, "Number of nearest neighbors in obs_knn_dist does not match specified k"
    # Convert to PyTorch tensors
    #obs_X_torch = convert_to_torch_sparse(obs_X)
    obs_X_torch = torch.tensor(obs_X)
    obs_knn_dist_torch = torch.tensor(obs_knn_dist)
    covar_mat_torch = torch.tensor(covar_mat).float()
    # Perform nearest neighbors search
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(obs_X_torch)
    obs_knn_dist, obs_knn_adj_list = nbrs.kneighbors(obs_X_torch, return_distance=True)
    # Create mask
    knn_mask = mask_knn(obs_knn_dist, cutoff_threshold=cutoff_threshold)
    # Compute the covariate difference
    covar_diff = compute_covariate_difference(obs_knn_adj_list, covar_mat_torch)
    # Compute the distance correction with mask
    dist_correction, betas = compute_distance_correction_simplified_with_beta_inclusion(
        covar_diff,
        obs_knn_dist_torch,
        inclusion_mask=knn_mask,
        locally_weighted=locally_weighted
    )
    # Correct the observed distances with mask
    corrected_obs_knn_dist = correct_observed_distances_mask(
        obs_knn_dist_torch,
        dist_correction, 
        knn_mask
    )
    # Convert corrected distances back to numpy and return
    return corrected_obs_knn_dist.numpy()
