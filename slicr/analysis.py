## analysis.py
import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .utils import convert_to_torch_sparse
from .correction import compute_covariate_difference, compute_distance_correction_simplified, correct_observed_distances, compute_distance_correction_simplified_with_beta_inclusion, correct_observed_distances_mask, resort_order, remeasure_distances
# , expand_knn_list, prune_knn_list, iterative_update, prune_only, inv_min_max_norm, G_from_adj_and_dist
from .graph import mask_knn, get_re_expanded_adj_and_dist, prune_only_mask, prune_only_mask_percent, mask_knn_local_diff_dist
from .results import AnalysisResults


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


def do_mask_and_regression(obs_knn_adj_list, 
                           obs_knn_dist_torch, 
                           covar_mat_torch, 
                           cutoff_threshold, 
                           original_mask, 
                           min_k, 
                           local_cutoff_threshold=3,
                           locally_weighted=False, 
                           local_mask_only=False, 
                           skip_mask = False, 
                           skip_mean_mask = False):
    # Create mask
    print("original mask:")
    print(original_mask)
    assert sanity_check_for_adj(
        obs_knn_adj_list, original_mask), "failed sanity check at point 3c"
    if local_mask_only:
        knn_mask = torch.tensor(
            mask_knn_local_diff_dist(
                obs_knn_dist_torch, original_mask, cutoff_threshold=local_cutoff_threshold, min_k=min_k),
            dtype=torch.bool
        )
    else:
        if skip_mask:
            knn_mask = torch.ones_like(obs_knn_adj_list, dtype=torch.bool)
        else:
            knn_mask = torch.tensor(mask_knn(
                obs_knn_dist_torch, 
                cutoff_threshold=cutoff_threshold, 
                skip_mean_mask=skip_mean_mask
                ), dtype=torch.bool
            )
            knn_mask = knn_mask * torch.tensor(original_mask, dtype=torch.bool)
    assert sanity_check_for_adj(
        obs_knn_adj_list, original_mask), "failed sanity check at point 3d_old"
    #print("original_mask[129, :]", original_mask[129, :])
    #print("obs_knn_dist_torch[129, :]", obs_knn_dist_torch[129, :])
    #print("knn_mask[129,:]",knn_mask[129,:])
    assert sanity_check_for_adj(
        obs_knn_adj_list, knn_mask), "failed sanity check at point 3d_new"
    print("updated mask:")
    print(knn_mask)
    # Compute the covariate difference
    covar_diff = compute_covariate_difference(
        obs_knn_adj_list, covar_mat_torch)
    # Compute the distance correction with mask
    dist_correction, betas = compute_distance_correction_simplified_with_beta_inclusion(
        covar_diff,
        obs_knn_dist_torch,
        inclusion_mask=knn_mask,
        locally_weighted=locally_weighted
    )
    assert sanity_check_for_adj(
        obs_knn_adj_list, knn_mask), "failed sanity check at point 3e"
    print("dist_correction[0,:]",dist_correction[0,:])
    print("obs_knn_dist_torch[0,:]", obs_knn_dist_torch[0, :])
    # Correct the observed distances with mask
    corrected_obs_knn_dist = correct_observed_distances_mask(
        obs_knn_dist_torch,
        dist_correction,
        knn_mask
    )
    assert sanity_check_for_adj(
        obs_knn_adj_list, knn_mask), "failed sanity check at point 3f"
    corrected_adj, corrected_obs_knn_dist, knn_mask = resort_order(
        obs_knn_adj_list,
        corrected_obs_knn_dist,
        knn_mask,
        min_k
    )
    assert sanity_check_for_adj(
        obs_knn_adj_list, knn_mask), "failed sanity check at point 3g"
    # Compute the mean of the absolute values of betas along axis 0
    abs_beta = torch.mean(torch.abs(betas), dim=0)
    # Compute the sum of the absolute betas
    total_beta = torch.sum(abs_beta)
    return corrected_adj, corrected_obs_knn_dist, knn_mask, betas, total_beta


def perform_analysis_with_mask(obs_X, covar_mat, k, cutoff_threshold, min_k, results=None, detailed_log=False, locally_weighted=False):
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
    >>> k = 20
    >>> min_k = 10
    >>> cutoff_threshold = 1.0
    >>> corrected_obs_knn_dist = perform_analysis_with_mask(obs_X, obs_knn_dist, covar_mat, k, cutoff_threshold, min_k)
    """
    # Check input types and formats
    # assert isinstance(obs_X, csr_matrix), "obs_X must be a scipy csr_matrix"
    #assert isinstance(covar_mat, np.ndarray), "covar_mat must be a numpy ndarray"
    assert obs_X.shape[0] == covar_mat.shape[0], "Inconsistent input shapes"
    # Convert to PyTorch tensors
    #obs_X_torch = convert_to_torch_sparse(obs_X)
    obs_X_torch = torch.tensor(obs_X)
    covar_mat_torch = torch.tensor(covar_mat).float()
    # Perform nearest neighbors search
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(obs_X_torch)
    obs_knn_dist_torch, obs_knn_adj_list = nbrs.kneighbors(obs_X_torch, return_distance=True)
    obs_knn_dist_torch = torch.tensor(obs_knn_dist_torch)
    obs_knn_adj_list = torch.tensor(obs_knn_adj_list, dtype=torch.long)
    original_mask = torch.ones_like(obs_knn_adj_list, dtype=torch.bool)
    if detailed_log:
        results.log_results(obs_knn_adj_list.clone(), obs_knn_dist_torch.clone(), original_mask.clone(), None, None)
    corrected_adj, corrected_obs_knn_dist, knn_mask, betas, total_beta = do_mask_and_regression(
        obs_knn_adj_list, obs_knn_dist_torch,
        covar_mat_torch, cutoff_threshold, original_mask, min_k)
    if detailed_log:
        results.log_results(
            corrected_adj.clone(), 
            corrected_obs_knn_dist.clone(),
            knn_mask.clone(), 
            betas,
            total_beta)
    # Convert corrected distances back to numpy and return
    return corrected_adj, corrected_obs_knn_dist, knn_mask, betas, total_beta


def sanity_check_for_adj(temp_adj, temp_mask):
    print("sanity check:")
    for i in range(temp_adj.shape[0]):
        temp_unq = torch.unique(temp_adj[i,temp_mask[i]])
        #print("temp_unq")
        #print(temp_unq)
        #print("temp_adj.shape[1]", temp_adj.shape[1])
        #print("temp_mask[i].sum()",temp_mask[i].sum())
        if temp_unq.shape[0] < temp_adj.shape[1] and i % int(temp_adj.shape[0]/2) == 0:
            print("temp_adj[i,:]")
            print(temp_adj[i, :])
            print("temp_mask[i]")
            print(temp_mask[i])
        if temp_unq.shape[0]<temp_mask[i].sum():
            print("temp_adj[i,:]")
            print(temp_adj[i, :])
            print("temp_mask[i]")
            print(temp_mask[i])
            print("temp_adj[i,temp_mask[i]]")
            print(temp_adj[i,temp_mask[i]])
            print("temp_unq:",temp_unq)
            return(False)
    return(True)


def check_adj_conservation(adj_1, mask_1, adj_2, mask_2):
    # Takes in two masked adj lists and their masks
    # then for each row, it checks the consistency between 
    # the two adjacency lists to see what percent of the 
    return()



def slicr_analysis(obs_X, 
                   covar_mat, 
                   k, 
                   cutoff_threshold=0,
                   local_cutoff_threshold=3,
                   min_k=10,
                   relative_beta_removal=0.15,
                   shrink_percentage=0.75,
                   max_iters=100,
                   detailed_log=False,
                   run_name=""):
    assert k>min_k, "the min_k variable must be higher than k"
    results = AnalysisResults()
    ## initialize the correction
    covar_mat = torch.tensor(covar_mat)
    obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, betas, temp_total_beta = perform_analysis_with_mask(
        obs_X, covar_mat, k, cutoff_threshold, min_k, results, detailed_log = detailed_log)
    assert torch.all(corrected_obs_knn_dist >=
                     0), "There's a bug. The first round distances have negatives"
    # log the mean absolute betas to quantify magnitude of 
    # covariate effects
    beta_list = [betas.clone()]
    # Also catelogue the sum of the total effects, so that we can 
    total_beta = [temp_total_beta]
    print("average total covariate effect:", temp_total_beta)
    #print("mean absolute betas for initialization round",":")
    #print(beta_list[-1])
    # we'll stop correcting either the first time that
    # the updated beta is greater than the last one
    temp_iter = 0
    converged = False
    early_stop = False
    while ((temp_iter<max_iters) and (converged==False) or early_stop):
        temp_iter+=1
        # prune
        #print(obs_knn_adj_list.shape)
        #print(corrected_obs_knn_dist.shape)
        #print(knn_mask.shape)
        #print(int(round(k/2)))
        assert sanity_check_for_adj(
            obs_knn_adj_list, knn_mask), "failed sanity check at point 1"
        #new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask = prune_only_mask(
        #    obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, max(min_k, int(round(k*shrink_percentage))))
        new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask = prune_only_mask_percent(
            obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, min_k, shrink_percentage)
        assert sanity_check_for_adj(
            new_obs_knn_adj_list, new_knn_mask), "failed sanity check at point 2"
        #print("post prune:")
        #print(obs_knn_adj_list.shape)
        #print(corrected_obs_knn_dist.shape)
        #print(knn_mask.shape)
        # expand to updated nearest neighbors
        new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask = get_re_expanded_adj_and_dist(
            new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask, k, local_iter=temp_iter)
        assert sanity_check_for_adj(
            new_obs_knn_adj_list, new_knn_mask), "failed sanity check at point 3"
        assert torch.all(new_corrected_obs_knn_dist >=
                         0), "There's a bug. The expanded distances have negatives"
        ## Now we'll recalculate the distances
        new_corrected_obs_knn_dist = remeasure_distances(new_obs_knn_adj_list, obs_X)
        assert sanity_check_for_adj(
            new_obs_knn_adj_list, new_knn_mask), "failed sanity check at point 3a"
        ## and resort them
        new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask = resort_order(
            new_obs_knn_adj_list,
            new_corrected_obs_knn_dist,
            new_knn_mask,
            min_k
        )
        assert sanity_check_for_adj(
            new_obs_knn_adj_list, new_knn_mask), "failed sanity check at point 3b"
        if detailed_log:
            results.log_results(
                new_obs_knn_adj_list.clone(),
                new_corrected_obs_knn_dist.clone(),
                new_knn_mask.clone(),
                None,
                None)
        ## from here on out, we allow the points to crawl
        new_obs_knn_adj_list, new_corrected_obs_knn_dist, new_knn_mask, betas, temp_total_beta = do_mask_and_regression(
            new_obs_knn_adj_list, new_corrected_obs_knn_dist,
            covar_mat, cutoff_threshold, new_knn_mask, min_k, local_cutoff_threshold=local_cutoff_threshold, local_mask_only=True, skip_mask=False, skip_mean_mask=True)  # locally_weighted=False
        if detailed_log:
            results.log_results(
                new_obs_knn_adj_list.clone(),
                new_corrected_obs_knn_dist.clone(),
                new_knn_mask.clone(),
                betas.clone(),
                temp_total_beta)
        assert sanity_check_for_adj(
            new_obs_knn_adj_list, new_knn_mask), "failed sanity check at point 4"
        # Also catelogue the sum of the total effects, so that we can
        total_beta.append(temp_total_beta)
        ## check for convergance
        percent_removed = 1-(total_beta[-1]/total_beta[0])
        print("\n\n", run_name)
        print("iter:",temp_iter)
        print("percent local covariate effect removed:", percent_removed)
        print(total_beta[-1], "/", total_beta[0],
                "=", total_beta[-1]/total_beta[0])
        beta_list.append(betas.clone())
        print("mean absolute betas for round",temp_iter,":")
        print(beta_list[-1])
        ## check for early stopping
        early_stop = False
        if len(total_beta)>2:
            ## Changed my mind, you can actually have this happen
            ## when you are still accurately correcting
            if False:#total_beta[-1]>total_beta[-2]:
                ## if it started to get worse again
                early_stop = True
                ## this means that we'll return the prior round's results
                ## so don't update old and new
                return (obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, beta_list, total_beta)
        if (percent_removed > relative_beta_removal):
            converged=True
        obs_knn_adj_list=new_obs_knn_adj_list
        corrected_obs_knn_dist=new_corrected_obs_knn_dist
        knn_mask = new_knn_mask
    return (obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, beta_list, total_beta)


""" 
a,d,m,beta_list, total_beta = slicr_analysis(real_X, covar_mat_st.to_numpy(),20,1)

pa, pd, pm = prune_only_mask(
            a.numpy().astype(int), 
            d.numpy(), 
            m.numpy(),
            10)
G2 = G_from_adj_and_dist_mask(pa, pd, pm)
pos2 = nx.spring_layout(
    G2,
    pos=pos_orig,#result_dict['k_100_thresh_2.5_frac_1']['end_positions'],
    weight="none",
    seed=123456)

pos_orig_array = convert_pos_dict_to_array(pos_orig)
pos_array = convert_pos_dict_to_array(pos2)
plt.scatter(pos_orig_array[:, 0], pos_orig_array[:, 1], c=cm.get_cmap('inferno')(covar_mat_st.to_numpy()[:, -1]))
plt.show()

plt.scatter(pos_array[:,0],pos_array[:,1],c= cm.get_cmap('inferno')(covar_mat_st.to_numpy()[:,-1]))
plt.show()

"""