import torch
from scipy.sparse import issparse, csr_matrix, csc_matrix


def get_sparse_corrected(adj, dist, mask, in_mat, covar_mat, sum_sq=False):
    ## TODO
    out_mat = csr_matrix(shape=in_mat.shape)
    return


def explore_residuals(adj, dist, mask, in_mat, covar_mat, sum_sq=False):
    """
    Takes as input an adjacency list and distance matrix in the format of 
    scikit-learn's Nearest Neighbor function, as well as a mask of the adj list,
    the input matrix to perform the analysis on, and the covariate mat that was
    used to adjust the graph.

    Goes through all nodes in the graph, 

    Returns an observation by 
    """
    assert in_mat.shape[0] == adj.shape[0]
    assert in_mat.shape[0] == dist.shape[0]
    assert in_mat.shape[0] == mask.shape[0]
    assert in_mat.shape[0] == covar_mat.shape[0]
    do_sparse = issparse(in_mat)
    # Create an empty csr destination matrix
    if do_sparse:
        return(get_sparse_corrected(adj, dist, mask, in_mat, covar_mat, sum_sq=sum_sq))
    out_mat = torch.zeros_like(in_mat)
    # Go through each node, and get the difference in the features,
    # Then regress out the difference in the residuals   
    # Now calculate the feature-leve average of the residuals across all neighbors
    for n in range(in_mat.shape[0]):
        # do on the fly tensor-ization in case we have a sparse matrix
        # This
        #temp_delta = torch.tensor(in_mat[n, :]) - torch.tensor(in_mat)[adj[n][mask[n]], :]
        # Vs this
        temp_delta = in_mat[n, :] - in_mat[adj[n][mask[n]], :]
        # Calculate the differences in the covariates
        temp_covar_delta = covar_mat[n, :] - covar_mat[adj[n][mask[n]], :]
        # Calculate betas and correct for the covariates
        beta = torch.pinverse(temp_covar_delta.T @
                            temp_covar_delta) @ temp_covar_delta.T @ temp_delta
        temp_correction = (temp_covar_delta @ beta).squeeze()
        # Now calculate the feature-level average of the residuals across all neighbors
        residuals = temp_delta - temp_correction
        #st, m = torch.std_mean(torch.tensor(in_mat)[adj[n][mask[n]], :], axis=0)
        residuals += in_mat[n, :]
        # Avg squared distance of the residuals
        if sum_sq:
            avg_resids = torch.mean(residuals**2, dim=0)
        else:
            avg_resids = torch.mean(residuals, dim=0)
        if do_sparse:
            pass
            #out_mat[n,:]=csr_matrix(avg_resids)
        else:
            out_mat[n, :]=avg_resids
    return (out_mat)


