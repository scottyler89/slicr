## utils.py
import torch
from scipy.sparse import csr_matrix


# Convert scipy sparse matrices to PyTorch sparse tensors
def convert_to_torch_sparse(X):
    """
    Convert a scipy sparse matrix to a PyTorch sparse tensor.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Input sparse matrix.
        
    Returns
    -------
    torch.Tensor
        The dense version of the PyTorch sparse tensor.
    """
    coo = X.tocoo()
    indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


