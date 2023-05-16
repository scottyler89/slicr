
# SLICR

SLICR: Sparse Locally Involved Covariate Regression. A Python package for efficient, memory-conscious kNN distance correction, based on technical covariates whose effect should be removed.

## Installation

You can install the package using pip:
`pip install slicr`


## Usage

Here is a simple usage example:

```python
import numpy as np
from scipy.sparse import csr_matrix
from slicr import perform_analysis

# Create some dummy data
n = 1000  # number of nodes
g = 5  # number of covariates
k = 10  # number of nearest neighbors
obs_X = csr_matrix(np.random.rand(n, n))
obs_knn_dist = np.random.rand(n, k)
covar_mat = np.random.rand(n, g)

# Perform the analysis
corrected_obs_knn_dist = perform_analysis(obs_X, obs_knn_dist, covar_mat, k)
```



