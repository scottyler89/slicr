
# SLICR

SLICR: Sparse Locally Involved Covariate Regression. A Python package for efficient, memory-conscious kNN distance correction, based on technical covariates whose effect should be removed.

## Installation

# TODO: upload to twine once we're happy with the algorithm
*This is in alpha mode, so I haven't uploaded to PyPi yet!* Just a placeholder
You can install the package using pip:
`python3 -m pip install slicr`


## Usage

Here is a simple usage example:

```python
import numpy as np
from scipy.sparse import csr_matrix
from slicr.analysis import slicr_analysis

# Create some dummy data
n = 1000  # number of nodes
g = 5  # number of covariates
k = 20  # number of nearest neighbors
dims = 50
obs_X = np.random.rand(n, dims)
obs_knn_dist = np.random.rand(n, k)
covar_mat = np.random.rand(n, g)

# Perform the analysis
results = slicr_analysis(obs_X, 
                   covar_mat, 
                   k, 
                   1)
```

# But let's be real - this isn't trivial, so what about a non-trivial example:
# TODO

