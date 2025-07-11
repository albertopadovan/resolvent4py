import scipy as sp
import numpy as np
import resolvent4py as res4py


def test_randomized_svd(comm, rectangular_random_matrix):
    """Test randomized SVD with different matrix sizes based on test level."""
    Apetsc, Apython = rectangular_random_matrix
    krylov_dim = np.min([30, np.min(Apython.shape)])
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    n_cycles = 2
    r = np.min([3, krylov_dim])
    _, S, _ = res4py.linalg.randomized_svd(
        linop, linop.apply_mat, krylov_dim, n_cycles, r, 1
    )
    _, s, _ = sp.linalg.svd(Apython, full_matrices=False)
    error = 100 * np.max((np.diag(S) - s[:r]) / s[:r])
    assert error < 5e-1  # Max percent error < 5e-1
