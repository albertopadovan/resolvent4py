import numpy as np
from petsc4py import PETSc
from resolvent4py.utils.matrix import convert_coo_to_csr, convert_coo_to_csr_v2
from resolvent4py import compute_local_size


def test_convert_coo_to_csr_v2_matches_v1(comm):
    r"""Test that convert_coo_to_csr_v2 produces the same output as
    convert_coo_to_csr"""
    N = 50
    nnz = 200
    Nl = compute_local_size(N)
    sizes = ((Nl, N), (Nl, N))

    rank = comm.getRank()
    np.random.seed(42 + rank)
    # Generate random COO entries (rows may belong to any rank)
    rows = np.random.randint(0, N, size=nnz)
    cols = np.random.randint(0, N, size=nnz)
    vals = np.random.randn(nnz) + 1j * np.random.randn(nnz)

    row_ptr_v1, cols_v1, vals_v1 = convert_coo_to_csr(
        [rows.copy(), cols.copy(), vals.copy()], sizes
    )
    row_ptr_v2, cols_v2, vals_v2 = convert_coo_to_csr_v2(
        [rows.copy(), cols.copy(), vals.copy()], sizes
    )

    assert np.array_equal(row_ptr_v1, row_ptr_v2)
    # Within each row, column order may differ, so compare row by row
    for i in range(len(row_ptr_v1) - 1):
        s, e = row_ptr_v1[i], row_ptr_v1[i + 1]
        idx_v1 = np.argsort(cols_v1[s:e])
        idx_v2 = np.argsort(cols_v2[s:e])
        assert np.array_equal(cols_v1[s:e][idx_v1], cols_v2[s:e][idx_v2])
        assert np.allclose(vals_v1[s:e][idx_v1], vals_v2[s:e][idx_v2])
