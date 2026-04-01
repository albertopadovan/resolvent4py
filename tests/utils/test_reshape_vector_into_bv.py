import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.utils.vector import reshape_harmonic_balanced_vector_into_bv
from resolvent4py.utils.comms import compute_local_size, scatter_array_from_root_to_all


def _build_distributed_vector(comm, arr_global):
    """Build a distributed PETSc vector from a numpy array known on rank 0."""
    N = comm.tompi4py().bcast(len(arr_global) if comm.getRank() == 0 else None, root=0)
    Nl = compute_local_size(N)
    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes((Nl, N))
    vec.setUp()

    # Scatter from rank 0
    rows_coo = None
    vals_coo = None
    if comm.getRank() == 0:
        rows_coo = np.arange(N, dtype=PETSc.IntType)
        vals_coo = np.asarray(arr_global, dtype=PETSc.ScalarType)
    rows = scatter_array_from_root_to_all(rows_coo)
    vals = scatter_array_from_root_to_all(vals_coo)
    vec.setValues(rows, vals)
    vec.assemble()
    return vec


def test_reshape_basic(comm):
    r"""Test that reshaping a stacked vector into BV gives the correct
    n x nblocks matrix."""
    n = 4
    nblocks = 5
    N = n * nblocks

    # Build a known vector on rank 0: vec = [v_0, v_1, ..., v_{nblocks-1}]
    # where v_j = (j+1) * [1, 2, ..., n]
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    arr = comm.tompi4py().bcast(arr, root=0)

    vec = _build_distributed_vector(comm, arr)
    bv = reshape_harmonic_balanced_vector_into_bv(vec, nblocks)

    # Expected: column j of the n x nblocks matrix is arr[j*n : (j+1)*n]
    expected = arr.reshape(nblocks, n).T  # shape (n, nblocks)

    # Gather BV to sequential dense matrix for comparison
    bvMat = bv.getMat()
    bvMat_seq = res4py.distributed_to_sequential_matrix(bvMat)
    bv.restoreMat(bvMat)
    result = bvMat_seq.getDenseArray().copy()
    bvMat_seq.destroy()

    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    assert error < 1e-12, f"Relative error: {error:.2e}"

    vec.destroy()
    bv.destroy()


def test_reshape_with_preallocated_bv(comm):
    r"""Test reshaping into a pre-allocated BV."""
    n = 6
    nblocks = 3
    N = n * nblocks

    rng = np.random.default_rng(99)
    arr = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    arr = comm.tompi4py().bcast(arr, root=0)

    vec = _build_distributed_vector(comm, arr)

    # Pre-allocate BV
    from slepc4py import SLEPc
    bv = SLEPc.BV().create(comm=comm)
    bv.setSizes((compute_local_size(n), n), nblocks)
    bv.setType("mat")

    bv = reshape_harmonic_balanced_vector_into_bv(vec, nblocks, bv=bv)

    expected = arr.reshape(nblocks, n).T

    bvMat = bv.getMat()
    bvMat_seq = res4py.distributed_to_sequential_matrix(bvMat)
    bv.restoreMat(bvMat)
    result = bvMat_seq.getDenseArray().copy()
    bvMat_seq.destroy()

    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    assert error < 1e-12, f"Relative error: {error:.2e}"

    vec.destroy()
    bv.destroy()


def test_reshape_single_block(comm):
    r"""Edge case: nblocks = 1, the BV is just a single column."""
    n = 8
    nblocks = 1

    rng = np.random.default_rng(7)
    arr = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    arr = comm.tompi4py().bcast(arr, root=0)

    vec = _build_distributed_vector(comm, arr)
    bv = reshape_harmonic_balanced_vector_into_bv(vec, nblocks)

    expected = arr.reshape(1, n).T  # n x 1

    bvMat = bv.getMat()
    bvMat_seq = res4py.distributed_to_sequential_matrix(bvMat)
    bv.restoreMat(bvMat)
    result = bvMat_seq.getDenseArray().copy()
    bvMat_seq.destroy()

    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    assert error < 1e-12, f"Relative error: {error:.2e}"

    vec.destroy()
    bv.destroy()
