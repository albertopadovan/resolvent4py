import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_compute_local_size_sums_to_global(comm):
    r"""Test that local sizes across all ranks sum to the global size"""
    for N in [50, 51, 100, 103]:
        Nl = res4py.compute_local_size(N)
        total = comm.tompi4py().allreduce(Nl)
        assert total == N


def test_distributed_to_sequential_vector_roundtrip(comm, square_matrix_size):
    r"""Test dist->seq->dist roundtrip for vectors"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    x_seq = res4py.distributed_to_sequential_vector(x)
    # Verify sequential vector matches python reference
    error = np.linalg.norm(x_seq.getArray() - xpython)
    x_seq.destroy()
    x.destroy()
    assert error < 1e-14


def test_sequential_to_distributed_vector(comm, square_matrix_size):
    r"""Test scattering a sequential vector to distributed"""
    N = square_matrix_size[0]
    # Create sequential reference on all ranks
    np.random.seed(42)
    array = np.random.randn(N) + 1j * np.random.randn(N)

    vec_seq = PETSc.Vec().createWithArray(array.copy(), comm=PETSc.COMM_SELF)
    Nl = res4py.compute_local_size(N)
    vec_dist = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec_dist.setSizes((Nl, N))
    vec_dist.setType("standard")

    vec_dist = res4py.sequential_to_distributed_vector(vec_seq, vec_dist)

    # Gather back and verify
    vec_check = res4py.distributed_to_sequential_vector(vec_dist)
    error = np.linalg.norm(vec_check.getArray() - array)
    vec_seq.destroy()
    vec_dist.destroy()
    vec_check.destroy()
    assert error < 1e-14


def test_distributed_to_sequential_matrix_roundtrip(comm, square_matrix_size):
    r"""Test dist->seq roundtrip for dense matrices"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Xm = X.getMat()
    Xms = res4py.distributed_to_sequential_matrix(Xm)
    X.restoreMat(Xm)
    error = np.linalg.norm(Xms.getDenseArray() - Xpython) / np.linalg.norm(
        Xpython
    )
    Xms.destroy()
    X.destroy()
    assert error < 1e-14


def test_scatter_array_from_root_to_all(comm):
    r"""Test scattering a numpy array from root to all ranks"""
    rank = comm.getRank()
    N = 53  # intentionally not divisible by common rank counts
    array = None
    if rank == 0:
        np.random.seed(42)
        array = np.random.randn(N)

    local_array = res4py.scatter_array_from_root_to_all(array)

    # Gather back and verify
    all_arrays = comm.tompi4py().gather(local_array, root=0)
    if rank == 0:
        reconstructed = np.concatenate(all_arrays)
        np.random.seed(42)
        expected = np.random.randn(N)
        error = np.linalg.norm(reconstructed - expected)
        assert error < 1e-14


def test_scatter_array_with_custom_locsize(comm):
    r"""Test scatter_array_from_root_to_all with explicit local sizes"""
    rank = comm.getRank()
    pool_size = comm.getSize()
    N = 50
    locsize = N // pool_size
    # Only first ranks get data; use uniform split for simplicity
    total = locsize * pool_size
    array = None
    if rank == 0:
        np.random.seed(123)
        array = np.random.randn(total)

    local_array = res4py.scatter_array_from_root_to_all(array, locsize)
    assert len(local_array) == locsize

    all_arrays = comm.tompi4py().gather(local_array, root=0)
    if rank == 0:
        reconstructed = np.concatenate(all_arrays)
        np.random.seed(123)
        expected = np.random.randn(total)
        error = np.linalg.norm(reconstructed - expected)
        assert error < 1e-14


def test_gather_vec_to_rank_root_zero(comm):
    r"""gather_vec_to_rank returns the full array on the root rank and
    None everywhere else, with values matching the distributed vector."""
    N = 47
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    arr = res4py.gather_vec_to_rank(x, dest_rank=0)
    if comm.getRank() == 0:
        assert arr is not None
        assert arr.shape == (N,)
        err = np.linalg.norm(arr - xpython)
        assert err < 1e-14, f"gathered array differs by {err:.3e}"
    else:
        assert arr is None
    x.destroy()


def test_gather_vec_to_rank_non_zero_root(comm):
    r"""gather_vec_to_rank works with a non-zero destination rank."""
    if comm.getSize() < 2:
        return  # trivial single-rank case: dest_rank=0 already covered
    N = 33
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    dest = 1
    arr = res4py.gather_vec_to_rank(x, dest_rank=dest)
    if comm.getRank() == dest:
        assert arr is not None
        err = np.linalg.norm(arr - xpython)
        assert err < 1e-14
    else:
        assert arr is None
    x.destroy()


def test_scatter_vec_from_rank_from_root(comm):
    r"""scatter_vec_from_rank writes a source-rank numpy array into a
    fresh distributed Vec so its concatenated slices match the source."""
    N = 47
    rank = comm.getRank()
    arr_on_source = None
    if rank == 0:
        np.random.seed(2024)
        arr_on_source = (
            np.random.randn(N) + 1j * np.random.randn(N)
        ).astype(PETSc.ScalarType)

    Nl = res4py.compute_local_size(N)
    target = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    target.setSizes((Nl, N))
    target.setType("standard")

    res4py.scatter_vec_from_rank(arr_on_source, target, source_rank=0)

    # Gather back and compare on rank 0.
    check = res4py.distributed_to_sequential_vector(target)
    if rank == 0:
        err = np.linalg.norm(check.getArray() - arr_on_source)
        assert err < 1e-14, f"scattered vec differs by {err:.3e}"
    check.destroy()
    target.destroy()


def test_gather_scatter_roundtrip(comm):
    r"""gather_vec_to_rank then scatter_vec_from_rank reproduces the
    original distributed vector bit-for-bit."""
    N = 61
    root = comm.getSize() - 1  # last rank
    x, _ = pytest_utils.generate_random_vector(comm, N)

    arr = res4py.gather_vec_to_rank(x, dest_rank=root)

    y = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    y.setSizes(x.getSizes())
    y.setType("standard")
    res4py.scatter_vec_from_rank(arr, y, source_rank=root)

    # Compare local slices.
    err_local = np.linalg.norm(y.getArray() - x.getArray()) ** 2
    err = np.sqrt(comm.tompi4py().allreduce(err_local))
    assert err < 1e-14, f"roundtrip err = {err:.3e}"
    x.destroy()
    y.destroy()
