import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.utils.vector import reshape_harmonic_balanced_vector_into_bv
from resolvent4py.utils.comms import compute_local_size
from .. import pytest_utils


def test_vec_real_inplace(comm, square_matrix_size):
    r"""Test vec_real with inplace=True"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=True)
    x_orig_id = x.getBuffer()  # to check identity
    y = res4py.vec_real(x, inplace=True)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(ys.getArray() - xpython.real)
    ys.destroy()
    x.destroy()
    assert error < 1e-14


def test_vec_real_copy(comm, square_matrix_size):
    r"""Test vec_real with inplace=False (returns new vector)"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=True)
    y = res4py.vec_real(x, inplace=False)
    # y should be the real part
    ys = res4py.distributed_to_sequential_vector(y)
    error_y = np.linalg.norm(ys.getArray() - xpython.real)
    ys.destroy()
    # x should be unchanged
    xs = res4py.distributed_to_sequential_vector(x)
    error_x = np.linalg.norm(xs.getArray() - xpython)
    xs.destroy()
    x.destroy()
    y.destroy()
    assert error_y < 1e-14
    assert error_x < 1e-14


def test_vec_imag_inplace(comm, square_matrix_size):
    r"""Test vec_imag with inplace=True"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=True)
    y = res4py.vec_imag(x, inplace=True)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(ys.getArray() - xpython.imag)
    ys.destroy()
    x.destroy()
    assert error < 1e-14


def test_vec_imag_copy(comm, square_matrix_size):
    r"""Test vec_imag with inplace=False (returns new vector)"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=True)
    y = res4py.vec_imag(x, inplace=False)
    ys = res4py.distributed_to_sequential_vector(y)
    error_y = np.linalg.norm(ys.getArray() - xpython.imag)
    ys.destroy()
    # x should be unchanged
    xs = res4py.distributed_to_sequential_vector(x)
    error_x = np.linalg.norm(xs.getArray() - xpython)
    xs.destroy()
    x.destroy()
    y.destroy()
    assert error_y < 1e-14
    assert error_x < 1e-14


def test_vec_real_of_real_vector(comm, square_matrix_size):
    r"""Test vec_real on an already-real vector is a no-op"""
    N = square_matrix_size[0]
    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=False)
    y = res4py.vec_real(x, inplace=False)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(ys.getArray() - xpython.real)
    ys.destroy()
    x.destroy()
    y.destroy()
    assert error < 1e-14


def test_enforce_and_check_complex_conjugacy(comm, square_matrix_size):
    r"""Test enforce_complex_conjugacy followed by check_complex_conjugacy"""
    N = square_matrix_size[0]
    nblocks = 5
    block_size = N // nblocks
    total_size = block_size * nblocks
    Nl = res4py.compute_local_size(total_size)
    x, _ = pytest_utils.generate_random_vector(comm, total_size, complex=True)

    # Before enforcement, conjugacy is generally not satisfied
    res4py.enforce_complex_conjugacy(comm, x, nblocks)
    result = res4py.check_complex_conjugacy(comm, x, nblocks)
    assert result == True
    x.destroy()


def test_check_complex_conjugacy_fails_for_random(comm, square_matrix_size):
    r"""Test that a random vector does NOT satisfy complex conjugacy"""
    N = square_matrix_size[0]
    nblocks = 5
    block_size = N // nblocks
    total_size = block_size * nblocks
    x, _ = pytest_utils.generate_random_vector(comm, total_size, complex=True)
    result = res4py.check_complex_conjugacy(comm, x, nblocks)
    # A random complex vector should almost certainly not be conjugate-symmetric
    # (this is a statistical test; probability of accidental pass is negligible)
    assert result == False
    x.destroy()


def test_enforce_complex_conjugacy_even_blocks_raises(comm):
    r"""Test that even nblocks raises ValueError"""
    N = 20
    x, _ = pytest_utils.generate_random_vector(comm, N, complex=True)
    try:
        res4py.enforce_complex_conjugacy(comm, x, 4)
        raised = False
    except ValueError:
        raised = True
    x.destroy()
    assert raised


def test_reshape_harmonic_balanced_vector_into_bv(comm):
    r"""Test that reshaping a stacked vector into BV gives the correct
    n x nblocks matrix."""
    n = 4
    nblocks = 5
    N = n * nblocks

    vec, arr = pytest_utils.generate_random_vector(comm, N)
    bv = reshape_harmonic_balanced_vector_into_bv(vec, nblocks)

    # Expected: column j of the n x nblocks matrix is arr[j*n : (j+1)*n]
    expected = arr.reshape(nblocks, n).T

    bvMat = bv.getMat()
    bvMat_seq = res4py.distributed_to_sequential_matrix(bvMat)
    bv.restoreMat(bvMat)
    result = bvMat_seq.getDenseArray().copy()
    bvMat_seq.destroy()

    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    vec.destroy()
    bv.destroy()
    assert error < 1e-12, f"Relative error: {error:.2e}"


def test_reshape_harmonic_balanced_vector_into_bv_preallocated(comm):
    r"""Test reshaping into a pre-allocated BV."""
    n = 6
    nblocks = 3
    N = n * nblocks

    vec, arr = pytest_utils.generate_random_vector(comm, N)

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
    vec.destroy()
    bv.destroy()
    assert error < 1e-12, f"Relative error: {error:.2e}"


def test_reshape_harmonic_balanced_vector_into_bv_single_block(comm):
    r"""Edge case: nblocks = 1, the BV is just a single column."""
    n = 8
    nblocks = 1

    vec, arr = pytest_utils.generate_random_vector(comm, n)
    bv = reshape_harmonic_balanced_vector_into_bv(vec, nblocks)

    expected = arr.reshape(1, n).T

    bvMat = bv.getMat()
    bvMat_seq = res4py.distributed_to_sequential_matrix(bvMat)
    bv.restoreMat(bvMat)
    result = bvMat_seq.getDenseArray().copy()
    bvMat_seq.destroy()

    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    vec.destroy()
    bv.destroy()
    assert error < 1e-12, f"Relative error: {error:.2e}"
