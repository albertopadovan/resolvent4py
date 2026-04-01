import numpy as np
import scipy as sp
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_create_dense_matrix(comm, square_matrix_size):
    r"""Test create_dense_matrix creates a matrix of correct size"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    M = res4py.create_dense_matrix(comm, ((Nl, N), (Nl, N)))
    sizes = M.getSizes()
    assert sizes[0][-1] == N
    assert sizes[-1][-1] == N
    M.destroy()


def test_create_AIJ_identity(comm, square_matrix_size):
    r"""Test that create_AIJ_identity produces an actual identity matrix"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    Id = res4py.create_AIJ_identity(comm, ((Nl, N), (Nl, N)))

    # Multiply by a random vector, should return the same vector
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = x.duplicate()
    Id.mult(x, y)

    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(ys.getArray() - xpython)
    ys.destroy()
    x.destroy()
    y.destroy()
    Id.destroy()
    assert error < 1e-14


def test_hermitian_transpose_out_of_place(comm, square_matrix_size):
    r"""Test hermitian_transpose with in_place=False"""
    N = square_matrix_size[0]
    Apetsc, Apython = pytest_utils.generate_random_matrix(comm, (N, N))

    # Compute A^H via the utility
    Ah = Apetsc.copy()
    Ah.hermitianTranspose()

    # Verify by multiplying A^H * x and comparing to python
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = x.duplicate()
    Ah.mult(x, y)

    ys = res4py.distributed_to_sequential_vector(y)
    expected = Apython.conj().T.dot(xpython)
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(
        expected
    )
    ys.destroy()
    x.destroy()
    y.destroy()
    Apetsc.destroy()
    Ah.destroy()
    assert error < 1e-10


def test_mat_solve_hermitian_transpose(comm, square_random_matrix):
    r"""Test mat_solve_hermitian_transpose: A^{-H} X"""
    from resolvent4py.utils.matrix import mat_solve_hermitian_transpose

    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    ksp = res4py.create_mumps_solver(Apetsc)

    s = 3
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Xm = X.getMat()
    Ym = mat_solve_hermitian_transpose(ksp, Xm)
    X.restoreMat(Xm)

    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Ypython = sp.linalg.inv(Apython).conj().T.dot(Xpython)
    error = np.linalg.norm(
        Yms.getDenseArray() - Ypython
    ) / np.linalg.norm(Ypython)
    Yms.destroy()
    Ym.destroy()
    X.destroy()
    ksp.destroy()
    assert error < 1e-8


def test_assemble_harmonic_resolvent_generator(comm, square_matrix_size):
    r"""Test assemble_harmonic_resolvent_generator produces T = -iwM + A
    (with M = Id by default)"""
    from resolvent4py.utils.matrix import assemble_harmonic_resolvent_generator

    N = square_matrix_size[0]
    nblocks = 3
    Ntotal = N * nblocks
    Nl = res4py.compute_local_size(Ntotal)
    Apetsc, Apython = pytest_utils.generate_random_matrix(
        comm, (Ntotal, Ntotal)
    )

    omega = 1.5
    freqs = omega * np.array([-1, 0, 1])
    T = assemble_harmonic_resolvent_generator(Apetsc, freqs)

    # Expected: T = A + diag(-i*freq_k * I_N) for each block k
    Tpython = Apython.copy()
    for k in range(nblocks):
        Tpython[k * N : (k + 1) * N, k * N : (k + 1) * N] += (
            -1j * freqs[k] * np.eye(N)
        )

    # Verify via matvec
    x, xpython = pytest_utils.generate_random_vector(comm, Ntotal)
    y = x.duplicate()
    T.mult(x, y)
    ys = res4py.distributed_to_sequential_vector(y)
    expected = Tpython.dot(xpython)
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(
        expected
    )
    ys.destroy()
    x.destroy()
    y.destroy()
    T.destroy()
    Apetsc.destroy()
    assert error < 1e-10
