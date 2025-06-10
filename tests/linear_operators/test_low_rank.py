import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


def test_low_rank_on_vectors(comm, rectangular_matrix_size):
    r"""Test LowRankLinearOperator on vectors"""
    Nr, Nc = rectangular_matrix_size
    rr, rc = 5, 9
    U, Upython = pytest_utils.generate_random_bv(comm, (Nr, rr))
    V, Vpython = pytest_utils.generate_random_bv(comm, (Nc, rc))
    S = np.random.randn(rr, rc) + 1j * np.random.randn(rr, rc)
    S = comm.bcast(S, root=0)
    A = Upython @ S @ Vpython.conj().T
    linop = res4py.linear_operators.LowRankLinearOperator(comm, U, S, V, None)

    x, xpython = pytest_utils.generate_random_vector(comm, Nc)
    y = linop.create_left_vector()
    error_vec = []
    error_vec.append(
        pytest_utils.compute_error_vector(
            comm, linop.apply, x, y, A.dot, xpython
        )
    )
    x.destroy()
    y.destroy()
    x, xpython = pytest_utils.generate_random_vector(comm, Nr)
    y = linop.create_right_vector()
    error_vec.append(
        pytest_utils.compute_error_vector(
            comm,
            linop.apply_hermitian_transpose,
            x,
            y,
            A.conj().T.dot,
            xpython,
        )
    )
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    assert error < 1e-8


def test_low_rank_on_bvs(comm, rectangular_matrix_size):
    r"""Test LowRankLinearOperator on vectors"""
    Nr, Nc = rectangular_matrix_size
    rr, rc = 5, 9
    U, Upython = pytest_utils.generate_random_bv(comm, (Nr, rr))
    V, Vpython = pytest_utils.generate_random_bv(comm, (Nc, rc))
    S = np.random.randn(rr, rc) + 1j * np.random.randn(rr, rc)
    S = comm.bcast(S, root=0)
    A = Upython @ S @ Vpython.conj().T
    linop = res4py.linear_operators.LowRankLinearOperator(comm, U, S, V, None)

    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (Nc, s))
    Y = linop.create_left_bv(s)
    error_vec = []
    error_vec.append(
        pytest_utils.compute_error_bv(
            comm, linop.apply_mat, X, Y, A.dot, Xpython
        )
    )
    X.destroy()
    Y.destroy()
    X, Xpython = pytest_utils.generate_random_bv(comm, (Nr, s))
    Y = linop.create_right_bv(s)
    error_vec.append(
        pytest_utils.compute_error_bv(
            comm,
            linop.apply_hermitian_transpose_mat,
            X,
            Y,
            A.conj().T.dot,
            Xpython,
        )
    )
    error = np.linalg.norm(error_vec)
    X.destroy()
    Y.destroy()
    linop.destroy()
    assert error < 1e-8
