import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


def test_shift_and_scale_on_vectors(comm, square_random_matrix):
    r"""Test ShiftAndScaleLinearOperator on vectors"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    alpha = 2.0 + 3.0j
    beta = -1.5 + 0.5j

    linop_A = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.ShiftAndScaleLinearOperator(
        linop_A, alpha, beta
    )

    Lpython = alpha * np.eye(N) + beta * Apython
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = linop.create_left_vector()

    actions_python = [Lpython.dot, Lpython.conj().T.dot]
    actions_petsc = [linop.apply, linop.apply_hermitian_transpose]

    error_vec = [
        pytest_utils.compute_error_vector(
            comm, actions_petsc[i], x, y, actions_python[i], xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    assert error < 1e-10


def test_shift_and_scale_on_bvs(comm, square_random_matrix):
    r"""Test ShiftAndScaleLinearOperator on BVs"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    alpha = 2.0 + 3.0j
    beta = -1.5 + 0.5j

    linop_A = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.ShiftAndScaleLinearOperator(
        linop_A, alpha, beta
    )

    Lpython = alpha * np.eye(N) + beta * Apython
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, 5))
    Y = linop.create_left_bv(X.getSizes()[-1])

    actions_python = [Lpython.dot, Lpython.conj().T.dot]
    actions_petsc = [linop.apply_mat, linop.apply_hermitian_transpose_mat]

    error_vec = [
        pytest_utils.compute_error_bv(
            comm, actions_petsc[i], X, Y, actions_python[i], Xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    X.destroy()
    Y.destroy()
    linop.destroy()
    assert error < 1e-10
