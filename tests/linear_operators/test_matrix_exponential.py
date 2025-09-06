import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_matrix_exponential_on_vectors(comm, square_stable_random_matrix):
    r"""Test MatrixExponentialLinearOperator on vectors"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.MatrixExponentialLinearOperator(
        Alop, tf, dt
    )
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply, linop.apply_hermitian_transpose]

    y = linop.create_left_vector()
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
    assert error < 1e-6


def test_matrix_exponential_on_bvs(comm, square_stable_random_matrix):
    r"""Test MatrixExponentialLinearOperator on BVs"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.MatrixExponentialLinearOperator(
        Alop, tf, dt
    )
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 3))

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply_mat, linop.apply_hermitian_transpose_mat]

    Y = linop.create_left_bv(X.getSizes()[-1])
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
    assert error < 1e-6
