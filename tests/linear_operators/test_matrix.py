import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_matrix_on_vectors(comm, square_random_matrix):
    r"""Test MatrixLinearOperator on vectors"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    res4py.check_lu_factorization(Apetsc, ksp)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

    Apython_inv = sp.linalg.inv(Apython)
    actions_python = [
        Apython.dot,
        Apython.conj().T.dot,
        Apython_inv.dot,
        Apython_inv.conj().T.dot,
    ]
    actions_petsc = [
        linop.apply,
        linop.apply_hermitian_transpose,
        linop.solve,
        linop.solve_hermitian_transpose,
    ]

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
    assert error < 1e-10


def test_matrix_on_bvs(comm, square_random_matrix):
    r"""Test MatrixLinearOperator on BVs"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 5))

    Apython_inv = sp.linalg.inv(Apython)
    actions_python = [
        Apython.dot,
        Apython.conj().T.dot,
        Apython_inv.dot,
        Apython_inv.conj().T.dot,
    ]
    actions_petsc = [
        linop.apply_mat,
        linop.apply_hermitian_transpose_mat,
        linop.solve_mat,
        linop.solve_hermitian_transpose_mat,
    ]

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
    assert error < 1e-10


def test_rectangular_matrix_on_vectors(comm, rectangular_random_matrix):
    r"""Test MatrixLinearOperator with rectangular matrix on vectors"""
    Apetsc, Apython = rectangular_random_matrix
    Nr, Nc = Apython.shape
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)

    # apply: A * x (x in R^Nc, result in R^Nr)
    x, xpython = pytest_utils.generate_random_vector(comm, Nc)
    y = linop.create_left_vector()
    error_apply = pytest_utils.compute_error_vector(
        comm, linop.apply, x, y, Apython.dot, xpython
    )
    x.destroy()
    y.destroy()

    # apply_hermitian_transpose: A^H * x (x in R^Nr, result in R^Nc)
    x, xpython = pytest_utils.generate_random_vector(comm, Nr)
    y = linop.create_right_vector()
    error_aht = pytest_utils.compute_error_vector(
        comm, linop.apply_hermitian_transpose, x, y,
        Apython.conj().T.dot, xpython,
    )
    x.destroy()
    y.destroy()

    error = np.linalg.norm([error_apply, error_aht])
    linop.destroy()
    assert error < 1e-10


def test_rectangular_matrix_on_bvs(comm, rectangular_random_matrix):
    r"""Test MatrixLinearOperator with rectangular matrix on BVs"""
    Apetsc, Apython = rectangular_random_matrix
    Nr, Nc = Apython.shape
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    s = 5

    # apply_mat: A * X
    X, Xpython = pytest_utils.generate_random_bv(comm, (Nc, s))
    Y = linop.create_left_bv(s)
    error_apply = pytest_utils.compute_error_bv(
        comm, linop.apply_mat, X, Y, Apython.dot, Xpython
    )
    X.destroy()
    Y.destroy()

    # apply_hermitian_transpose_mat: A^H * X
    X, Xpython = pytest_utils.generate_random_bv(comm, (Nr, s))
    Y = linop.create_right_bv(s)
    error_aht = pytest_utils.compute_error_bv(
        comm, linop.apply_hermitian_transpose_mat, X, Y,
        Apython.conj().T.dot, Xpython,
    )
    X.destroy()
    Y.destroy()

    error = np.linalg.norm([error_apply, error_aht])
    linop.destroy()
    assert error < 1e-10


def test_matrix_vectors_y_none(comm, square_random_matrix):
    r"""Test MatrixLinearOperator with Y=None (output auto-allocated)"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)

    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])
    Apython_inv = sp.linalg.inv(Apython)

    # Test all actions with y=None
    actions = [
        (linop.apply, Apython.dot),
        (linop.apply_hermitian_transpose, Apython.conj().T.dot),
        (linop.solve, Apython_inv.dot),
        (linop.solve_hermitian_transpose, Apython_inv.conj().T.dot),
    ]
    error_vec = []
    for petsc_action, python_action in actions:
        y = petsc_action(x)
        ys = res4py.distributed_to_sequential_vector(y)
        ysa = ys.getArray().copy()
        ys.destroy()
        ypython = python_action(xpython)
        error_vec.append(
            np.linalg.norm(ypython - ysa) / np.linalg.norm(ypython)
        )
        y.destroy()

    error = np.linalg.norm(error_vec)
    x.destroy()
    linop.destroy()
    assert error < 1e-10


def test_matrix_bvs_y_none(comm, square_random_matrix):
    r"""Test MatrixLinearOperator BV actions with Y=None"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)

    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 5))
    Apython_inv = sp.linalg.inv(Apython)

    actions = [
        (linop.apply_mat, Apython.dot),
        (linop.apply_hermitian_transpose_mat, Apython.conj().T.dot),
        (linop.solve_mat, Apython_inv.dot),
        (linop.solve_hermitian_transpose_mat, Apython_inv.conj().T.dot),
    ]
    error_vec = []
    for petsc_action, python_action in actions:
        Y = petsc_action(X)
        Ym = Y.getMat()
        Yms = res4py.distributed_to_sequential_matrix(Ym)
        Y.restoreMat(Ym)
        Ymsa = Yms.getDenseArray().copy()
        Yms.destroy()
        Ypython = python_action(Xpython)
        error_vec.append(
            np.linalg.norm(Ypython - Ymsa) / np.linalg.norm(Ypython)
        )
        Y.destroy()

    error = np.linalg.norm(error_vec)
    X.destroy()
    linop.destroy()
    assert error < 1e-10


def test_matrix_repeated_apply(comm, square_random_matrix):
    r"""Test that repeated apply calls produce consistent results"""
    Apetsc, Apython = square_random_matrix
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)

    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])
    y1 = linop.apply(x)
    y2 = linop.apply(x)

    y1s = res4py.distributed_to_sequential_vector(y1)
    y2s = res4py.distributed_to_sequential_vector(y2)
    error = np.linalg.norm(y1s.getArray() - y2s.getArray())
    y1s.destroy()
    y2s.destroy()
    x.destroy()
    y1.destroy()
    y2.destroy()
    linop.destroy()
    assert error < 1e-14
