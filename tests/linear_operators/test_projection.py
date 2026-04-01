import numpy as np
import scipy as sp
import resolvent4py as res4py
from .. import pytest_utils


def test_projection_on_vectors(comm, square_matrix_size):
    r"""Test ProjectionLinearOperator on vectors"""

    N = square_matrix_size[0]
    r = 5
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))
    S = sp.linalg.inv(Vpython.conj().T @ Upython)

    complements = [False, True]
    P = Upython @ S @ Vpython.conj().T
    Apython = [P, np.eye(N) - P]
    for k, compl in enumerate(complements):
        linop = res4py.linear_operators.ProjectionLinearOperator(U, V, compl)
        x, xpython = pytest_utils.generate_random_vector(comm, N)
        actions_python = [Apython[k].dot, Apython[k].conj().T.dot]
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
    assert error < 1e-8


def test_projection_on_bvs(comm, square_matrix_size):
    r"""Test ProjectionLinearOperator on BVs"""

    N = square_matrix_size[0]
    r = 5
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))
    S = sp.linalg.inv(Vpython.conj().T @ Upython)

    complements = [False, True]
    P = Upython @ S @ Vpython.conj().T
    Apython = [P, np.eye(N) - P]
    for k, compl in enumerate(complements):
        linop = res4py.linear_operators.ProjectionLinearOperator(U, V, compl)
        X, Xpython = pytest_utils.generate_random_bv(comm, (N, 7))
        actions_python = [Apython[k].dot, Apython[k].conj().T.dot]
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
    assert error < 1e-8


def test_projection_idempotency_vectors(comm, square_matrix_size):
    r"""Test that P^2 = P for ProjectionLinearOperator on vectors"""

    N = square_matrix_size[0]
    r = 5
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))

    for compl in [False, True]:
        linop = res4py.linear_operators.ProjectionLinearOperator(
            U, V, compl
        )
        x, _ = pytest_utils.generate_random_vector(comm, N)
        Px = linop.apply(x)
        PPx = linop.apply(Px)
        Pxs = res4py.distributed_to_sequential_vector(Px)
        PPxs = res4py.distributed_to_sequential_vector(PPx)
        error = np.linalg.norm(Pxs.getArray() - PPxs.getArray())
        norm = np.linalg.norm(Pxs.getArray())
        Pxs.destroy()
        PPxs.destroy()
        x.destroy()
        Px.destroy()
        PPx.destroy()
        assert error / norm < 1e-10

    linop.destroy()


def test_projection_idempotency_bvs(comm, square_matrix_size):
    r"""Test that P^2 = P for ProjectionLinearOperator on BVs"""

    N = square_matrix_size[0]
    r = 5
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))

    for compl in [False, True]:
        linop = res4py.linear_operators.ProjectionLinearOperator(
            U, V, compl
        )
        X, _ = pytest_utils.generate_random_bv(comm, (N, 7))
        PX = linop.apply_mat(X)
        PPX = linop.apply_mat(PX)

        PXm = PX.getMat()
        PPXm = PPX.getMat()
        PXms = res4py.distributed_to_sequential_matrix(PXm)
        PPXms = res4py.distributed_to_sequential_matrix(PPXm)
        PX.restoreMat(PXm)
        PPX.restoreMat(PPXm)

        PXa = PXms.getDenseArray().copy()
        PPXa = PPXms.getDenseArray().copy()
        error = np.linalg.norm(PXa - PPXa) / np.linalg.norm(PXa)

        PXms.destroy()
        PPXms.destroy()
        X.destroy()
        PX.destroy()
        PPX.destroy()
        assert error < 1e-10

    linop.destroy()
