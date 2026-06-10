import numpy as np
import scipy as sp
import resolvent4py as res4py
from .. import pytest_utils


def _create_low_rank_updated_operator(comm, Apetsc, Apython):
    r"""Helper to build a LowRankUpdatedLinearOperator and its numpy equivalent"""
    N = Apython.shape[0]
    rr, rc = 5, 9
    U, Upython = pytest_utils.generate_random_bv(comm, (N, rr))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, rc))
    S = np.random.randn(rr, rc) + 1j * np.random.randn(rr, rc)
    S = comm.tompi4py().bcast(S, root=0)
    Lpython = Apython + Upython @ S @ Vpython.conj().T
    ksp = res4py.create_mumps_solver(Apetsc)
    linop1 = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)
    linop = res4py.linear_operators.LowRankUpdatedLinearOperator(
        linop1, U, S, V
    )
    # Objects created here that the operators no longer destroy on their own
    # (Apetsc is owned by the fixture).
    owned = [linop, linop1, ksp, U, V]
    return linop, Lpython, owned


def test_low_rank_updated_on_vectors(comm, square_random_matrix):
    r"""Test LowRankUpdatedLinearOperator on vectors"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    linop, Lpython, owned = _create_low_rank_updated_operator(
        comm, Apetsc, Apython
    )

    x, xpython = pytest_utils.generate_random_vector(comm, N)
    Lpython_inv = sp.linalg.inv(Lpython)
    actions_python = [
        Lpython.dot,
        Lpython.conj().T.dot,
        Lpython_inv.dot,
        Lpython_inv.conj().T.dot,
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
    for o in owned:
        o.destroy()
    assert error < 1e-8


def test_low_rank_updated_on_bvs(comm, square_random_matrix):
    r"""Test LowRankUpdatedLinearOperator on BVs"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    linop, Lpython, owned = _create_low_rank_updated_operator(
        comm, Apetsc, Apython
    )

    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Lpython_inv = sp.linalg.inv(Lpython)
    actions_python = [
        Lpython.dot,
        Lpython.conj().T.dot,
        Lpython_inv.dot,
        Lpython_inv.conj().T.dot,
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
    for o in owned:
        o.destroy()
    assert error < 1e-8


def test_low_rank_updated_repeated_apply_mat(comm, square_random_matrix):
    r"""Test that repeated apply_mat calls produce consistent results
    (validates the cached intermediate BV)"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    linop, Lpython, owned = _create_low_rank_updated_operator(
        comm, Apetsc, Apython
    )

    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))

    Y1 = linop.apply_mat(X)
    Y2 = linop.apply_mat(X)

    Y1m = Y1.getMat()
    Y2m = Y2.getMat()
    Y1ms = res4py.distributed_to_sequential_matrix(Y1m)
    Y2ms = res4py.distributed_to_sequential_matrix(Y2m)
    Y1.restoreMat(Y1m)
    Y2.restoreMat(Y2m)
    Y1a = Y1ms.getDenseArray().copy()
    Y2a = Y2ms.getDenseArray().copy()
    error = np.linalg.norm(Y1a - Y2a) / np.linalg.norm(Y1a)
    Y1ms.destroy()
    Y2ms.destroy()

    # Also verify against python
    Ypython = Lpython.dot(Xpython)
    error_python = np.linalg.norm(Ypython - Y1a) / np.linalg.norm(Ypython)

    X.destroy()
    Y1.destroy()
    Y2.destroy()
    for o in owned:
        o.destroy()
    assert error < 1e-14
    assert error_python < 1e-8


def test_low_rank_updated_varying_column_counts(comm, square_random_matrix):
    r"""Test apply_mat with different column counts to exercise BV cache
    resize logic"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    linop, Lpython, owned = _create_low_rank_updated_operator(
        comm, Apetsc, Apython
    )

    error_vec = []
    for s in [3, 7, 3]:
        X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
        Y = linop.apply_mat(X)
        Ym = Y.getMat()
        Yms = res4py.distributed_to_sequential_matrix(Ym)
        Y.restoreMat(Ym)
        Ymsa = Yms.getDenseArray().copy()
        Yms.destroy()
        Ypython = Lpython.dot(Xpython)
        error_vec.append(
            np.linalg.norm(Ypython - Ymsa) / np.linalg.norm(Ypython)
        )
        X.destroy()
        Y.destroy()

    error = np.linalg.norm(error_vec)
    for o in owned:
        o.destroy()
    assert error < 1e-8
