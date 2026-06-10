import scipy as sp
import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


def test_product_on_vectors(comm, square_matrix_size):
    r"""Test ProductLinearOperator on vectors"""

    N, _ = square_matrix_size
    Apetsc1, Apython1 = pytest_utils.generate_random_matrix(comm, (N, N))
    Apetsc2, Apython2 = pytest_utils.generate_random_matrix(comm, (N, N // 2))
    Apetsc3, Apython3 = pytest_utils.generate_random_matrix(
        comm, (N // 4, N // 2)
    )
    Apetsc4, Apython4 = pytest_utils.generate_random_matrix(
        comm, (N // 4, N // 4)
    )

    r = 3
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    S = comm.tompi4py().bcast(S, root=0)
    Apython1 += Upython @ S @ Vpython.conj().T

    ksp = res4py.create_mumps_solver(Apetsc1)
    linop_ = res4py.linear_operators.MatrixLinearOperator(Apetsc1, ksp)
    linop1 = res4py.linear_operators.LowRankUpdatedLinearOperator(
        linop_, U, S, V
    )
    linop2 = res4py.linear_operators.MatrixLinearOperator(Apetsc2)
    linop3 = res4py.linear_operators.MatrixLinearOperator(Apetsc3)
    linop4 = res4py.linear_operators.MatrixLinearOperator(Apetsc4)

    A = sp.linalg.inv(Apython1) @ Apython2 @ Apython3.conj().T @ Apython4
    linops = [linop1, linop2, linop3, linop4]
    actions = [
        linop1.solve,
        linop2.apply,
        linop3.apply_hermitian_transpose,
        linop4.apply,
    ]
    linop = res4py.linear_operators.ProductLinearOperator(linops, actions)

    x, xpython = pytest_utils.generate_random_vector(comm, A.shape[-1])
    y = linop.create_left_vector()
    error_vec = []
    error_vec.append(
        pytest_utils.compute_error_vector(
            comm, linop.apply, x, y, A.dot, xpython
        )
    )
    x.destroy()
    y.destroy()
    x, xpython = pytest_utils.generate_random_vector(comm, A.shape[0])
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
    linop1.destroy()
    linop_.destroy()
    linop2.destroy()
    linop3.destroy()
    linop4.destroy()
    ksp.destroy()
    U.destroy()
    V.destroy()
    Apetsc1.destroy()
    Apetsc2.destroy()
    Apetsc3.destroy()
    Apetsc4.destroy()
    assert error < 1e-10


def test_product_on_bvs(comm, square_matrix_size):
    r"""Test ProductLinearOperator on BVs"""

    N, _ = square_matrix_size
    Apetsc1, Apython1 = pytest_utils.generate_random_matrix(comm, (N, N))
    Apetsc2, Apython2 = pytest_utils.generate_random_matrix(comm, (N, N // 2))
    Apetsc3, Apython3 = pytest_utils.generate_random_matrix(
        comm, (N // 4, N // 2)
    )
    Apetsc4, Apython4 = pytest_utils.generate_random_matrix(
        comm, (N // 4, N // 4)
    )

    r = 3
    U, Upython = pytest_utils.generate_random_bv(comm, (N, r))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, r))
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    S = comm.tompi4py().bcast(S, root=0)
    Apython1 += Upython @ S @ Vpython.conj().T

    ksp = res4py.create_mumps_solver(Apetsc1)
    linop_ = res4py.linear_operators.MatrixLinearOperator(Apetsc1, ksp)
    linop1 = res4py.linear_operators.LowRankUpdatedLinearOperator(
        linop_, U, S, V
    )
    linop2 = res4py.linear_operators.MatrixLinearOperator(Apetsc2)
    linop3 = res4py.linear_operators.MatrixLinearOperator(Apetsc3)
    linop4 = res4py.linear_operators.MatrixLinearOperator(Apetsc4)

    A = sp.linalg.inv(Apython1) @ Apython2 @ Apython3.conj().T @ Apython4
    linops = [linop1, linop2, linop3, linop4]
    actions = [
        linop1.solve,
        linop2.apply,
        linop3.apply_hermitian_transpose,
        linop4.apply,
    ]
    linop = res4py.linear_operators.ProductLinearOperator(linops, actions)

    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (A.shape[-1], s))
    Y = linop.create_left_bv(s)
    error_vec = []
    error_vec.append(
        pytest_utils.compute_error_bv(
            comm, linop.apply_mat, X, Y, A.dot, Xpython
        )
    )
    X.destroy()
    Y.destroy()
    X, Xpython = pytest_utils.generate_random_bv(comm, (A.shape[0], s))
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
    linop1.destroy()
    linop_.destroy()
    linop2.destroy()
    linop3.destroy()
    linop4.destroy()
    ksp.destroy()
    U.destroy()
    V.destroy()
    Apetsc1.destroy()
    Apetsc2.destroy()
    Apetsc3.destroy()
    Apetsc4.destroy()
    assert error < 1e-10
