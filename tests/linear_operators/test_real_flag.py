import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


def test_real_flag_complex_matrix(comm, square_matrix_size):
    r"""Test that a complex-valued matrix has real_flag = False"""
    Apetsc, Apython = pytest_utils.generate_random_matrix(
        comm, square_matrix_size, complex=True
    )
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    assert linop.get_real_flag() == False
    linop.destroy()


def test_real_flag_real_matrix(comm, square_matrix_size):
    r"""Test that a real-valued matrix has real_flag = True"""
    Apetsc, Apython = pytest_utils.generate_random_matrix(
        comm, square_matrix_size, complex=False
    )
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    assert linop.get_real_flag() == True
    linop.destroy()


def test_real_flag_real_low_rank(comm, square_matrix_size):
    r"""Test real_flag for a real-valued LowRankLinearOperator"""
    N = square_matrix_size[0]
    r = 5
    U, _ = pytest_utils.generate_random_bv(comm, (N, r), complex=False)
    V, _ = pytest_utils.generate_random_bv(comm, (N, r), complex=False)
    S = np.random.randn(r, r)
    S = comm.tompi4py().bcast(S, root=0)
    linop = res4py.linear_operators.LowRankLinearOperator(U, S, V, None)
    assert linop.get_real_flag() == True
    linop.destroy()


def test_real_flag_complex_low_rank(comm, square_matrix_size):
    r"""Test real_flag for a complex-valued LowRankLinearOperator"""
    N = square_matrix_size[0]
    r = 5
    U, _ = pytest_utils.generate_random_bv(comm, (N, r), complex=True)
    V, _ = pytest_utils.generate_random_bv(comm, (N, r), complex=True)
    S = np.random.randn(r, r) + 1j * np.random.randn(r, r)
    S = comm.tompi4py().bcast(S, root=0)
    linop = res4py.linear_operators.LowRankLinearOperator(U, S, V, None)
    assert linop.get_real_flag() == False
    linop.destroy()


def test_real_flag_shift_and_scale(comm, square_matrix_size):
    r"""Test real_flag for ShiftAndScaleLinearOperator with real components"""
    Apetsc, _ = pytest_utils.generate_random_matrix(
        comm, square_matrix_size, complex=False
    )
    linop_A = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    # Real alpha and beta with real A should give real operator
    linop = res4py.linear_operators.ShiftAndScaleLinearOperator(
        linop_A, alpha=2.0, beta=-1.0
    )
    assert linop.get_real_flag() == True
    linop.destroy()

    # Complex alpha with real A should give complex operator
    linop_A2 = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop2 = res4py.linear_operators.ShiftAndScaleLinearOperator(
        linop_A2, alpha=1.0 + 1.0j, beta=1.0
    )
    assert linop2.get_real_flag() == False
    linop2.destroy()
