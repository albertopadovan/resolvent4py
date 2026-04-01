import numpy as np
import resolvent4py as res4py
from .. import pytest_utils


def test_bv_add(comm, square_matrix_size):
    r"""Test bv_add: X <- X + alpha * Y"""
    N = square_matrix_size[0]
    s = 5
    alpha = 2.5 + 1.0j
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y, Ypython = pytest_utils.generate_random_bv(comm, (N, s))
    expected = Xpython + alpha * Ypython

    res4py.bv_add(alpha, X, Y)

    Xm = X.getMat()
    Xms = res4py.distributed_to_sequential_matrix(Xm)
    X.restoreMat(Xm)
    error = np.linalg.norm(Xms.getDenseArray() - expected) / np.linalg.norm(
        expected
    )
    Xms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_conj_copy(comm, square_matrix_size):
    r"""Test bv_conj with inplace=False"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y = res4py.bv_conj(X, inplace=False)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython.conj()
    ) / np.linalg.norm(Xpython)
    Yms.destroy()

    # X should be unchanged
    Xm = X.getMat()
    Xms = res4py.distributed_to_sequential_matrix(Xm)
    X.restoreMat(Xm)
    error_x = np.linalg.norm(Xms.getDenseArray() - Xpython) / np.linalg.norm(
        Xpython
    )
    Xms.destroy()

    X.destroy()
    Y.destroy()
    assert error < 1e-14
    assert error_x < 1e-14


def test_bv_conj_inplace(comm, square_matrix_size):
    r"""Test bv_conj with inplace=True"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y = res4py.bv_conj(X, inplace=True)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython.conj()
    ) / np.linalg.norm(Xpython)
    Yms.destroy()
    X.destroy()
    assert error < 1e-14


def test_bv_real_copy(comm, square_matrix_size):
    r"""Test bv_real with inplace=False"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y = res4py.bv_real(X, inplace=False)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython.real
    ) / np.linalg.norm(Xpython)
    Yms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_imag_copy(comm, square_matrix_size):
    r"""Test bv_imag with inplace=False"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y = res4py.bv_imag(X, inplace=False)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython.imag
    ) / np.linalg.norm(Xpython)
    Yms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_slice(comm, square_matrix_size):
    r"""Test bv_slice extracts the correct columns"""
    N = square_matrix_size[0]
    s = 7
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    columns = np.array([0, 3, 5])
    Y = res4py.bv_slice(X, columns)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython[:, columns]
    ) / np.linalg.norm(Xpython[:, columns])
    Yms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_slice_with_preallocated_output(comm, square_matrix_size):
    r"""Test bv_slice with a pre-allocated Y"""
    N = square_matrix_size[0]
    s = 7
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    columns = np.array([1, 4, 6])

    from slepc4py import SLEPc

    Y = SLEPc.BV().create(comm=comm)
    Nl = res4py.compute_local_size(N)
    Y.setSizes((Nl, N), len(columns))
    Y.setType("mat")

    Y = res4py.bv_slice(X, columns, Y)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(
        Yms.getDenseArray() - Xpython[:, columns]
    ) / np.linalg.norm(Xpython[:, columns])
    Yms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_roll_columns(comm, square_matrix_size):
    r"""Test bv_roll with axis=-1 (column roll)"""
    from resolvent4py.utils.bv import bv_roll

    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    roll_amount = 2
    Y = bv_roll(X, roll_amount, axis=-1, in_place=False)

    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    expected = np.roll(Xpython, roll_amount, axis=-1)
    error = np.linalg.norm(
        Yms.getDenseArray() - expected
    ) / np.linalg.norm(expected)
    Yms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14


def test_bv_add_zero_alpha(comm, square_matrix_size):
    r"""Test bv_add with alpha=0 leaves X unchanged"""
    N = square_matrix_size[0]
    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Y, _ = pytest_utils.generate_random_bv(comm, (N, s))

    res4py.bv_add(0.0, X, Y)

    Xm = X.getMat()
    Xms = res4py.distributed_to_sequential_matrix(Xm)
    X.restoreMat(Xm)
    error = np.linalg.norm(Xms.getDenseArray() - Xpython) / np.linalg.norm(
        Xpython
    )
    Xms.destroy()
    X.destroy()
    Y.destroy()
    assert error < 1e-14
