import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from resolvent4py.utils.miscellaneous import get_mpi_type


def test_get_mpi_type_int32():
    r"""Test get_mpi_type for int32"""
    assert get_mpi_type(np.dtype(np.int32)) == MPI.INT


def test_get_mpi_type_int64():
    r"""Test get_mpi_type for int64"""
    assert get_mpi_type(np.dtype(np.int64)) == MPI.INT64_T


def test_get_mpi_type_float64():
    r"""Test get_mpi_type for float64"""
    assert get_mpi_type(np.dtype(np.float64)) == MPI.DOUBLE


def test_get_mpi_type_complex128():
    r"""Test get_mpi_type for complex128"""
    assert get_mpi_type(np.dtype(np.complex128)) == MPI.DOUBLE_COMPLEX


def test_get_mpi_type_petsc_int():
    r"""Test get_mpi_type for PETSc.IntType"""
    mpi_type = get_mpi_type(np.dtype(PETSc.IntType))
    assert mpi_type is not None


def test_get_mpi_type_petsc_scalar():
    r"""Test get_mpi_type for PETSc.ScalarType"""
    mpi_type = get_mpi_type(np.dtype(PETSc.ScalarType))
    assert mpi_type is not None


def test_get_mpi_type_invalid_raises():
    r"""Test get_mpi_type raises ValueError for unsupported dtype"""
    try:
        get_mpi_type(np.dtype(np.float16))
        raised = False
    except ValueError:
        raised = True
    assert raised
