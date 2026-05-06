__all__ = [
    "generate_random_petsc_sparse_matrix",
    "generate_random_petsc_vector",
]

import typing

import numpy as np
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc

from .comms import scatter_array_from_root_to_all
from .matrix import convert_coo_to_csr


def generate_random_petsc_sparse_matrix(
    sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
    nnz: int,
    complex: typing.Optional[bool] = False,
) -> PETSc.Mat:
    r"""
    :param sizes: :code:`((local rows, global rows), (local cols, global cols))`
    :type sizes: Tuple[Tuple[int, int], Tuple[int, int]]
    :param nnz: number of non-zero entries
    :type nnz: int
    :param complex: :code:`True` if you want a complex-valued matrix
    :type complex: Optional[bool], default is False

    :return: a sparse PETSc matrix
    :rtype: PETSc.Mat
    """
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    nrows, ncols = sizes[0][-1], sizes[-1][-1]
    # Generate random matrix on root
    arrays = [None, None, None]
    if rank == 0:
        dtype = np.complex128 if complex else np.float64
        A = sp.sparse.random(
            nrows,
            ncols,
            density=nnz / (nrows * ncols),
            format="csr",
            dtype=dtype,
        )
        if nrows == ncols:  # add identity to make A invertible
            A += sp.sparse.identity(nrows, dtype=dtype, format="csr")
        A = A.tocoo()
        arrays = [A.row, A.col, A.data + 1j * 0]  # need complex type for PETSc
    # Scatter to all other processors and assemble
    recv_bufs = [scatter_array_from_root_to_all(a) for a in arrays]
    row_ptrs, cols, data = convert_coo_to_csr(recv_bufs, sizes)
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(sizes)
    A.setUp()
    A.setPreallocationCSR((row_ptrs, cols, data))
    A.setValuesCSR(row_ptrs, cols, data)
    A.assemble(None)
    return A


def generate_random_petsc_vector(
    sizes: typing.Tuple[int, int],
    complex: typing.Optional[bool] = False,
) -> PETSc.Vec:
    r"""
    :param sizes: vector sizes
    :type sizes: Tuple[int, int]
    :param complex: :code:`True` if you want a complex-valued vector
    :type complex: Optional[bool], default is False

    :rtype: PETSc.Vec
    """
    comm = PETSc.COMM_WORLD
    array = None
    if comm.getRank() == 0:
        vec = np.random.randn(sizes[-1])
        array = vec + 1j * np.random.randn(sizes[-1]) if complex else vec
    array = scatter_array_from_root_to_all(array, sizes[0])
    vec = PETSc.Vec().createWithArray(array, sizes, None, comm=comm)
    return vec
