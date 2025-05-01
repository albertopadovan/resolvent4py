__all__ = [
    "create_dense_matrix",
    "create_AIJ_identity",
    "mat_solve_hermitian_transpose",
    "hermitian_transpose",
    "convert_coo_to_csr",
]

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from .miscellaneous import get_mpi_type


def create_dense_matrix(comm, sizes):
    r"""
    Create dense matrix

    :param comm: MPI communicator
    :param sizes: `MatSizeSpec`_

    :rtype: PETSc.Mat.Type.DENSE
    """
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.setUp()
    return M


def create_AIJ_identity(comm, sizes):
    r"""
    Create identity matrix of sparse AIJ type

    :param comm: MPI Communicator
    :param sizes: `MatSizeSpec`_

    :return: identity matrix
    :rtype: PETSc.Mat.Type.AIJ
    """
    Id = PETSc.Mat().createConstantDiagonal(sizes, 1.0, comm)
    Id.convert(PETSc.Mat.Type.AIJ)
    return Id


def mat_solve_hermitian_transpose(ksp, X, Y=None):
    r"""
    Solve :math:`A^{-1}X = Z`, where :math:`X` is a PETSc matrix of type
    :code:`PETSc.Mat.Type.DENSE`

    :param ksp: a KPS solver structure
    :type ksp: PETSc.KSP
    :param X: a dense PETSc matrix
    :type X: PETSc.Mat.Type.DENSE

    :return: a matrix of the same type and size as :math:`X`
    :rtype: PETSc.Mat.Type.DENSE
    """
    sizes = X.getSizes()
    Yarray = np.zeros((sizes[0][0], sizes[-1][-1]), dtype=np.complex128)
    Y = X.duplicate() if Y == None else Y
    y = X.createVecLeft()
    for i in range(X.getSizes()[-1][-1]):
        x = X.getColumnVector(i)
        x.conjugate()
        ksp.solveTranspose(x, y)
        x.conjugate()
        y.conjugate()
        Yarray[:, i] = y.getArray()
        x.destroy()
    y.destroy()
    offset, _ = Y.getOwnershipRange()
    rows = np.arange(Yarray.shape[0], dtype=np.int32) + offset
    cols = np.arange(Yarray.shape[-1], dtype=np.int32)
    Y.setValues(rows, cols, Yarray.reshape(-1))
    Y.assemble(None)
    return Y


def hermitian_transpose(comm, Mat, in_place=False, MatHT=None):
    r"""
    Return the hermitian transpose of the matrix :code:`Mat`.

    :param comm: MPI communicator
    :param Mat: PETSc matrix of any kind
    :param in_place: [optional] in place transposition if :code:`True` and
        out of place otherwise
    :type in_place: bool
    :param MatHT: [optional] matrix with the correct layout to hold the
        hermitian transpose of :code:`Mat`
    :param Mat: PETSc matrix of any kind
    """
    if in_place == False:
        if MatHT == None:
            sizes = Mat.getSizes()
            MatHT = PETSc.Mat().create(comm)
            MatHT.setType(Mat.getType())
            MatHT.setSizes((sizes[-1], sizes[0]))
            MatHT.setUp()
        Mat.setTransposePrecursor(MatHT)
        Mat.hermitianTranspose(MatHT)
        return MatHT
    else:
        MatHT_ = Mat.hermitianTranspose()
        return MatHT_


def convert_coo_to_csr(comm, arrays, sizes):
    r"""
    Convert arrays = [row indices, col indices, values] for COO matrix
    assembly to [row pointers, col indices, values] for CSR matrix assembly.
    (petsc4py currently does not support coo matrix assembly, hence the need
    to convert.)

    :param comm: MPI communicator (only MPI.COMM_WORLD is supported for now)
    :param arrays: a list of numpy arrays (e.g., arrays = [rows,cols,vals])
    :param sizes: matrix size
    :type sizes: `MatSizeSpec <MatSizeSpec_>`_

    :return: csr row pointers, column indices and matrix values for CSR
        matrix assembly
    :rtype: list
    """

    rank = comm.Get_rank()
    pool = np.arange(comm.Get_size())
    rows, cols, vals = arrays
    idces = np.argsort(rows).reshape(-1)
    rows, cols, vals = rows[idces], cols[idces], vals[idces]

    mat_row_sizes_local = np.asarray(
        comm.allgather(sizes[0][0]), dtype=np.int32
    )
    mat_row_displ = np.concatenate(([0], np.cumsum(mat_row_sizes_local[:-1])))
    ownership_ranges = np.zeros((comm.Get_size(), 2), dtype=np.int32)
    ownership_ranges[:, 0] = mat_row_displ
    ownership_ranges[:-1, 1] = ownership_ranges[1:, 0]
    ownership_ranges[-1, 1] = sizes[0][-1]

    send_rows, send_cols = [], []
    send_vals, lengths = [], []
    for i in pool:
        idces = np.argwhere(
            (rows >= ownership_ranges[i, 0]) & (rows < ownership_ranges[i, 1])
        ).reshape(-1)
        lengths.append(np.asarray([len(idces)], dtype=np.int32))
        send_rows.append(rows[idces])
        send_cols.append(cols[idces])
        send_vals.append(vals[idces])

    recv_bufs = [np.empty(1, dtype=np.int32) for _ in pool]
    recv_reqs = [comm.Irecv(bf, source=i) for (bf, i) in zip(recv_bufs, pool)]
    send_reqs = [comm.Isend(sz, dest=i) for (i, sz) in enumerate(lengths)]
    MPI.Request.waitall(send_reqs + recv_reqs)
    lengths = [buf[0] for buf in recv_bufs]

    dtypes = [np.int32, np.int32, np.complex128]
    my_arrays = []
    for j, array in enumerate([send_rows, send_cols, send_vals]):
        dtype = dtypes[j]
        mpi_type = get_mpi_type(np.dtype(dtype))
        recv_bufs = [
            [np.empty(lengths[i], dtype=dtype), mpi_type] for i in pool
        ]
        recv_reqs = [
            comm.Irecv(bf, source=i) for (bf, i) in zip(recv_bufs, pool)
        ]
        send_reqs = [comm.Isend(array[i], dest=i) for i in pool]
        MPI.Request.waitall(send_reqs + recv_reqs)
        my_arrays.append([recv_bufs[i][0] for i in pool])

    my_rows, my_cols, my_vals = [], [], []
    for i in pool:
        my_rows.extend(my_arrays[0][i])
        my_cols.extend(my_arrays[1][i])
        my_vals.extend(my_arrays[2][i])

    my_rows = np.asarray(my_rows, dtype=np.int32) - ownership_ranges[rank, 0]
    my_cols = np.asarray(my_cols, dtype=np.int32)
    my_vals = np.asarray(my_vals, dtype=np.complex128)

    idces = np.argsort(my_rows).reshape(-1)
    my_rows = my_rows[idces]
    my_cols = my_cols[idces]
    my_vals = my_vals[idces]

    ni = 0
    my_rows_ptr = np.zeros(sizes[0][0] + 1, dtype=np.int32)
    for i in range(sizes[0][0]):
        ni += np.count_nonzero(my_rows == i)
        my_rows_ptr[i + 1] = ni

    return my_rows_ptr, my_cols, my_vals
