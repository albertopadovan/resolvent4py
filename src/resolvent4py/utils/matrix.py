__all__ = [
    "create_dense_matrix",
    "create_AIJ_identity",
    "mat_solve_hermitian_transpose",
    "hermitian_transpose",
    "convert_coo_to_csr",
    "assemble_harmonic_resolvent_generator",
]

def show_type(np_type):
    mpi_t = get_mpi_type(np.dtype(np_type))
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] {np_type} "
        f"→ MPI {mpi_t.Get_name()} (size {mpi_t.Get_size()} bytes)")

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Optional

from .miscellaneous import get_mpi_type


def create_dense_matrix(
    comm: PETSc.Comm, sizes: tuple[tuple[int, int], tuple[int, int]]
) -> PETSc.Mat:
    r"""
    Create dense matrix

    :param comm: PETSc communicator
    :type comm: PETSc.Comm
    :param sizes: tuple[tuple[int, int], tuple[int, int]]

    :rtype: PETSc.Mat.Type.DENSE
    """
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.setUp()
    return M


def create_AIJ_identity(
    comm: PETSc.Comm, sizes: tuple[tuple[int, int], tuple[int, int]]
) -> PETSc.Mat:
    r"""
    Create identity matrix of sparse AIJ type

    :param comm: MPI Communicator
    :type comm: PETSc.Comm
    :param sizes: see `MatSizeSpec <MatSizeSpec_>`_
    :type sizes: tuple[tuple[int, int], tuple[int, int]]

    :return: identity matrix
    :rtype: PETSc.Mat.Type.AIJ
    """
    Id = PETSc.Mat().createConstantDiagonal(sizes, 1.0, comm)
    Id.convert(PETSc.Mat.Type.AIJ)
    return Id


def mat_solve_hermitian_transpose(
    ksp: PETSc.KSP, X: PETSc.Mat, Y: Optional[PETSc.Mat] = None
) -> PETSc.Mat:
    r"""
    Solve :math:`A^{-*}X = Y`, where :math:`X` is a PETSc matrix of type
    :code:`PETSc.Mat.Type.DENSE`

    :param ksp: a KPS solver structure
    :type ksp: PETSc.KSP
    :param X: a dense PETSc matrix
    :type X: PETSc.Mat.Type.DENSE
    :param Y: a dense PETSc matrix
    :type Y: Optional[PETSc.Mat.Type.DENSE] defaults to :code:`None`

    :return: matrix to store the result
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
    rows = np.arange(Yarray.shape[0], dtype=PETSc.IntType) + offset
    cols = np.arange(Yarray.shape[-1], dtype=PETSc.IntType)
    Y.setValues(rows, cols, Yarray.reshape(-1))
    Y.assemble(None)
    return Y


def hermitian_transpose(
    Mat: PETSc.Mat, in_place=False, MatHT=None
) -> PETSc.Mat:
    r"""
    Return the hermitian transpose of the matrix :code:`Mat`.

    :param Mat: PETSc matrix
    :type Mat: PETSc.Mat
    :param in_place: in-place transposition if :code:`True` and
        out of place otherwise
    :type in_place: Optional[bool] defaults to :code:`False`
    :param MatHT: [optional] matrix with the correct layout to hold the
        hermitian transpose of :code:`Mat`
    :param MatHT: Optional[PETSc.Mat] defaults to :code:`None`
    """
    if in_place == False:
        if MatHT == None:
            sizes = Mat.getSizes()
            MatHT = PETSc.Mat().create(comm=Mat.getComm())
            MatHT.setType(Mat.getType())
            MatHT.setSizes((sizes[-1], sizes[0]))
            MatHT.setUp()
        Mat.setTransposePrecursor(MatHT)
        Mat.hermitianTranspose(MatHT)
        return MatHT
    else:
        MatHT_ = Mat.hermitianTranspose()
        return MatHT_


def convert_coo_to_csr(
    arrays: tuple[np.array, np.array, np.array],
    sizes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.array, np.array, np.array]:
    r"""
    Convert arrays = [row indices, col indices, values] for COO matrix
    assembly to [row pointers, col indices, values] for CSR matrix assembly.
    (Petsc4py currently does not support COO matrix assembly, hence the need
    to convert.)

    :param arrays: a list of numpy arrays (e.g., arrays = [rows,cols,vals])
    :type array: tuple[np.array, np.array, np.array]
    :param sizes: see `MatSizeSpec <MatSizeSpec_>`_
    :type sizes: tuple[np.array, np.array, np.array]

    :return: csr row pointers, column indices and matrix values for CSR
        matrix assembly
    :rtype: tuple[np.array, np.array, np.array]
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool = np.arange(comm.Get_size())
    rows, cols, vals = arrays
    idces = np.argsort(rows).reshape(-1)
    # Make sure the arrays have the correct data types (save for MPI comms)
    rows = np.asarray(rows[idces], dtype=PETSc.IntType)
    cols = np.asarray(cols[idces], dtype=PETSc.IntType)
    vals = np.asarray(vals[idces], dtype=PETSc.ScalarType)

    mat_row_sizes_local = np.asarray(
        comm.allgather(sizes[0][0]), dtype=PETSc.IntType
    )
    mat_row_displ = np.concatenate(([0], np.cumsum(mat_row_sizes_local[:-1])))
    ownership_ranges = np.zeros((comm.Get_size(), 2), dtype=PETSc.IntType)
    ownership_ranges[:, 0] = mat_row_displ
    ownership_ranges[:-1, 1] = ownership_ranges[1:, 0]
    ownership_ranges[-1, 1] = sizes[0][-1]

    send_rows, send_cols = [], []
    send_vals, lengths = [], []
    for i in pool:
        idces = np.argwhere(
            (rows >= ownership_ranges[i, 0]) & (rows < ownership_ranges[i, 1])
        ).reshape(-1)
        lengths.append(np.asarray([len(idces)], dtype=PETSc.IntType))
        send_rows.append(rows[idces])
        send_cols.append(cols[idces])
        send_vals.append(vals[idces])

    recv_bufs = [np.empty(1, dtype=PETSc.IntType) for _ in pool]
    recv_reqs = [comm.Irecv(bf, source=i) for (bf, i) in zip(recv_bufs, pool)]
    send_reqs = [comm.Isend(sz, dest=i) for (i, sz) in enumerate(lengths)]
    MPI.Request.waitall(send_reqs + recv_reqs)
    lengths = [buf[0] for buf in recv_bufs]

    dtypes = [PETSc.IntType, PETSc.IntType, np.complex128]
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

    my_rows = (
        np.asarray(my_rows, dtype=PETSc.IntType) - ownership_ranges[rank, 0]
    )
    my_cols = np.asarray(my_cols, dtype=PETSc.IntType)
    my_vals = np.asarray(my_vals, dtype=np.complex128)

    idces = np.argsort(my_rows).reshape(-1)
    my_rows = my_rows[idces]
    my_cols = my_cols[idces]
    my_vals = my_vals[idces]

    ni = 0
    my_rows_ptr = np.zeros(sizes[0][0] + 1, dtype=PETSc.IntType)
    my_rows_ptr[1:] = np.cumsum(np.bincount(my_rows, minlength=sizes[0][0]))

    return my_rows_ptr, my_cols, my_vals


def assemble_harmonic_resolvent_generator(
    A: PETSc.Mat, freqs: np.array, M: Optional[PETSc.Mat]=None
) -> PETSc.Mat:
    r"""
    Assemble :math:`T = -\tilde{M} + A`, where :math:`A` is the output of
    :func:`resolvent4py.utils.io.read_harmonic_balanced_matrix`
    and :math:`M` is a block
    diagonal matrix with block :math:`k` given by :math:`\tilde{M}_k = i k \omega M_k`
    where :math:`k\omega` is the :math:`k`th entry of :code:`freqs` and 
    :math:`M_k` is the math:`k`th block of the matrix :math:`M` (if provided).
    Otherwise, :math:`M_k = Id`.

    :param A: assembled PETSc matrix
    :type A: PETSc.Mat
    :param freqs: array :math:`\omega\left(\ldots, -1, 0, 1, \ldots\right)`
    :type freqs: np.array
    :param M: assembled PETSc (mass) matrix
    :type M: Optional[PETSc.Mat], default is None

    :rtype: PETSc.Mat
    """
    rows_lst = []
    vals_lst = []

    rows = np.arange(*A.getOwnershipRange())
    N = A.getSizes()[0][-1] // len(freqs)
    for i in range(len(freqs)):
        idces = np.intersect1d(rows, np.arange(N * i, N * (i + 1)))
        if len(idces) > 0:
            rows_lst.extend(idces)
            vals_lst.extend(-1j * freqs[i] * np.ones(len(idces)))

    rows = np.asarray(rows_lst, dtype=PETSc.IntType)
    vals = np.asarray(vals_lst, dtype=np.complex128)

    rows_ptr, cols, vals = convert_coo_to_csr([rows, rows, vals], A.getSizes())
    omId = PETSc.Mat().createAIJ(A.getSizes(), comm=A.getComm())
    omId.setPreallocationCSR((rows_ptr, cols))
    omId.setValuesCSR(rows_ptr, cols, vals, True)
    omId.assemble(False)
    if M == None:
        omId.axpy(1.0, A)
        return omId
    else:
        Mat = omId.matMult(M)
        Mat.axpy(1.0, A)
        omId.destroy()
        return Mat

