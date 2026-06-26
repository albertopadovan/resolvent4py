__all__ = [
    "create_dense_matrix",
    "create_aij_matrix",
    "create_AIJ_identity",
    "add_identity_block",
    "add_matrix_block",
    "left_diagonal_solve",
    "matrix_diagonal_array",
    "matrix_subblock",
    "mat_solve_hermitian_transpose",
    "hermitian_transpose",
    "convert_coo_to_csr",
    "assemble_harmonic_resolvent_generator",
]


def show_type(np_type):
    mpi_t = get_mpi_type(np.dtype(np_type))
    print(
        f"[Rank {MPI.COMM_WORLD.Get_rank()}] {np_type} "
        f"→ MPI {mpi_t.Get_name()} (size {mpi_t.Get_size()} bytes)"
    )


import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Optional

from .miscellaneous import get_mpi_type, petscprint
from .vector import array_from_petsc_vector


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


def create_aij_matrix(
    comm: PETSc.Comm,
    nrows: int,
    ncols: int,
    nnz: int | None = None,
) -> PETSc.Mat:
    r"""
    Create an AIJ matrix.

    :param comm: PETSc communicator
    :type comm: PETSc.Comm
    :param nrows: Global row count
    :type nrows: int
    :param ncols: Global column count
    :type ncols: int
    :param nnz: Optional row nonzero preallocation
    :type nnz: int | None

    :return: Sparse AIJ matrix
    :rtype: PETSc.Mat
    """
    mat = PETSc.Mat().createAIJ(((PETSc.DECIDE, nrows), (PETSc.DECIDE, ncols)), nnz=nnz, comm=comm)
    mat.setUp()
    return mat


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


def matrix_diagonal_array(matrix: PETSc.Mat) -> np.ndarray:
    r"""
    Return the global matrix diagonal.

    :param matrix: PETSc matrix
    :type matrix: PETSc.Mat

    :return: Matrix diagonal
    :rtype: np.ndarray
    """
    diagonal = matrix.createVecLeft()
    try:
        matrix.getDiagonal(diagonal)
        return array_from_petsc_vector(diagonal)
    finally:
        diagonal.destroy()


def matrix_subblock(
    matrix: PETSc.Mat,
    rows: np.ndarray | list[int],
    columns: np.ndarray | list[int],
) -> PETSc.Mat:
    r"""
    Extract a PETSc submatrix.

    :param matrix: PETSc matrix
    :type matrix: PETSc.Mat
    :param rows: Global row indices
    :type rows: np.ndarray | list[int]
    :param columns: Global column indices
    :type columns: np.ndarray | list[int]

    :return: PETSc submatrix
    :rtype: PETSc.Mat
    """
    comm = matrix.getComm()
    rank = comm.getRank()
    size = comm.getSize()
    rows = np.asarray(rows, dtype=PETSc.IntType)
    columns = np.asarray(columns, dtype=PETSc.IntType)
    row_start, row_end = _ownership_slice(rows.size, rank, size)
    col_start, col_end = _ownership_slice(columns.size, rank, size)
    row_is = PETSc.IS().createGeneral(rows[row_start:row_end], comm=comm)
    col_is = PETSc.IS().createGeneral(columns[col_start:col_end], comm=comm)
    try:
        return matrix.createSubMatrix(row_is, col_is)
    finally:
        col_is.destroy()
        row_is.destroy()


def _ownership_slice(size: int, rank: int, n_ranks: int) -> tuple[int, int]:
    base = size // n_ranks
    extra = size % n_ranks
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def add_matrix_block(
    target: PETSc.Mat,
    row_offset: int,
    col_offset: int,
    scale: complex,
    block: PETSc.Mat,
) -> None:
    r"""
    Add a matrix block into another matrix.

    :param target: Target matrix
    :type target: PETSc.Mat
    :param row_offset: Target row offset
    :type row_offset: int
    :param col_offset: Target column offset
    :type col_offset: int
    :param scale: Block scaling
    :type scale: complex
    :param block: Source matrix block
    :type block: PETSc.Mat
    :rtype: None
    """
    start, end = block.getOwnershipRange()
    for row in range(start, end):
        columns, values = block.getRow(row)
        if len(columns) == 0:
            continue
        target.setValues(row_offset + row, np.asarray(columns, dtype=PETSc.IntType) + col_offset, scale * values, addv=PETSc.InsertMode.ADD_VALUES)


def add_identity_block(
    target: PETSc.Mat,
    row_offset: int,
    col_offset: int,
    size: int,
    scale: complex = 1.0,
) -> None:
    r"""
    Add a scaled identity block.

    :param target: Target matrix
    :type target: PETSc.Mat
    :param row_offset: Target row offset
    :type row_offset: int
    :param col_offset: Target column offset
    :type col_offset: int
    :param size: Identity size
    :type size: int
    :param scale: Identity scaling
    :type scale: complex
    :rtype: None
    """
    start, end = target.getOwnershipRange()
    first = max(start, row_offset)
    last = min(end, row_offset + size)
    for row in range(first, last):
        col = col_offset + row - row_offset
        target.setValue(row, col, scale, addv=PETSc.InsertMode.ADD_VALUES)


def left_diagonal_solve(diagonal: np.ndarray, rhs: PETSc.Mat) -> PETSc.Mat:
    r"""
    Solve a diagonal left system against a matrix.

    :param diagonal: Left-hand-side diagonal
    :type diagonal: np.ndarray
    :param rhs: Right-hand-side matrix
    :type rhs: PETSc.Mat

    :return: Matrix with rows divided by the diagonal
    :rtype: PETSc.Mat
    """
    diagonal = np.asarray(diagonal, dtype=complex)
    nrows, ncols = rhs.getSize()
    if len(diagonal) != nrows:
        raise ValueError("Diagonal length does not match matrix row count.")

    out = create_aij_matrix(rhs.getComm(), nrows, ncols)
    start, end = rhs.getOwnershipRange()
    for row in range(start, end):
        value = diagonal[row]
        if abs(value) == 0:
            raise ValueError("Diagonal solve encountered a zero entry.")

        columns, values = rhs.getRow(row)
        if len(columns) == 0:
            continue
        out.setValues(row, columns, values / value, addv=PETSc.InsertMode.INSERT_VALUES)

    out.assemble()
    return out


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
    pool_size = comm.Get_size()
    pool = np.arange(pool_size)
    rows, cols, vals = arrays
    idces = np.argsort(rows).reshape(-1)
    # Make sure the arrays have the correct data types (save for MPI comms)
    # and sort for CSR consistency
    rows = np.asarray(rows[idces], dtype=PETSc.IntType)
    cols = np.asarray(cols[idces], dtype=PETSc.IntType)
    vals = np.asarray(vals[idces], dtype=PETSc.ScalarType)

    # Each processor (ID rank) is aware of how many local rows are
    # owned by the other processors in the pool
    mat_row_sizes_local = np.asarray(
        comm.allgather(sizes[0][0]), dtype=PETSc.IntType
    )
    mat_row_displ = np.concatenate(([0], np.cumsum(mat_row_sizes_local[:-1])))
    ownership_ranges = np.zeros((comm.Get_size(), 2), dtype=PETSc.IntType)
    ownership_ranges[:, 0] = mat_row_displ
    ownership_ranges[:-1, 1] = ownership_ranges[1:, 0]
    ownership_ranges[-1, 1] = sizes[0][-1]

    # Each processor (ID rank) computes how many which rows, cols, data
    # need to be sent to every other processor in the pool. The number of
    # rows, cols, data values is store in the 'lengths' list
    send_rows = []
    send_cols = []
    send_vals = []
    send_lengths = []
    for i in pool:
        idces = np.argwhere(
            (rows >= ownership_ranges[i, 0]) & (rows < ownership_ranges[i, 1])
        ).reshape(-1)
        send_lengths.append(np.asarray([len(idces)], dtype=PETSc.IntType))
        send_rows.append(rows[idces])
        send_cols.append(cols[idces])
        send_vals.append(vals[idces])

    recv_bufs = [np.empty(1, dtype=PETSc.IntType) for _ in pool]
    recv_reqs = [comm.Irecv(bf, source=i) for (bf, i) in zip(recv_bufs, pool)]
    send_reqs = [comm.Isend(sz, dest=i) for (i, sz) in enumerate(send_lengths)]
    MPI.Request.waitall(send_reqs + recv_reqs)
    recv_lengths = [buf[0] for buf in recv_bufs]

    comm.Barrier()  # Sync processors after non-blocking send/recv for safety

    dtypes = [PETSc.IntType, PETSc.IntType, PETSc.ScalarType]
    my_arrays = []
    for j, array in enumerate([send_rows, send_cols, send_vals]):
        dtype = dtypes[j]
        mpi_type = get_mpi_type(np.dtype(dtype))
        recv_bufs = [
            [np.empty(recv_lengths[i], dtype=dtype), mpi_type] for i in pool
        ]
        recv_reqs = [
            comm.Irecv(
                bf, source=i, tag=rank * pool_size + i + j * pool_size**2
            )
            for (bf, i) in zip(recv_bufs, pool)
        ]
        send_reqs = [
            comm.Isend(
                array[i], dest=i, tag=rank + i * pool_size + j * pool_size**2
            )
            for i in pool
        ]
        MPI.Request.waitall(send_reqs + recv_reqs)
        my_arrays.append([recv_bufs[i][0] for i in pool])
        comm.Barrier()  # Sync processors after non-blocking send/recv for safety

    my_rows, my_cols, my_vals = [], [], []
    for i in pool:
        my_rows.extend(my_arrays[0][i])
        my_cols.extend(my_arrays[1][i])
        my_vals.extend(my_arrays[2][i])

    my_rows = (
        np.asarray(my_rows, dtype=PETSc.IntType) - ownership_ranges[rank, 0]
    )
    my_cols = np.asarray(my_cols, dtype=PETSc.IntType)
    my_vals = np.asarray(my_vals, dtype=PETSc.ScalarType)

    idces = np.argsort(my_rows).reshape(-1)
    my_rows = my_rows[idces]
    my_cols = my_cols[idces]
    my_vals = my_vals[idces]

    my_rows_ptr = np.zeros(sizes[0][0] + 1, dtype=PETSc.IntType)
    my_rows_ptr[1:] = np.cumsum(np.bincount(my_rows, minlength=sizes[0][0]))

    return my_rows_ptr, my_cols, my_vals


def assemble_harmonic_resolvent_generator(
    A: PETSc.Mat, freqs: np.array, M: Optional[PETSc.Mat] = None
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
