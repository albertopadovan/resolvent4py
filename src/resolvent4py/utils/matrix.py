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
    "extract_matrix_block",
    "extract_block_banded",
    "assemble_matrix_from_coo",
]

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Optional

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
    Y = X.duplicate() if Y is None else Y
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
    :param MatHT: matrix with the correct layout to hold the
        hermitian transpose of :code:`Mat` (out-of-place only; allocated
        internally when :code:`None`)
    :type MatHT: Optional[PETSc.Mat] defaults to :code:`None`

    :return: :math:`\text{Mat}^{*}` (a new matrix, or :code:`MatHT`, or the
        in-place transposed :code:`Mat`)
    :rtype: PETSc.Mat
    """
    if not in_place:
        if MatHT is None:
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

    Uses Alltoall for counts and pairwise Sendrecv for data, which avoids
    MPI request exhaustion and tag overflow on large machines while keeping
    memory usage bounded.

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
    ownership_ranges = np.zeros((pool_size, 2), dtype=PETSc.IntType)
    ownership_ranges[:, 0] = mat_row_displ
    ownership_ranges[:-1, 1] = ownership_ranges[1:, 0]
    ownership_ranges[-1, 1] = sizes[0][-1]

    # Determine which rank owns each row and partition entries by destination
    dest = np.searchsorted(mat_row_displ, rows, side="right") - 1
    send_counts = np.bincount(dest, minlength=pool_size).astype(PETSc.IntType)

    # Sort entries by destination rank for contiguous per-rank slicing
    sort_idx = np.argsort(dest, kind="stable")
    rows = np.ascontiguousarray(rows[sort_idx])
    cols = np.ascontiguousarray(cols[sort_idx])
    vals = np.ascontiguousarray(vals[sort_idx])
    send_displs = np.concatenate(([0], np.cumsum(send_counts[:-1])))

    # Exchange counts via Alltoall (O(P) integers, negligible memory)
    recv_counts = np.empty(pool_size, dtype=PETSc.IntType)
    comm.Alltoall(send_counts, recv_counts)

    # Exchange data via pairwise Sendrecv (1 send + 1 recv at a time,
    # no tag management, deadlock-free)
    total_recv = int(recv_counts.sum())
    my_rows = np.empty(total_recv, dtype=PETSc.IntType)
    my_cols = np.empty(total_recv, dtype=PETSc.IntType)
    my_vals = np.empty(total_recv, dtype=PETSc.ScalarType)
    recv_displs = np.concatenate(([0], np.cumsum(recv_counts[:-1])))

    for k in range(pool_size):
        send_to = (rank + k) % pool_size
        recv_from = (rank - k) % pool_size
        s0 = int(send_displs[send_to])
        sn = int(send_counts[send_to])
        r0 = int(recv_displs[recv_from])
        rn = int(recv_counts[recv_from])
        comm.Sendrecv(
            rows[s0 : s0 + sn],
            dest=send_to,
            recvbuf=my_rows[r0 : r0 + rn],
            source=recv_from,
        )
        comm.Sendrecv(
            cols[s0 : s0 + sn],
            dest=send_to,
            recvbuf=my_cols[r0 : r0 + rn],
            source=recv_from,
        )
        comm.Sendrecv(
            vals[s0 : s0 + sn],
            dest=send_to,
            recvbuf=my_vals[r0 : r0 + rn],
            source=recv_from,
        )

    # Convert to local row indices and sort by row for CSR
    my_rows = my_rows - ownership_ranges[rank, 0]
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
    if M is None:
        omId.axpy(1.0, A)
        return omId
    else:
        Mat = omId.matMult(M)
        Mat.axpy(1.0, A)
        omId.destroy()
        return Mat


def assemble_matrix_from_coo(comm, coo_arrays, mat_sizes):
    r"""
    Assemble a PETSc AIJ sparse matrix from COO arrays via CSR conversion.

    :param comm: MPI communicator
    :type comm: PETSc.Comm
    :param coo_arrays: ``[rows, cols, vals]`` in COO format (only populated
        on rank 0; ``None`` on other ranks)
    :type coo_arrays: list
    :param mat_sizes: PETSc size spec
        ``((local_rows, global_rows), (local_cols, global_cols))``
    :type mat_sizes: tuple

    :return: assembled PETSc sparse matrix
    :rtype: PETSc.Mat
    """
    from .comms import scatter_array_from_root_to_all

    rows_coo, cols_coo, data_coo = coo_arrays
    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    data = scatter_array_from_root_to_all(data_coo)
    rows_ptr, cols, vals = convert_coo_to_csr([rows, cols, data], mat_sizes)

    M = PETSc.Mat().createAIJ(mat_sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble()

    return M


def extract_matrix_block(
    Mat: PETSc.Mat, nblocks: int, rowblock: int, colblock: int
) -> PETSc.Mat:
    r"""
    Extract a single :math:`N \times N` block from a block-structured
    :math:`nN \times nN` PETSc matrix, where the block at position
    ``(rowblock, colblock)`` occupies rows
    ``rowblock*N .. (rowblock+1)*N - 1`` and columns
    ``colblock*N .. (colblock+1)*N - 1``.

    The extraction is done via two selector multiplications:

    .. math::

        \text{block} = \hat{I}_r^T \; \text{Mat} \; \hat{I}_c

    where :math:`\hat{I}_r` and :math:`\hat{I}_c` are :math:`nN \times N`
    matrices with :math:`I_N` at the appropriate block-row.

    :param Mat: assembled :math:`nN \times nN` PETSc sparse matrix
    :type Mat: PETSc.Mat
    :param nblocks: number of blocks along each dimension
    :type nblocks: int
    :param rowblock: 0-based row-block index
    :type rowblock: int
    :param colblock: 0-based column-block index
    :type colblock: int

    :return: the extracted :math:`N \times N` PETSc sparse matrix
    :rtype: PETSc.Mat
    """
    from .comms import compute_local_size

    comm = Mat.getComm()
    size = Mat.getSizes()[0]
    N = size[-1] // nblocks

    # Build selector for the column block: Ic is nN x N with I_N at colblock
    rows_coo, cols_coo, data_coo = None, None, None
    if comm.getRank() == 0:
        data_coo = np.ones(N, dtype=PETSc.ScalarType)
        cols_coo = np.arange(N, dtype=PETSc.IntType)
        rows_coo = cols_coo + colblock * N

    mat_sizes_sel = (size, (compute_local_size(N), N))
    Ic = assemble_matrix_from_coo(
        comm, [rows_coo, cols_coo, data_coo], mat_sizes_sel
    )

    # Mat @ Ic gives nN x N (the colblock-th block-column)
    MatIc = Mat.matMult(Ic)
    Ic.destroy()

    # Build selector for the row block: Ir is nN x N with I_N at rowblock
    rows_coo, cols_coo, data_coo = None, None, None
    if comm.getRank() == 0:
        data_coo = np.ones(N, dtype=PETSc.ScalarType)
        cols_coo = np.arange(N, dtype=PETSc.IntType)
        rows_coo = cols_coo + rowblock * N

    Ir = assemble_matrix_from_coo(
        comm, [rows_coo, cols_coo, data_coo], mat_sizes_sel
    )

    # Ir^H @ MatIc = (N x nN) @ (nN x N) = N x N block
    Ir.hermitianTranspose()
    block = Ir.matMult(MatIc)
    Ir.destroy()
    MatIc.destroy()

    return block


def extract_block_banded(
    Mat: PETSc.Mat, nblocks: int, n_off_diags: int = 0
) -> PETSc.Mat:
    r"""
    Extract the block-banded part of a block-structured
    :math:`nN \times nN` PETSc matrix and assemble it as a new
    :math:`nN \times nN` sparse matrix.  Every block :math:`(i, j)`
    with :math:`|i - j| \le k` is retained, where
    :math:`k = ` ``n_off_diags``:

    * ``n_off_diags = 0`` → block-diagonal,
    * ``n_off_diags = 1`` → block-tridiagonal,
    * ``n_off_diags = 2`` → block-pentadiagonal, etc.

    Uses the identity

    .. math::

        B = \sum_{|i - j| \le k} E_i \, A \, E_j,

    where :math:`E_k` is the :math:`nN \times nN` block projector with
    :math:`I_N` in the :math:`(k, k)` block position and zeros
    elsewhere, so that :math:`E_i A E_j` isolates the :math:`(i, j)`
    block of :math:`A` in place.

    :param Mat: assembled :math:`nN \times nN` PETSc sparse matrix
    :type Mat: PETSc.Mat
    :param nblocks: number of blocks along each dimension
    :type nblocks: int
    :param n_off_diags: number of block off-diagonals to keep on each
        side of the main block-diagonal (default 0, i.e. block-diagonal)
    :type n_off_diags: int

    :return: the block-banded :math:`nN \times nN` PETSc sparse matrix
    :rtype: PETSc.Mat
    """
    if n_off_diags < 0:
        raise ValueError(f"n_off_diags must be >= 0; got {n_off_diags}.")

    comm = Mat.getComm()
    size = Mat.getSizes()[0]
    nN = size[-1]
    N = nN // nblocks
    # Use Mat's own local row count so the Es projectors share its parallel
    # layout — necessary because Mat.matMult requires the operands to have
    # matching local row/col partitions (block-aligned, default, etc.).
    Nl = size[0]

    mat_sizes = ((Nl, nN), (Nl, nN))

    # Build the block projectors E_0, ..., E_{nblocks-1} once and reuse
    # them across the diagonal and off-diagonal accumulations.
    Es = []
    for k in range(nblocks):
        rows_coo, cols_coo, vals_coo = None, None, None
        if comm.getRank() == 0:
            rows_coo = np.arange(k * N, (k + 1) * N, dtype=PETSc.IntType)
            cols_coo = rows_coo.copy()
            vals_coo = np.ones(N, dtype=PETSc.ScalarType)
        Es.append(
            assemble_matrix_from_coo(
                comm, [rows_coo, cols_coo, vals_coo], mat_sizes
            )
        )

    # (row-block, col-block) pairs within the requested bandwidth
    pairs = [
        (i, j)
        for i in range(nblocks)
        for j in range(nblocks)
        if abs(i - j) <= n_off_diags
    ]

    B = None
    for i, j in pairs:
        tmp = Mat.matMult(Es[j])  # Mat @ E_j  → keeps col-block j
        blk = Es[i].matMult(tmp)  # E_i @ Mat @ E_j  → block (i, j)
        tmp.destroy()
        if B is None:
            B = blk
        else:
            B.axpy(1.0, blk)
            blk.destroy()

    for Ek in Es:
        Ek.destroy()

    # Drop explicit zeros from the CSR: each E_i @ Mat @ E_j triple product
    # can leave structurally-allocated entries that happen to be numerically
    # zero (and the axpy union preserves them), and downstream consumers like
    # MUMPS' distributed-input ingestion choke on the resulting padded j[]
    # arrays.  IGNORE_ZERO_ENTRIES + a re-assemble compress the pattern.
    B.assemble()
    B.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    try:
        B.eliminateZeros()
    except AttributeError:
        pass

    return B
