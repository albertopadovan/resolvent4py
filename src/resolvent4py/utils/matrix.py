__all__ = [
    "create_dense_matrix",
    "create_AIJ_identity",
    "mat_solve_hermitian_transpose",
    "hermitian_transpose",
    "convert_coo_to_csr",
    "convert_coo_to_csr_v2",
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


def convert_coo_to_csr_v2(
    arrays: tuple[np.array, np.array, np.array],
    sizes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.array, np.array, np.array]:
    r"""
    Convert arrays = [row indices, col indices, values] for COO matrix
    assembly to [row pointers, col indices, values] for CSR matrix assembly.
    (Petsc4py currently does not support COO matrix assembly, hence the need
    to convert.)

    This version replaces the O(P^2) non-blocking Isend/Irecv pattern in
    :func:`convert_coo_to_csr` with Alltoall for counts and pairwise
    Sendrecv for data. This avoids MPI request exhaustion and tag overflow
    on large machines while keeping memory usage bounded.

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
            rows[s0 : s0 + sn], dest=send_to,
            recvbuf=my_rows[r0 : r0 + rn], source=recv_from,
        )
        comm.Sendrecv(
            cols[s0 : s0 + sn], dest=send_to,
            recvbuf=my_cols[r0 : r0 + rn], source=recv_from,
        )
        comm.Sendrecv(
            vals[s0 : s0 + sn], dest=send_to,
            recvbuf=my_vals[r0 : r0 + rn], source=recv_from,
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

    rows_ptr, cols, vals = convert_coo_to_csr_v2([rows, rows, vals], A.getSizes())
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
