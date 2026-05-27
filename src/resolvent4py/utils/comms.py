__all__ = [
    "compute_local_size",
    "compute_local_size_block_aligned",
    "sequential_to_distributed_matrix",
    "sequential_to_distributed_vector",
    "distributed_to_sequential_matrix",
    "distributed_to_sequential_vector",
    "scatter_array_from_root_to_all",
]

import typing

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from .miscellaneous import get_mpi_type


def compute_local_size(Ng: int) -> int:
    r"""
    Given the global size :code:`Nglob` of a vector, compute
    the local size :code:`Nloc` that will be owned by each processor
    in the MPI pool according to the formula

    .. math::

        N_l = \begin{cases}
            \left\lfloor \frac{N_g}{S} \right\rfloor + 1 & \text{if} 
                \, N_g \text{mod} S > r \\
            \left\lfloor \frac{N_g}{S} \right\rfloor & \text{otherwise}
        \end{cases}

    where :math:`N_g` is the global size, :math:`N_l` is the local size,
    :math:`S` is the size of the MPI pool (i.e., the total number of processors)
    and :math:`r` is the rank of the current processor.

    :param Ng: global size
    :type Ng: int
    
    :return: local size
    :rtype: int
    """
    size, rank = PETSc.COMM_WORLD.getSize(), PETSc.COMM_WORLD.Get_rank()
    Nl = Ng // size + 1 if np.mod(Ng, size) > rank else Ng // size
    return Nl


def compute_local_size_block_aligned(n: int, N: int) -> typing.Tuple[int, int]:
    r"""
    Compute per-rank local sizes :code:`(nl, Nl)` for a block-structured
    :math:`N \times N` PETSc matrix made of :code:`nblocks = N // n`
    diagonal blocks of size :math:`n \times n`, such that the cumulative
    row count at every block boundary lands on a rank ownership boundary.

    This is required when the matrix will be preconditioned with
    PCBJACOBI(nblocks): PETSc snaps sub-block boundaries to the nearest
    rank boundary, so a misaligned distribution yields the wrong
    sub-blocks (some rows leak into a neighbour's sub-block where they
    have no nonzeros, making that sub-block near-singular).

    The function distributes :math:`n` rows across
    :code:`ranks_per_block = pool_size // nblocks` ranks (using the same
    formula as :func:`compute_local_size`) and replicates that pattern
    for every block. Each rank therefore owns exactly one block's
    portion, so :code:`Nl == nl`.

    Requires :code:`pool_size % nblocks == 0` and :code:`N % n == 0`.

    :param n: block size (each diagonal block is :math:`n \times n`)
    :type n: int
    :param N: total matrix size; must be divisible by :code:`n`
    :type N: int

    :return: :code:`(nl, Nl)` for the current rank.
    :rtype: tuple[int, int]
    """
    if N % n != 0:
        raise ValueError(f"N ({N}) must be divisible by n ({n})")
    nblocks = N // n
    size = PETSc.COMM_WORLD.getSize()
    rank = PETSc.COMM_WORLD.getRank()
    if size == 1:
        return n, N
    if size % nblocks != 0:
        raise ValueError(
            f"MPI pool size ({size}) must be divisible by nblocks "
            f"({nblocks}) for block boundaries to align with rank "
            f"boundaries"
        )
    ranks_per_block = size // nblocks
    sub_rank = rank % ranks_per_block
    nl = n // ranks_per_block + (1 if (n % ranks_per_block) > sub_rank else 0)
    return nl, nl


def sequential_to_distributed_matrix(
    Mat_seq: PETSc.Mat, Mat_dist: PETSc.Mat
) -> PETSc.Mat:
    r"""
    Scatter a sequential dense PETSc matrix (type PETSc.Mat.Type.DENSE) to
    a distributed dense PETSc matrix (type PETSc.Mat.Type.DENSE)

    :param Mat_seq: a sequential dense matrix
    :type Mat_seq: PETSc.Mat.Type.DENSE
    :param Mat_dist: a distributed dense matrix
    :type Mat_dist: PETSc.Mat.Type.DENSE

    :return: a distributed dense matrix
    :rtype: PETSc.Mat.Type.DENSE
    """
    array = Mat_seq.getDenseArray()
    r0, r1 = Mat_dist.getOwnershipRange()
    rows = np.arange(r0, r1, dtype=PETSc.IntType)
    cols = np.arange(0, Mat_seq.getSizes()[-1][-1], dtype=PETSc.IntType)
    Mat_dist.setValues(rows, cols, array[r0:r1,].reshape(-1))
    Mat_dist.assemble(None)
    return Mat_dist


def sequential_to_distributed_vector(
    vec_seq: PETSc.Vec, vec_dist: PETSc.Vec
) -> PETSc.Vec:
    r"""
    Scatter a sequential PETSc vector (type PETSc.Vec.Type.SEQ) to
    a distributed PETSc vector (type PETSc.Vec.Type.STANDARD)

    :param vec_seq: a sequential vector
    :type vec_seq: PETSc.Vec.Type.SEQ
    :param vec_dist: a distributed vector
    :type vec_dist: PETSc.Vec.Type.STANDARD

    :return: a distributed vector
    :rtype: PETSc.Vec.Type.STANDARD
    """
    array = vec_seq.getArray()
    r0, r1 = vec_dist.getOwnershipRange()
    rows = np.arange(r0, r1, dtype=PETSc.IntType)
    vec_dist.setValues(rows, array[r0:r1,])
    vec_dist.assemble()
    return vec_dist


def distributed_to_sequential_matrix(Mat_dist: PETSc.Mat) -> PETSc.Mat:
    r"""
    Allgather dense distributed PETSc matrix into a dense sequential PETSc
    matrix.

    :param Mat_dist: a distributed dense matrix
    :type Mat_dist: PETSc.Mat.Type.DENSE

    :rtype: PETSc.Mat.Type.DENSE
    """
    comm = Mat_dist.getComm().tompi4py()
    array = Mat_dist.getDenseArray().copy().reshape(-1)
    counts = np.asarray(comm.allgather(len(array)))
    disps = np.concatenate(([0], np.cumsum(counts[:-1])))
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    sizes = Mat_dist.getSizes()
    nr, nc = sizes[0][-1], sizes[-1][-1]
    recvbuf = recvbuf.reshape((nr, nc))
    Mat_seq = PETSc.Mat().createDense((nr, nc), None, recvbuf, PETSc.COMM_SELF)
    return Mat_seq


def distributed_to_sequential_vector(vec_dist: PETSc.Vec) -> PETSc.Vec:
    r"""
    Allgather distribued PETSc vector into PETSc sequential vector.

    :param vec_dist: a distributed vector
    :type vec_dist: PETSc.Vec.Type.STANDARD

    :rtype: PETSc.Vec.Type.SEQ
    """
    comm = vec_dist.getComm().tompi4py()
    array = vec_dist.getArray().copy()
    counts = comm.allgather(len(array))
    disps = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(PETSc.IntType)
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    vec_seq = PETSc.Vec().createWithArray(recvbuf, comm=PETSc.COMM_SELF)
    return vec_seq


def scatter_array_from_root_to_all(
    array: np.array, locsize: typing.Optional[int] = None
) -> np.array:
    r"""
    Scatter numpy array from root to all other processors

    :param array: numpy array to be scattered
    :type array: numpy.array
    :param locsize: local size owned by each processor. If :code:`None`,
        this is computed using the same logic as in the
        :func:`.compute_local_size` routine.
    :type locsize: Optional[int] default to None

    :rtype: numpy.array
    """
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    counts, displs = None, None
    if locsize == None:
        if rank == 0:
            n = len(array)
            counts = np.asarray(
                [
                    n // size + 1 if np.mod(n, size) > j else n // size
                    for j in range(size)
                ]
            )
            displs = np.concatenate(([0], np.cumsum(counts[:-1])))
            count = counts[0]
            dtype = array.dtype
            for j in range(1, size):
                comm.send(counts[j], dest=j, tag=0)
                comm.send(dtype, dest=j, tag=1)
        else:
            count = comm.recv(source=0, tag=0)
            dtype = comm.recv(source=0, tag=1)
    else:
        count = locsize
        counts = comm.gather(locsize, root=0)
        if rank == 0:
            dtype = array.dtype
            displs = np.concatenate(([0], np.cumsum(counts[:-1])))
            for j in range(1, size):
                comm.send(dtype, dest=j, tag=1)
        else:
            dtype = comm.recv(source=0, tag=1)
    recvbuf = np.empty(count, dtype=dtype)
    counts = (
        np.asarray(counts, dtype=PETSc.IntType)
        if counts is not None
        else counts
    )
    displs = (
        np.asarray(displs, dtype=PETSc.IntType)
        if displs is not None
        else displs
    )
    comm.Scatterv(
        [array, counts, displs, get_mpi_type(dtype)], recvbuf, root=0
    )
    return recvbuf
