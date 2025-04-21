from .. import np
from .. import MPI
from .. import PETSc
from .. import typing
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
    size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    Nl = Ng // size + 1 if np.mod(Ng, size) > rank else Ng // size
    return Nl


def sequential_to_distributed_matrix(
    Mat_seq: PETSc.Mat, Mat_dist: PETSc.Mat
) -> None:
    r"""
    Scatter a sequential dense PETSc matrix (type PETSc.Mat.Type.DENSE) to
    a distributed dense PETSc matrix (type PETSc.Mat.Type.DENSE)

    :param Mat_seq: a sequential dense matrix
    :type Mat_seq: PETSc.Mat.Type.DENSE
    :param Mat_dist: a distributed dense matrix
    :type Mat_dist: PETSc.Mat.Type.DENSE

    :return: None
    """
    array = Mat_seq.getDenseArray()
    r0, r1 = Mat_dist.getOwnershipRange()
    rows = np.arange(r0, r1)
    cols = np.arange(0, Mat_seq.getSizes()[-1][-1])
    Mat_dist.setValues(rows, cols, array[r0:r1,].reshape(-1))
    Mat_dist.assemble(None)


def sequential_to_distributed_vector(
    vec_seq: PETSc.Vec, vec_dist: PETSc.Vec
) -> None:
    r"""
    Scatter a sequential dense PETSc matrix (type PETSc.Vec.Type.SEQ) to
    a distributed dense PETSc matrix (type PETSc.Vec.Type.STANDARD)

    :param vec_seq: a sequential vector
    :type vec_seq: PETSc.Vec.Type.SEQ
    :param vec_dist: a distributed vector
    :type vec_dist: PETSc.Vec.Type.STANDARD

    :return: None
    """
    array = vec_seq.getArray()
    r0, r1 = vec_dist.getOwnershipRange()
    rows = np.arange(r0, r1)
    vec_dist.setValues(rows, array[r0:r1,])
    vec_dist.assemble(None)


def distributed_to_sequential_matrix(
    comm: MPI.Comm, Mat_dist: PETSc.Mat
) -> PETSc.Mat:
    r"""
    Allgather dense distributed PETSc matrix into a dense sequential PETSc
    matrix.

    :param comm: MPI communicator :code:`MPI.COMM_WORLD`
    :type comm: MPI.Comm
    :param Mat_dist: a distributed dense matrix
    :type Mat_dist: PETSc.Mat.Type.DENSE

    :rtype: PETSc.Mat.Type.DENSE
    """
    array = Mat_dist.getDenseArray().copy().reshape(-1)
    counts = np.asarray(comm.allgather(len(array)))
    disps = np.concatenate(([0], np.cumsum(counts[:-1])))
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    sizes = Mat_dist.getSizes()
    nr, nc = sizes[0][-1], sizes[-1][-1]
    recvbuf = recvbuf.reshape((nr, nc))
    Mat_seq = PETSc.Mat().createDense((nr, nc), None, recvbuf, MPI.COMM_SELF)
    return Mat_seq


def distributed_to_sequential_vector(
    comm: MPI.Comm, vec_dist: PETSc.Vec
) -> PETSc.Vec:
    r"""
    Allgather distribued PETSc vector into PETSc sequential vector.

    :param comm: MPI communicator
    :type comm: MPI.Comm
    :param vec_dist: a distributed vector
    :type vec_dist: PETSc.Vec.Type.STANDARD

    :rtype: PETSc.Vec.Type.SEQ
    """
    array = vec_dist.getArray().copy()
    counts = comm.allgather(len(array))
    disps = np.concatenate(([0], np.cumsum(counts[:-1])))
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    vec_seq = PETSc.Vec().createWithArray(recvbuf, comm=MPI.COMM_SELF)
    return vec_seq


def scatter_array_from_root_to_all(
    comm: MPI.Comm, array: np.array, locsize: typing.Optional[int]
) -> np.array:
    r"""
    Scatter numpy array from root to all other processors

    :param comm: MPI communicator
    :type comm: MPI.Comm
    :param array: numpy array to be scattered
    :type array: numpy.array
    :param locsize: local size owned by each processor. If :code:`None`,
        this is computed using the same logic as in the
        :func:`.compute_local_size` routine.
    :type locsize: Optional[int]

    :rtype: numpy.array
    """
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
        np.asarray(counts, dtype=np.int64) if counts is not None else counts
    )
    displs = (
        np.asarray(displs, dtype=np.int64) if displs is not None else displs
    )
    comm.Scatterv(
        [array, counts, displs, get_mpi_type(dtype)], recvbuf, root=0
    )
    return recvbuf
