from .. import np
from .. import MPI
from .. import tracemalloc

numpy_to_mpi_dtype = {
    np.dtype(np.int32): MPI.INT,
    np.dtype(np.int64): MPI.INT64_T,
    np.dtype(np.float64): MPI.DOUBLE,
    np.dtype(np.complex64): MPI.COMPLEX,
    np.dtype(np.complex128): MPI.DOUBLE_COMPLEX,
}


def get_mpi_type(dtype: np.dtypes) -> MPI.Datatype:
    r"""
    Get the corresponding MPI type for a given numpy data type.

    :param dtype: (e.g., :code:`np.dtype(np.int32)`)
    :type dtype: np.dtypes
    :rtype: MPI.Datatype
    """
    mpi_dtype = numpy_to_mpi_dtype.get(dtype)
    if mpi_dtype is None:
        raise ValueError(f"No MPI type found for numpy dtype {dtype}")
    return mpi_dtype


def petscprint(comm: MPI.Comm, arg: any) -> None:
    r"""
    Print to terminal

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param arg: argument to be fed into print()
    :type arg: any
    """
    if comm == MPI.COMM_SELF:
        print(arg)
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(arg)


def get_memory_usage(comm: MPI.Comm) -> float:
    r"""
    Compute the used memory (in Mb) across the MPI pool

    :type comm: MPI.Comm
    :rtype: float
    """
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    total_memory = sum(stat.size for stat in top_stats) / (1024**2)
    value = (
        sum(MPI.COMM_WORLD.allgather(total_memory))
        if comm == MPI.COMM_WORLD
        else total_memory
    )
    return value
