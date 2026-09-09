__all__ = ["get_mpi_type", "petscprint", "get_memory_usage"]

import tracemalloc

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

numpy_to_mpi_dtype = {
    np.dtype(np.int32): MPI.INT,
    np.dtype(np.int64): MPI.INT64_T,
    np.dtype(np.float64): MPI.DOUBLE,
    np.dtype(np.complex64): MPI.COMPLEX,
    np.dtype(np.complex128): MPI.DOUBLE_COMPLEX,
}


def get_mpi_type(dtype: np.dtype) -> MPI.Datatype:
    r"""
    Get the corresponding MPI type for a given numpy data type.

    :param dtype: (e.g., :code:`np.dtype(PETSc.IntType)`)
    :type dtype: np.dtypes
    :rtype: MPI.Datatype
    """
    mpi_dtype = numpy_to_mpi_dtype.get(dtype)
    if mpi_dtype is None:
        raise ValueError(f"No MPI type found for numpy dtype {dtype}")
    return mpi_dtype


def petscprint(comm: PETSc.Comm, arg: any) -> None:
    r"""
    Print to terminal

    :param comm: MPI communicator (PETSc.COMM_WORLD or PETSc.COMM_SELF)
    :type comm: PETSc.Comm
    :param arg: argument to be fed into print()
    :type arg: Any
    """
    if comm == PETSc.COMM_SELF or comm == MPI.COMM_SELF:
        print(arg)
    else:
        if PETSc.COMM_WORLD.getRank() == 0:
            print(arg)


def get_memory_usage(comm: PETSc.Comm) -> float:
    r"""
    Compute the used memory (in MB) across the MPI pool.

    :param comm: MPI communicator.  When equal to :code:`PETSc.COMM_WORLD`
        the returned value is the sum across all ranks; otherwise it is
        the memory used on the calling rank only
    :type comm: PETSc.Comm

    :return: memory usage in megabytes
    :rtype: float
    """
    comm = comm.tompi4py()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    total_memory = sum(stat.size for stat in top_stats) / (1024**2)
    value = (
        sum(comm.allgather(total_memory))
        if comm == MPI.COMM_WORLD
        else total_memory
    )
    return value
