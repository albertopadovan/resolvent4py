import pytest
import scipy as sp
import numpy as np
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc
import os


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def rank_size(comm):
    return comm.Get_rank(), comm.Get_size()


@pytest.fixture(
    params=[
        pytest.param(
            (100, 100), marks=pytest.mark.local
        ),  # local: small problem
        pytest.param(
            (500, 500), marks=pytest.mark.development
        ),  # dev: medium problem
        pytest.param(
            (1000, 1000), marks=pytest.mark.main
        ),  # main: large problem
    ]
)
def matrix_size(request):
    """Matrix size for different test levels"""
    return request.param


@pytest.fixture
def test_output_dir(tmp_path):
    return tmp_path / "test_eigendecomposition"


def test_eigendecomposition(comm, rank_size, matrix_size, test_output_dir):
    """Test eigendecomposition with different matrix sizes based on test level."""
    rank, size = rank_size
    N, _ = matrix_size  # We use N x N for eigendecomposition
    omega = 20.0  # Angular frequency parameter
    s = 10  # Number of eigenvectors to compute

    # Create path for temporary files
    path = str(test_output_dir) + "/"

    fnames_jac, fnames = None, None
    if rank == 0:
        test_output_dir.mkdir(exist_ok=True)
        A = sp.sparse.csr_matrix(
            np.random.randn(N, N) + 1j * np.random.randn(N, N)
        )
        A = A.tocoo()
        rows, cols = A.nonzero()
        data = A.data
        arrays = [rows, cols, data]
        fnames_jac = [path + "rows.dat", path + "cols.dat", path + "vals.dat"]
        for i, array in enumerate(arrays):
            vec = PETSc.Vec().createWithArray(
                array, len(array), None, MPI.COMM_SELF
            )
            res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
            vec.destroy()
        A = A.todense()
        evals, evecs = sp.linalg.eig(A)

    comm.Barrier()

    fnames_jac = comm.bcast(fnames_jac, root=0)
    Nl = res4py.compute_local_size(N)
    A_dist = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N), (Nl, N)))
    oId = PETSc.Mat().createConstantDiagonal(A_dist.getSizes(), 1.0, comm=comm)
    oId.scale(1j * omega)
    oId.convert(PETSc.Mat.Type.MPIAIJ)
    oId.axpy(-1.0, A_dist)
    ksp = res4py.create_mumps_solver(comm, oId)
    linop = res4py.MatrixLinearOperator(comm, oId, ksp)
