import pytest
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc
import pathlib


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def rank_size(comm):
    return comm.Get_rank(), comm.Get_size()


@pytest.fixture(
    params=[
        pytest.param(
            (10, 10, 2, 60, 7), marks=pytest.mark.local
        ),  # local: small problem
        pytest.param(
            (100, 100, 3, 300, 7), marks=pytest.mark.development
        ),  # dev: medium problem
        pytest.param(
            (1000, 1000, 5, 600, 7), marks=pytest.mark.main
        ),  # main: large problem
    ]
)
def test_params(request):
    """Test parameters for different test levels: matrix_size, n_periods, n_timesteps, n_rand"""
    return request.param


@pytest.fixture
def matrix_size(test_params):
    N, Nc, _, _, _ = test_params
    return (N, Nc)


@pytest.fixture
def random_matrix(matrix_size):
    N, Nc = matrix_size
    return np.random.randn(N, Nc) + 1j * np.random.randn(N, Nc)


@pytest.fixture
def test_output_dir(tmp_path):
    return tmp_path / "test_randomized_time_stepping_svd"


def test_randomized_time_stepping_svd(
    comm, rank_size, test_params, random_matrix, test_output_dir
):
    """Test randomized time-stepping SVD with different matrix sizes based on test level."""
    rank, size = rank_size
    N, Nc, n_periods, n_timesteps, n_rand = test_params

    # Create path for temporary files
    path = str(test_output_dir) + "/"

    # Set up frequencies array (omega)
    omega = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Example frequencies

    fnames_jac, fnames = None, None
    if rank == 0:
        test_output_dir.mkdir(exist_ok=True)
        A = sp.sparse.csr_matrix(random_matrix)
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
        u, s, v = sp.linalg.svd(A.conj().T)
        v = v.conj().T

    comm.Barrier()

    fnames_jac = comm.bcast(fnames_jac, root=0)
    Nl = res4py.compute_local_size(N)
    Ncl = res4py.compute_local_size(Nc)
    A_dist = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N), (Ncl, Nc)))

    # Create linear operator
    linop = res4py.MatrixLinearOperator(comm, A_dist)

    mass = PETSc.Mat().create(comm=comm)
    mass.setSizes(A_dist.getSizes())
    mass.setType("aij")
    mass.setUp()

    start, end = mass.getOwnershipRange()
    for i in range(start, end):
        mass.setValue(i, i, 1.0)

    mass.assemblyBegin()
    mass.assemblyEnd()
    mass_linop = res4py.MatrixLinearOperator(comm, mass)

    # Run randomized time-stepping SVD
    n_loops = 3  # Power iterations
    n_svals = min(3, min(N, Nc))  # Number of singular values to compute
    U, S, V = res4py.linalg.randomized_time_stepping_svd(
        linop,
        mass_linop,
        linop.apply_mat,
        omega,
        n_periods,
        n_timesteps,
        n_rand,
        n_loops,
        n_svals,
    )
    # if rank == 0:
    #     plt.figure()
    #     plt.plot(s.real, 'ko')
    #     plt.plot(Sseq.real, 'rx')
    #     plt.gca().set_yscale('log')
    #     plt.savefig(path + "svals.png")
