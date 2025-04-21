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
        pytest.param((100, 5), marks=pytest.mark.local),
        pytest.param((1000, 10), marks=pytest.mark.development),
        pytest.param((5000, 20), marks=pytest.mark.main),
    ]
)
def matrix_size(request):
    return request.param


@pytest.fixture
def random_matrix(matrix_size):
    N, Nc = matrix_size
    return np.random.randn(N, Nc) + 1j * np.random.randn(N, Nc)


@pytest.fixture
def test_output_dir(tmp_path):
    return tmp_path / "test_randomized_svd"


def test_randomized_svd(
    comm, rank_size, matrix_size, random_matrix, test_output_dir
):
    """Test randomized SVD with different matrix sizes based on test level."""
    rank, size = rank_size
    N, Nc = matrix_size

    # Create path for temporary files
    path = str(test_output_dir) + "/"

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
    linop = res4py.MatrixLinearOperator(comm, A_dist)

    # Run randomized SVD
    k = min(10, min(N, Nc))  # Number of singular values to compute
    U, S, V = res4py.linalg.randomized_svd(linop, linop.apply_mat, k, 3, 3)
    Sseq = np.diag(S)

    if rank == 0:
        plt.figure()
        plt.plot(s.real, "ko")
        plt.plot(Sseq.real, "rx")
        plt.gca().set_yscale("log")
        plt.savefig(path + "svals.png")

    for i in range(len(Sseq)):
        u = U.getColumn(i)
        v = V.getColumn(i)
        Av = linop.apply(v)
        error = np.abs(Av.dot(u) - Sseq[i])
        res4py.petscprint(comm, "Error = %1.15e" % error)
        U.restoreColumn(i, u)
        V.restoreColumn(i, v)
