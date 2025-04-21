import pytest
import numpy as np
from mpi4py import MPI


@pytest.fixture(scope="session")
def comm():
    """MPI communicator fixture."""
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def rank_size(comm):
    """Return rank and size of MPI communicator."""
    return comm.Get_rank(), comm.Get_size()


@pytest.fixture(
    params=[
        pytest.param((100, 5), marks=pytest.mark.local),
        pytest.param((1000, 10), marks=pytest.mark.development),
        pytest.param((5000, 20), marks=pytest.mark.main),
    ]
)
def matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param


@pytest.fixture
def random_matrix(matrix_size, comm):
    """Generate random test matrix."""
    N, Nc = matrix_size
    if comm.Get_rank() == 0:
        return np.random.randn(N, Nc) + 1j * np.random.randn(N, Nc)
    return None


@pytest.fixture
def test_output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    return tmp_path / "test_output"
