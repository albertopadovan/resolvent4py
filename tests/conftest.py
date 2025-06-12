import sys
import os
from mpi4py import MPI
from . import pytest_utils
import pytest

# This ensures that only the root processor prints to the terminal
if MPI.COMM_WORLD.Get_rank() != 0:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


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
        pytest.param((50, 50), marks=pytest.mark.local),
        pytest.param((100, 100), marks=pytest.mark.development),
        pytest.param((300, 300), marks=pytest.mark.main),
    ]
)
def square_matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param

@pytest.fixture(
    params=[
        pytest.param((60, 60), marks=pytest.mark.local),
        pytest.param((80, 80), marks=pytest.mark.development),
        pytest.param((100, 100), marks=pytest.mark.main),
    ]
)
def square_negative_semidefinite_matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param

@pytest.fixture(
    params=[
        pytest.param((50, 20), marks=pytest.mark.local),
        pytest.param((100, 70), marks=pytest.mark.development),
        pytest.param((300, 190), marks=pytest.mark.main),
    ]
)
def rectangular_matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param


@pytest.fixture
def square_random_matrix(comm, square_matrix_size):
    """Generate random test matrix."""
    return pytest_utils.generate_random_matrix(comm, square_matrix_size)


@pytest.fixture
def square_random_negative_semidefinite_matrix(comm, square_negative_semidefinite_matrix_size):
    """Generate random negative semidefinite test matrix."""
    return pytest_utils.generate_negative_semidefinite_matrix(comm, square_negative_semidefinite_matrix_size)

@pytest.fixture
def rectangular_random_matrix(comm, rectangular_matrix_size):
    """Generate random test matrix."""
    return pytest_utils.generate_random_matrix(comm, rectangular_matrix_size)


@pytest.fixture
def test_output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    return tmp_path / "test_output"
