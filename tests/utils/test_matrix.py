import numpy as np
import scipy as sp
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.utils.comms import (
    scatter_array_from_root_to_all,
    compute_local_size,
)
from resolvent4py.utils.matrix import (
    convert_coo_to_csr,
    extract_block_diagonal,
)
from .. import pytest_utils


def test_create_dense_matrix(comm, square_matrix_size):
    r"""Test create_dense_matrix creates a matrix of correct size"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    M = res4py.create_dense_matrix(comm, ((Nl, N), (Nl, N)))
    sizes = M.getSizes()
    assert sizes[0][-1] == N
    assert sizes[-1][-1] == N
    M.destroy()


def test_create_AIJ_identity(comm, square_matrix_size):
    r"""Test that create_AIJ_identity produces an actual identity matrix"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    Id = res4py.create_AIJ_identity(comm, ((Nl, N), (Nl, N)))

    # Multiply by a random vector, should return the same vector
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = x.duplicate()
    Id.mult(x, y)

    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(ys.getArray() - xpython)
    ys.destroy()
    x.destroy()
    y.destroy()
    Id.destroy()
    assert error < 1e-14


def test_hermitian_transpose_out_of_place(comm, square_matrix_size):
    r"""Test hermitian_transpose with in_place=False"""
    N = square_matrix_size[0]
    Apetsc, Apython = pytest_utils.generate_random_matrix(comm, (N, N))

    # Compute A^H via the utility
    Ah = Apetsc.copy()
    Ah.hermitianTranspose()

    # Verify by multiplying A^H * x and comparing to python
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = x.duplicate()
    Ah.mult(x, y)

    ys = res4py.distributed_to_sequential_vector(y)
    expected = Apython.conj().T.dot(xpython)
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(expected)
    ys.destroy()
    x.destroy()
    y.destroy()
    Apetsc.destroy()
    Ah.destroy()
    assert error < 1e-10


def test_mat_solve_hermitian_transpose(comm, square_random_matrix):
    r"""Test mat_solve_hermitian_transpose: A^{-H} X"""
    from resolvent4py.utils.matrix import mat_solve_hermitian_transpose

    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    ksp = res4py.create_mumps_solver(Apetsc)

    s = 3
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Xm = X.getMat()
    Ym = mat_solve_hermitian_transpose(ksp, Xm)
    X.restoreMat(Xm)

    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Ypython = sp.linalg.inv(Apython).conj().T.dot(Xpython)
    error = np.linalg.norm(Yms.getDenseArray() - Ypython) / np.linalg.norm(
        Ypython
    )
    Yms.destroy()
    Ym.destroy()
    X.destroy()
    ksp.destroy()
    assert error < 1e-8


def test_assemble_harmonic_resolvent_generator(comm, square_matrix_size):
    r"""Test assemble_harmonic_resolvent_generator produces T = -iwM + A
    (with M = Id by default)"""
    from resolvent4py.utils.matrix import assemble_harmonic_resolvent_generator

    N = square_matrix_size[0]
    nblocks = 3
    Ntotal = N * nblocks
    Nl = res4py.compute_local_size(Ntotal)
    Apetsc, Apython = pytest_utils.generate_random_matrix(
        comm, (Ntotal, Ntotal)
    )

    omega = 1.5
    freqs = omega * np.array([-1, 0, 1])
    T = assemble_harmonic_resolvent_generator(Apetsc, freqs)

    # Expected: T = A + diag(-i*freq_k * I_N) for each block k
    Tpython = Apython.copy()
    for k in range(nblocks):
        Tpython[k * N : (k + 1) * N, k * N : (k + 1) * N] += (
            -1j * freqs[k] * np.eye(N)
        )

    # Verify via matvec
    x, xpython = pytest_utils.generate_random_vector(comm, Ntotal)
    y = x.duplicate()
    T.mult(x, y)
    ys = res4py.distributed_to_sequential_vector(y)
    expected = Tpython.dot(xpython)
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(expected)
    ys.destroy()
    x.destroy()
    y.destroy()
    T.destroy()
    Apetsc.destroy()
    assert error < 1e-10


def _numpy_to_petsc(comm, A_np):
    """Convert a dense numpy matrix (known on all ranks) to a distributed
    PETSc AIJ matrix."""
    N = A_np.shape[0]
    rows_coo, cols_coo, vals_coo = None, None, None
    if comm.getRank() == 0:
        r, c = np.nonzero(A_np)
        rows_coo = np.asarray(r, dtype=PETSc.IntType)
        cols_coo = np.asarray(c, dtype=PETSc.IntType)
        vals_coo = np.asarray(A_np[r, c], dtype=PETSc.ScalarType)
    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    vals = scatter_array_from_root_to_all(vals_coo)
    Nl = compute_local_size(N)
    sizes = ((Nl, N), (Nl, N))
    rp, cs, vs = convert_coo_to_csr([rows, cols, vals], sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rp, cs))
    M.setValuesCSR(rp, cs, vs, True)
    M.assemble()
    return M


def test_extract_block_diagonal(comm):
    r"""Test that extract_block_diagonal extracts the correct diagonal blocks
    from a random block-structured matrix."""
    n = 4
    nblocks = 5
    nN = n * nblocks

    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((nN, nN)) + 1j * rng.standard_normal((nN, nN))
    A_np = comm.tompi4py().bcast(A_np, root=0)

    A_petsc = _numpy_to_petsc(comm, A_np)
    B_petsc = extract_block_diagonal(A_petsc, nblocks)

    # Build expected block-diagonal in numpy
    B_expected = np.zeros_like(A_np)
    for k in range(nblocks):
        B_expected[k * n : (k + 1) * n, k * n : (k + 1) * n] = A_np[
            k * n : (k + 1) * n, k * n : (k + 1) * n
        ]

    # Verify via matvec
    x, x_np = pytest_utils.generate_random_vector(comm, nN)
    y = x.duplicate()
    B_petsc.mult(x, y)
    y_seq = res4py.distributed_to_sequential_vector(y)
    y_expected = B_expected @ x_np
    error = np.linalg.norm(y_seq.getArray() - y_expected) / np.linalg.norm(
        y_expected
    )

    y_seq.destroy()
    x.destroy()
    y.destroy()
    A_petsc.destroy()
    B_petsc.destroy()
    assert error < 1e-10, f"extract_block_diagonal error: {error:.2e}"


def test_extract_block_diagonal_of_block_diagonal_is_identity_map(comm):
    r"""Extracting the block-diagonal of an already block-diagonal matrix
    should return the same matrix."""
    n = 3
    nblocks = 4
    nN = n * nblocks

    rng = np.random.default_rng(99)
    B_np = np.zeros((nN, nN), dtype=np.complex128)
    for k in range(nblocks):
        block = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        B_np[k * n : (k + 1) * n, k * n : (k + 1) * n] = block
    B_np = comm.tompi4py().bcast(B_np, root=0)

    B_petsc = _numpy_to_petsc(comm, B_np)
    B2_petsc = extract_block_diagonal(B_petsc, nblocks)

    # Verify: B2 x == B x for random x
    x, x_np = pytest_utils.generate_random_vector(comm, nN)
    y1 = x.duplicate()
    y2 = x.duplicate()
    B_petsc.mult(x, y1)
    B2_petsc.mult(x, y2)

    y1_seq = res4py.distributed_to_sequential_vector(y1)
    y2_seq = res4py.distributed_to_sequential_vector(y2)
    error = np.linalg.norm(
        y1_seq.getArray() - y2_seq.getArray()
    ) / np.linalg.norm(y1_seq.getArray())

    y1_seq.destroy()
    y2_seq.destroy()
    x.destroy()
    y1.destroy()
    y2.destroy()
    B_petsc.destroy()
    B2_petsc.destroy()
    assert error < 1e-10, f"block-diagonal idempotency error: {error:.2e}"
