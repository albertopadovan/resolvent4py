import numpy as np
import pytest
import scipy as sp
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.utils.matrix import (
    _ownership_slice,
    add_identity_block,
    add_matrix_block,
    create_aij_matrix,
    extract_block_banded,
    left_diagonal_solve,
    matrix_diagonal_array,
    matrix_subblock,
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
    ksp = res4py.create_direct_solver(Apetsc)

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


def test_extract_block_banded_diagonal(comm):
    r"""extract_block_banded with n_off_diags=0 extracts the correct
    diagonal blocks from a random block-structured matrix."""
    n = 4
    nblocks = 5
    nN = n * nblocks

    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((nN, nN)) + 1j * rng.standard_normal((nN, nN))
    A_np = comm.tompi4py().bcast(A_np, root=0)

    A_petsc = pytest_utils.numpy_to_petsc(comm, A_np)
    B_petsc = extract_block_banded(A_petsc, nblocks, 0)

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
    assert error < 1e-10, f"extract_block_banded(.,0) error: {error:.2e}"


def test_extract_block_banded_bandwidths(comm):
    r"""extract_block_banded with n_off_diags=k keeps exactly the blocks
    (i, j) with |i - j| <= k (tridiagonal for k=1, pentadiagonal for
    k=2, etc.)."""
    n = 3
    nblocks = 6
    nN = n * nblocks

    rng = np.random.default_rng(7)
    A_np = rng.standard_normal((nN, nN)) + 1j * rng.standard_normal((nN, nN))
    A_np = comm.tompi4py().bcast(A_np, root=0)
    A_petsc = pytest_utils.numpy_to_petsc(comm, A_np)

    for n_off_diags in (1, 2, 3):
        B_petsc = extract_block_banded(A_petsc, nblocks, n_off_diags)

        # Expected: keep blocks (i, j) with |i - j| <= n_off_diags
        B_expected = np.zeros_like(A_np)
        for i in range(nblocks):
            for j in range(nblocks):
                if abs(i - j) <= n_off_diags:
                    B_expected[
                        i * n : (i + 1) * n, j * n : (j + 1) * n
                    ] = A_np[i * n : (i + 1) * n, j * n : (j + 1) * n]

        x, x_np = pytest_utils.generate_random_vector(comm, nN)
        y = x.duplicate()
        B_petsc.mult(x, y)
        y_seq = res4py.distributed_to_sequential_vector(y)
        y_expected = B_expected @ x_np
        error = np.linalg.norm(
            y_seq.getArray() - y_expected
        ) / np.linalg.norm(y_expected)

        y_seq.destroy()
        x.destroy()
        y.destroy()
        B_petsc.destroy()
        assert error < 1e-10, (
            f"extract_block_banded(.,{n_off_diags}) error: {error:.2e}"
        )

    A_petsc.destroy()


def test_extract_block_banded_of_block_banded_is_identity_map(comm):
    r"""Extracting the block-banded part (n_off_diags=k) of a matrix that
    is already block-banded with bandwidth k should return the same
    matrix.  Covers block-diagonal (k=0), block-tridiagonal (k=1) and
    block-pentadiagonal (k=2)."""
    n = 3
    nblocks = 6
    nN = n * nblocks

    for n_off_diags in (0, 1, 2):
        rng = np.random.default_rng(99 + n_off_diags)
        # Build a matrix that is already block-banded with bandwidth k.
        B_np = np.zeros((nN, nN), dtype=np.complex128)
        for i in range(nblocks):
            for j in range(nblocks):
                if abs(i - j) <= n_off_diags:
                    block = rng.standard_normal((n, n)) + 1j * rng.standard_normal(
                        (n, n)
                    )
                    B_np[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
        B_np = comm.tompi4py().bcast(B_np, root=0)

        B_petsc = pytest_utils.numpy_to_petsc(comm, B_np)
        B2_petsc = extract_block_banded(B_petsc, nblocks, n_off_diags)

        # Verify: B2 x == B x for random x
        x, _ = pytest_utils.generate_random_vector(comm, nN)
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
        assert error < 1e-10, (
            f"block-banded idempotency error (n_off_diags={n_off_diags}): "
            f"{error:.2e}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers introduced alongside the OWNS resolvent analysis
# ──────────────────────────────────────────────────────────────────────────────


def _reference_matrix(comm, nrows, ncols, seed=17):
    r"""Deterministic dense reference, identical on every rank, plus its
    distributed PETSc counterpart."""
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((nrows, ncols)) + 1j * rng.standard_normal(
        (nrows, ncols)
    )
    A_np = comm.tompi4py().bcast(A_np, root=0)
    return A_np, pytest_utils.numpy_to_petsc(comm, A_np)


def _dense_from_petsc(comm, M, nrows, ncols):
    r"""Materialise a distributed PETSc matrix as a dense numpy array by
    applying it to the columns of the identity."""
    cols = []
    for j in range(ncols):
        e = M.createVecRight()
        if e.getOwnershipRange()[0] <= j < e.getOwnershipRange()[1]:
            e.setValue(j, 1.0)
        e.assemble()
        y = M.createVecLeft()
        M.mult(e, y)
        y_seq = res4py.distributed_to_sequential_vector(y)
        cols.append(y_seq.getArray().copy())
        y_seq.destroy()
        y.destroy()
        e.destroy()
    return np.column_stack(cols)


def test_ownership_slice_partitions_exactly():
    r"""_ownership_slice must tile [0, size) exactly: contiguous, no gaps,
    no overlaps, and balanced to within one element -- including the
    uneven cases that a 2-rank test run never exercises."""
    for size in range(0, 18):
        for n_ranks in range(1, 7):
            slices = [_ownership_slice(size, r, n_ranks) for r in range(n_ranks)]

            assert slices[0][0] == 0, f"size={size} n={n_ranks}: gap at start"
            assert slices[-1][1] == size, f"size={size} n={n_ranks}: short end"
            for r in range(1, n_ranks):
                assert slices[r][0] == slices[r - 1][1], (
                    f"size={size} n={n_ranks}: discontinuity at rank {r}"
                )

            counts = [e - s for s, e in slices]
            assert all(c >= 0 for c in counts), (
                f"size={size} n={n_ranks}: negative count {counts}"
            )
            assert sum(counts) == size, (
                f"size={size} n={n_ranks}: counts {counts} sum != {size}"
            )
            assert max(counts) - min(counts) <= 1, (
                f"size={size} n={n_ranks}: unbalanced {counts}"
            )


def test_create_aij_matrix(comm, square_matrix_size):
    r"""create_aij_matrix returns an assembled AIJ matrix of the requested
    global shape."""
    N = square_matrix_size[0]
    M = create_aij_matrix(comm, N, N - 3)
    M.assemble()

    sizes = M.getSizes()
    assert sizes[0][-1] == N
    assert sizes[-1][-1] == N - 3
    assert M.getType().startswith("seqaij") or M.getType().startswith("mpiaij")
    M.destroy()


def test_matrix_diagonal_array(comm, square_matrix_size):
    r"""matrix_diagonal_array returns the global diagonal on every rank."""
    N = square_matrix_size[0]
    A_np, A = _reference_matrix(comm, N, N)

    diag = matrix_diagonal_array(A)

    assert diag.shape == (N,)
    error = np.linalg.norm(diag - np.diag(A_np)) / np.linalg.norm(np.diag(A_np))
    everyones = comm.tompi4py().allgather(diag)
    consistent = all(np.array_equal(d, everyones[0]) for d in everyones)

    A.destroy()
    assert error < 1e-12, f"matrix_diagonal_array error: {error:.2e}"
    assert consistent, "ranks disagree on the diagonal"


def test_matrix_subblock(comm):
    r"""matrix_subblock must reproduce numpy fancy indexing A[np.ix_(rows,
    cols)].  Rows and columns are deliberately non-contiguous and unsorted,
    and their counts do not divide evenly across typical rank counts."""
    N = 12
    A_np, A = _reference_matrix(comm, N, N)
    rows = [0, 3, 4, 7, 11]
    cols = [1, 2, 6, 9]

    sub = matrix_subblock(A, rows, cols)

    assert sub.getSizes()[0][-1] == len(rows)
    assert sub.getSizes()[-1][-1] == len(cols)

    got = _dense_from_petsc(comm, sub, len(rows), len(cols))
    expected = A_np[np.ix_(rows, cols)]
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)

    sub.destroy()
    A.destroy()
    assert error < 1e-12, f"matrix_subblock error: {error:.2e}"


def test_add_matrix_block(comm):
    r"""add_matrix_block scatters a scaled block into a larger matrix at the
    requested offset, leaving everything else untouched."""
    n, N = 4, 10
    B_np, B = _reference_matrix(comm, n, n, seed=5)
    scale = 2.0 - 1.5j
    row_offset, col_offset = 3, 5

    target = create_aij_matrix(comm, N, N, nnz=N)
    add_matrix_block(target, row_offset, col_offset, scale, B)
    target.assemble()

    expected = np.zeros((N, N), dtype=complex)
    expected[row_offset : row_offset + n, col_offset : col_offset + n] = (
        scale * B_np
    )
    got = _dense_from_petsc(comm, target, N, N)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)

    target.destroy()
    B.destroy()
    assert error < 1e-12, f"add_matrix_block error: {error:.2e}"


def test_add_matrix_block_accumulates(comm):
    r"""Two calls at the same offset must add, not overwrite -- the function
    uses ADD_VALUES and callers rely on that to build block systems."""
    n, N = 3, 8
    B_np, B = _reference_matrix(comm, n, n, seed=9)

    target = create_aij_matrix(comm, N, N, nnz=N)
    add_matrix_block(target, 0, 0, 1.0, B)
    add_matrix_block(target, 0, 0, 2.0, B)
    target.assemble()

    expected = np.zeros((N, N), dtype=complex)
    expected[:n, :n] = 3.0 * B_np
    got = _dense_from_petsc(comm, target, N, N)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)

    target.destroy()
    B.destroy()
    assert error < 1e-12, f"add_matrix_block did not accumulate: {error:.2e}"


def test_add_identity_block(comm):
    r"""add_identity_block places a scaled identity at an arbitrary offset,
    including an off-diagonal one."""
    N, size = 10, 4
    scale = -0.5 + 2.0j
    row_offset, col_offset = 2, 6

    target = create_aij_matrix(comm, N, N, nnz=N)
    add_identity_block(target, row_offset, col_offset, size, scale)
    target.assemble()

    expected = np.zeros((N, N), dtype=complex)
    for k in range(size):
        expected[row_offset + k, col_offset + k] = scale
    got = _dense_from_petsc(comm, target, N, N)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)

    target.destroy()
    assert error < 1e-12, f"add_identity_block error: {error:.2e}"


def test_left_diagonal_solve(comm):
    r"""left_diagonal_solve divides row i by diagonal[i], i.e. computes
    diag(d)^-1 @ A."""
    N = 9
    A_np, A = _reference_matrix(comm, N, N, seed=3)
    rng = np.random.default_rng(11)
    d = rng.standard_normal(N) + 1j * rng.standard_normal(N) + 2.0
    d = comm.tompi4py().bcast(d, root=0)

    out = left_diagonal_solve(d, A)

    expected = A_np / d[:, None]
    got = _dense_from_petsc(comm, out, N, N)
    error = np.linalg.norm(got - expected) / np.linalg.norm(expected)

    out.destroy()
    A.destroy()
    assert error < 1e-12, f"left_diagonal_solve error: {error:.2e}"


def test_left_diagonal_solve_raises_on_size_mismatch(comm):
    r"""A diagonal whose length does not match the row count is a caller
    error and must be rejected rather than silently truncated."""
    N = 6
    _, A = _reference_matrix(comm, N, N, seed=4)
    bad = np.ones(N - 1, dtype=complex)

    with pytest.raises(ValueError):
        left_diagonal_solve(bad, A)

    A.destroy()


def test_left_diagonal_solve_raises_on_zero_diagonal(comm):
    r"""A zero diagonal entry means the system is singular; it must raise
    instead of producing infs."""
    N = 6
    _, A = _reference_matrix(comm, N, N, seed=6)
    d = np.ones(N, dtype=complex)
    d[N // 2] = 0.0

    with pytest.raises(ValueError):
        left_diagonal_solve(d, A)

    A.destroy()
