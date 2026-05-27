import os
import tempfile

import numpy as np
import scipy as sp
import resolvent4py as res4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
from .. import pytest_utils


def _shared_tmpdir(comm):
    r"""Create a temp directory on rank 0 and broadcast the path to all."""
    tmpdir = None
    if comm.getRank() == 0:
        tmpdir = tempfile.mkdtemp()
    tmpdir = comm.tompi4py().bcast(tmpdir, root=0)
    return tmpdir


def test_vector_write_read_roundtrip(comm, square_matrix_size):
    r"""Test write_to_file -> read_vector roundtrip for PETSc.Vec"""
    N = square_matrix_size[0]
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "vec.dat")

    x, xpython = pytest_utils.generate_random_vector(comm, N)
    res4py.write_to_file(filepath, x)

    Nl = res4py.compute_local_size(N)
    y = res4py.read_vector(filepath, (Nl, N))

    xs = res4py.distributed_to_sequential_vector(x)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(xs.getArray() - ys.getArray())

    xs.destroy()
    ys.destroy()
    x.destroy()
    y.destroy()
    assert error < 1e-14


def test_vector_write_read_complex(comm, square_matrix_size):
    r"""Test roundtrip for a complex-valued vector"""
    N = square_matrix_size[0]
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "vec_complex.dat")

    x, xpython = pytest_utils.generate_random_vector(comm, N, complex=True)
    res4py.write_to_file(filepath, x)

    Nl = res4py.compute_local_size(N)
    y = res4py.read_vector(filepath, (Nl, N))

    xs = res4py.distributed_to_sequential_vector(x)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(xs.getArray() - ys.getArray())

    xs.destroy()
    ys.destroy()
    x.destroy()
    y.destroy()
    assert error < 1e-14


def test_dense_matrix_write_read_roundtrip(comm, square_matrix_size):
    r"""Test write_to_file -> read_dense_matrix roundtrip"""
    N = square_matrix_size[0]
    s = 5
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "dense_mat.dat")

    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Xm = X.getMat()
    res4py.write_to_file(filepath, Xm)
    X.restoreMat(Xm)

    Nl = res4py.compute_local_size(N)
    Ncl = res4py.compute_local_size(s)
    Y = res4py.read_dense_matrix(filepath, ((Nl, N), (Ncl, s)))

    Yseq = res4py.distributed_to_sequential_matrix(Y)
    error = np.linalg.norm(Yseq.getDenseArray() - Xpython) / np.linalg.norm(
        Xpython
    )

    Yseq.destroy()
    Y.destroy()
    X.destroy()
    assert error < 1e-14


def test_bv_write_read_roundtrip(comm, square_matrix_size):
    r"""Test write_to_file -> read_bv roundtrip for SLEPc.BV"""
    N = square_matrix_size[0]
    s = 5
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "bv.dat")

    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    res4py.write_to_file(filepath, X)

    Nl = res4py.compute_local_size(N)
    Y = res4py.read_bv(filepath, ((Nl, N), s))

    Ym = Y.getMat()
    Yseq = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    error = np.linalg.norm(Yseq.getDenseArray() - Xpython) / np.linalg.norm(
        Xpython
    )

    Yseq.destroy()
    Y.destroy()
    X.destroy()
    assert error < 1e-14


def test_coo_matrix_write_read_roundtrip(comm, square_matrix_size):
    r"""Test writing COO vectors and reading back as sparse matrix via
    read_coo_matrix"""
    N = square_matrix_size[0]
    tmpdir = _shared_tmpdir(comm)

    # Generate a random sparse matrix and get its COO arrays
    Nl = res4py.compute_local_size(N)
    sizes = ((Nl, N), (Nl, N))
    Apetsc = res4py.generate_random_petsc_sparse_matrix(
        sizes, int(0.3 * N * N)
    )

    # Convert to dense for reference
    Ad = Apetsc.copy()
    Ad.convert(PETSc.Mat.Type.DENSE)
    Ad_seq = res4py.distributed_to_sequential_matrix(Ad)
    Apython = Ad_seq.getDenseArray().copy()
    Ad_seq.destroy()
    Ad.destroy()

    # Extract COO from the sparse matrix: gather to sequential, then get
    # row/col/val arrays
    # We need to write the COO arrays as PETSc vectors
    # Simplest approach: create a small known sparse matrix from COO
    rank = comm.getRank()
    np.random.seed(42 + rank)
    nnz_local = 30
    rows_arr = np.random.randint(0, N, size=nnz_local).astype(np.float64)
    cols_arr = np.random.randint(0, N, size=nnz_local).astype(np.float64)
    vals_arr = np.random.randn(nnz_local) + 1j * np.random.randn(nnz_local)

    # Write COO arrays as PETSc vectors
    rows_vec = PETSc.Vec().createWithArray(
        rows_arr.copy() + 0j, comm=PETSc.COMM_WORLD
    )
    cols_vec = PETSc.Vec().createWithArray(
        cols_arr.copy() + 0j, comm=PETSc.COMM_WORLD
    )
    vals_vec = PETSc.Vec().createWithArray(
        vals_arr.copy(), comm=PETSc.COMM_WORLD
    )

    frows = os.path.join(tmpdir, "rows.dat")
    fcols = os.path.join(tmpdir, "cols.dat")
    fvals = os.path.join(tmpdir, "vals.dat")
    res4py.write_to_file(frows, rows_vec)
    res4py.write_to_file(fcols, cols_vec)
    res4py.write_to_file(fvals, vals_vec)
    rows_vec.destroy()
    cols_vec.destroy()
    vals_vec.destroy()

    # Read back as sparse matrix
    M = res4py.read_coo_matrix((frows, fcols, fvals), sizes)

    # Build python reference from the same COO data (gathered across ranks)
    all_rows = np.concatenate(comm.tompi4py().allgather(rows_arr.astype(int)))
    all_cols = np.concatenate(comm.tompi4py().allgather(cols_arr.astype(int)))
    all_vals = np.concatenate(comm.tompi4py().allgather(vals_arr))
    # Remove near-zeros (same as read_coo_matrix does)
    mask = np.abs(all_vals) > 1e-16
    all_rows = all_rows[mask]
    all_cols = all_cols[mask]
    all_vals = all_vals[mask]
    Mpython = sp.sparse.coo_matrix(
        (all_vals, (all_rows, all_cols)), shape=(N, N)
    ).toarray()

    # Verify via matvec
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = x.duplicate()
    M.mult(x, y)
    ys = res4py.distributed_to_sequential_vector(y)
    expected = Mpython.dot(xpython)
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(expected)

    ys.destroy()
    x.destroy()
    y.destroy()
    M.destroy()
    Apetsc.destroy()
    assert error < 1e-10


def test_vector_read_without_sizes(comm, square_matrix_size):
    r"""Test read_vector with sizes=None (PETSc determines sizes)"""
    N = square_matrix_size[0]
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "vec_nosize.dat")

    x, _ = pytest_utils.generate_random_vector(comm, N)
    res4py.write_to_file(filepath, x)

    y = res4py.read_vector(filepath)
    assert y.getSize() == N

    xs = res4py.distributed_to_sequential_vector(x)
    ys = res4py.distributed_to_sequential_vector(y)
    error = np.linalg.norm(xs.getArray() - ys.getArray())
    xs.destroy()
    ys.destroy()
    x.destroy()
    y.destroy()
    assert error < 1e-14


def test_harmonic_balanced_vector_roundtrip(comm, square_matrix_size):
    r"""Test write/read roundtrip for harmonic balanced vectors"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    tmpdir = _shared_tmpdir(comm)
    nfreqs = 3  # v_0, v_1, v_2

    # Create and write Fourier coefficient vectors
    vecs_python = []
    filenames = []
    for k in range(nfreqs):
        v, vpython = pytest_utils.generate_random_vector(comm, N, complex=True)
        filepath = os.path.join(tmpdir, f"v_{k}.dat")
        res4py.write_to_file(filepath, v)
        filenames.append(filepath)
        vecs_python.append(vpython)
        v.destroy()

    # Read back as harmonic balanced vector (real_bflow=True means
    # filenames contain v_0, v_1, v_2 and negative freqs are conjugates)
    nblocks = 2 * nfreqs - 1  # v_{-2}, v_{-1}, v_0, v_1, v_2
    full_N = nblocks * N
    full_Nl = res4py.compute_local_size(full_N)
    Vec = res4py.read_harmonic_balanced_vector(
        filenames,
        real_bflow=True,
        block_sizes=(Nl, N),
        full_sizes=(full_Nl, full_N),
    )

    # Build expected: [conj(v_2), conj(v_1), v_0, v_1, v_2]
    expected = np.concatenate(
        [
            vecs_python[2].conj(),
            vecs_python[1].conj(),
            vecs_python[0],
            vecs_python[1],
            vecs_python[2],
        ]
    )

    Vs = res4py.distributed_to_sequential_vector(Vec)
    error = np.linalg.norm(Vs.getArray() - expected) / np.linalg.norm(expected)
    Vs.destroy()
    Vec.destroy()
    assert error < 1e-14


def _write_coo_to_files(comm, tmpdir, name, A_python):
    r"""Write the COO representation of a dense numpy ``A_python`` to a
    PETSc-readable triple of (rows, cols, vals) files.  All COO data is
    placed on rank 0; other ranks contribute empty local sections.
    """
    rank = comm.getRank()
    A_coo = sp.sparse.coo_matrix(A_python)
    if rank == 0:
        rows = A_coo.row.astype(np.float64)
        cols = A_coo.col.astype(np.float64)
        vals = A_coo.data.astype(np.complex128)
    else:
        rows = np.empty(0, dtype=np.float64)
        cols = np.empty(0, dtype=np.float64)
        vals = np.empty(0, dtype=np.complex128)

    rows_vec = PETSc.Vec().createWithArray(
        rows.astype(np.complex128).copy(),
        comm=PETSc.COMM_WORLD,
    )
    cols_vec = PETSc.Vec().createWithArray(
        cols.astype(np.complex128).copy(),
        comm=PETSc.COMM_WORLD,
    )
    vals_vec = PETSc.Vec().createWithArray(
        vals.copy(),
        comm=PETSc.COMM_WORLD,
    )

    paths = (
        os.path.join(tmpdir, f"rows_{name}.dat"),
        os.path.join(tmpdir, f"cols_{name}.dat"),
        os.path.join(tmpdir, f"vals_{name}.dat"),
    )
    res4py.write_to_file(paths[0], rows_vec)
    res4py.write_to_file(paths[1], cols_vec)
    res4py.write_to_file(paths[2], vals_vec)
    rows_vec.destroy()
    cols_vec.destroy()
    vals_vec.destroy()
    return paths


def _gather_petsc_to_dense(M):
    r"""Convert a parallel PETSc.Mat to a dense numpy array on every rank."""
    Md = M.copy()
    Md.convert(PETSc.Mat.Type.DENSE)
    Mseq = res4py.distributed_to_sequential_matrix(Md)
    arr = Mseq.getDenseArray().copy()
    Mseq.destroy()
    Md.destroy()
    return arr


def _build_block_toeplitz(A_blocks, nfp):
    r"""Build the ``(2*nfp+1) x (2*nfp+1)`` block-Toeplitz matrix from
    ``A_blocks = [A_{-nfb}, ..., A_0, ..., A_{nfb}]``.
    """
    nfb = (len(A_blocks) - 1) // 2
    n = A_blocks[0].shape[0]
    m = A_blocks[0].shape[1]
    nb = 2 * nfp + 1
    M = np.zeros((nb * n, nb * m), dtype=complex)
    for i in range(nb):
        for j in range(nb):
            k = i - j + nfb
            if 0 <= k < 2 * nfb + 1:
                M[i * n : (i + 1) * n, j * m : (j + 1) * m] = A_blocks[k]
    return M


def test_read_hb_matrix_block_diagonal(comm, square_matrix_size):
    r"""``len(filenames_lst) == 1`` and ``real_bflow=True`` ⇒ ``M`` is
    block diagonal with every diagonal block equal to ``A_0``."""
    n = 8
    nfp = 2
    nblocks = 2 * nfp + 1
    full_n = nblocks * n
    nl = res4py.compute_local_size(n)
    full_nl = res4py.compute_local_size(full_n)
    tmpdir = _shared_tmpdir(comm)

    # Build a known A_0 (same on every rank for reproducibility)
    rng = np.random.default_rng(42)
    A0 = (
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ).astype(np.complex128)

    paths = _write_coo_to_files(comm, tmpdir, "0", A0)

    M = res4py.read_harmonic_balanced_matrix(
        [paths],
        real_bflow=True,
        block_sizes=((nl, n), (nl, n)),
        full_sizes=((full_nl, full_n), (full_nl, full_n)),
    )

    Marr = _gather_petsc_to_dense(M)
    M.destroy()

    # On-diagonal blocks must equal A0; off-diagonal blocks must be zero.
    M_expected = np.zeros((full_n, full_n), dtype=complex)
    for b in range(nblocks):
        M_expected[b * n : (b + 1) * n, b * n : (b + 1) * n] = A0

    err = np.linalg.norm(Marr - M_expected) / np.linalg.norm(M_expected)
    assert err < 1e-12, f"block-diagonal mismatch, err = {err:.3e}"

    # Stronger check: every off-diagonal block exactly zero.
    for i in range(nblocks):
        for j in range(nblocks):
            if i == j:
                continue
            block_ij = Marr[i * n : (i + 1) * n, j * n : (j + 1) * n]
            assert np.max(np.abs(block_ij)) < 1e-14, (
                f"off-diagonal block ({i},{j}) is not zero "
                f"(max |.| = {np.max(np.abs(block_ij)):.3e})"
            )


def test_read_hb_matrix_real_bflow_toeplitz(comm, square_matrix_size):
    r"""``real_bflow=True`` with ``[A_0, A_1, A_2]`` should produce the
    block-Toeplitz matrix whose ``A_{-k}`` blocks are ``conj(A_k)``."""
    n = 6
    nfb = 2  # we supply A_0, A_1, A_2
    nfp = 3  # 7 blocks; need nfp >= nfb
    nblocks = 2 * nfp + 1
    full_n = nblocks * n
    nl = res4py.compute_local_size(n)
    full_nl = res4py.compute_local_size(full_n)
    tmpdir = _shared_tmpdir(comm)

    rng = np.random.default_rng(7)
    Ak_pos = [
        (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ).astype(np.complex128)
        for _ in range(nfb + 1)
    ]
    paths_lst = [
        _write_coo_to_files(comm, tmpdir, f"k{k}", Ak_pos[k])
        for k in range(nfb + 1)
    ]

    M = res4py.read_harmonic_balanced_matrix(
        paths_lst,
        real_bflow=True,
        block_sizes=((nl, n), (nl, n)),
        full_sizes=((full_nl, full_n), (full_nl, full_n)),
    )

    Marr = _gather_petsc_to_dense(M)
    M.destroy()

    # Expected: A_blocks ordered [conj(A_2), conj(A_1), A_0, A_1, A_2]
    A_blocks = [Ak_pos[k].conj() for k in range(nfb, 0, -1)] + Ak_pos
    M_expected = _build_block_toeplitz(A_blocks, nfp)

    err = np.linalg.norm(Marr - M_expected) / np.linalg.norm(M_expected)
    assert err < 1e-12, f"Toeplitz (real_bflow) mismatch, err = {err:.3e}"


def test_read_hb_matrix_two_sided_toeplitz(comm, square_matrix_size):
    r"""``real_bflow=False`` consumes the full two-sided list
    ``[A_{-nfb}, ..., A_0, ..., A_{nfb}]`` without imposing conjugacy."""
    n = 5
    nfb = 1  # supply A_{-1}, A_0, A_1
    nfp = 2  # 5 blocks
    nblocks = 2 * nfp + 1
    full_n = nblocks * n
    nl = res4py.compute_local_size(n)
    full_nl = res4py.compute_local_size(full_n)
    tmpdir = _shared_tmpdir(comm)

    rng = np.random.default_rng(11)
    A_blocks = [
        (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ).astype(np.complex128)
        for _ in range(2 * nfb + 1)
    ]
    paths_lst = [
        _write_coo_to_files(comm, tmpdir, f"i{i}", A_blocks[i])
        for i in range(2 * nfb + 1)
    ]

    M = res4py.read_harmonic_balanced_matrix(
        paths_lst,
        real_bflow=False,
        block_sizes=((nl, n), (nl, n)),
        full_sizes=((full_nl, full_n), (full_nl, full_n)),
    )

    Marr = _gather_petsc_to_dense(M)
    M.destroy()

    M_expected = _build_block_toeplitz(A_blocks, nfp)

    err = np.linalg.norm(Marr - M_expected) / np.linalg.norm(M_expected)
    assert err < 1e-12, f"Toeplitz (two-sided) mismatch, err = {err:.3e}"


def test_read_hb_matrix_too_few_blocks_raises(comm, square_matrix_size):
    r"""``nfp < nfb`` (more Fourier modes than blocks fit) must raise."""
    import pytest

    n = 4
    nfb = 3  # supply A_0..A_3 with real_bflow=True ⇒ nfb=3
    nfp = 1  # but only 3 blocks
    nblocks = 2 * nfp + 1
    full_n = nblocks * n
    nl = res4py.compute_local_size(n)
    full_nl = res4py.compute_local_size(full_n)
    tmpdir = _shared_tmpdir(comm)

    rng = np.random.default_rng(99)
    Ak_pos = [
        (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ).astype(np.complex128)
        for _ in range(nfb + 1)
    ]
    paths_lst = [
        _write_coo_to_files(comm, tmpdir, f"k{k}", Ak_pos[k])
        for k in range(nfb + 1)
    ]

    with pytest.raises(ValueError, match="must be larger"):
        res4py.read_harmonic_balanced_matrix(
            paths_lst,
            real_bflow=True,
            block_sizes=((nl, n), (nl, n)),
            full_sizes=((full_nl, full_n), (full_nl, full_n)),
        )


def test_write_to_file_creates_file(comm, square_matrix_size):
    r"""Test that write_to_file actually creates the file on disk"""
    N = square_matrix_size[0]
    tmpdir = _shared_tmpdir(comm)
    filepath = os.path.join(tmpdir, "exists_test.dat")

    x, _ = pytest_utils.generate_random_vector(comm, N)
    res4py.write_to_file(filepath, x)
    x.destroy()

    # Only check on rank 0 (file is shared)
    if comm.getRank() == 0:
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
