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
    error = np.linalg.norm(
        Yseq.getDenseArray() - Xpython
    ) / np.linalg.norm(Xpython)

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
    error = np.linalg.norm(
        Yseq.getDenseArray() - Xpython
    ) / np.linalg.norm(Xpython)

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
    all_rows = np.concatenate(
        comm.tompi4py().allgather(rows_arr.astype(int))
    )
    all_cols = np.concatenate(
        comm.tompi4py().allgather(cols_arr.astype(int))
    )
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
    error = np.linalg.norm(ys.getArray() - expected) / np.linalg.norm(
        expected
    )

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
    error = np.linalg.norm(Vs.getArray() - expected) / np.linalg.norm(
        expected
    )
    Vs.destroy()
    Vec.destroy()
    assert error < 1e-14


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
