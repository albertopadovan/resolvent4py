import numpy as np
import scipy as sp
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_mumps_solver_accuracy(comm, square_random_matrix):
    r"""Test that MUMPS solver produces accurate solutions"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    ksp = res4py.create_mumps_solver(Apetsc)

    b, bpython = pytest_utils.generate_random_vector(comm, N)
    x = b.duplicate()
    ksp.solve(b, x)

    xs = res4py.distributed_to_sequential_vector(x)
    xpython = np.linalg.solve(Apython, bpython)
    error = np.linalg.norm(xs.getArray() - xpython) / np.linalg.norm(xpython)
    xs.destroy()
    b.destroy()
    x.destroy()
    ksp.destroy()
    assert error < 1e-10


def test_check_lu_factorization_passes(comm, square_random_matrix):
    r"""Test that check_lu_factorization does not raise on valid factorization"""
    Apetsc, _ = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    # Should not raise
    res4py.check_lu_factorization(Apetsc, ksp)
    ksp.destroy()


def test_mumps_solver_multiple_rhs(comm, square_random_matrix):
    r"""Test MUMPS solver with multiple sequential right-hand sides"""
    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    ksp = res4py.create_mumps_solver(Apetsc)

    errors = []
    for _ in range(3):
        b, bpython = pytest_utils.generate_random_vector(comm, N)
        x = b.duplicate()
        ksp.solve(b, x)
        xs = res4py.distributed_to_sequential_vector(x)
        xpython = np.linalg.solve(Apython, bpython)
        errors.append(
            np.linalg.norm(xs.getArray() - xpython)
            / np.linalg.norm(xpython)
        )
        xs.destroy()
        b.destroy()
        x.destroy()

    ksp.destroy()
    assert max(errors) < 1e-10


def test_gmres_bjacobi_block_diagonal_one_iter(comm):
    r"""Build a general block-diagonal matrix (different invertible block
    on each diagonal slot) and verify that GMRES preconditioned by
    block-Jacobi with one bjacobi block per diagonal block converges in
    exactly one Krylov iteration.

    For block-Jacobi alignment, each MPI rank must own a whole number of
    diagonal blocks.  We pick

    * ``nblocks = 5`` when ``comm.size == 1``, and
    * ``nblocks = comm.size`` otherwise (one block per rank),

    so the test always runs.
    """
    nprocs = comm.getSize()
    nblocks = 5 if nprocs == 1 else nprocs
    n = 8                              # rows/cols per diagonal block
    full_n = nblocks * n
    full_nl = res4py.compute_local_size(full_n)

    # Build a general block-diagonal numpy reference: a different random
    # invertible block on every diagonal slot.  Identical on every rank
    # (same seed sequence) so the COO assembly below is consistent.
    M_np = np.zeros((full_n, full_n), dtype=np.complex128)
    for b in range(nblocks):
        rng = np.random.default_rng(2024 + b)
        Ab = (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ).astype(np.complex128)
        Ab += 5.0 * np.eye(n)          # diagonal shift → invertible
        M_np[b * n:(b + 1) * n, b * n:(b + 1) * n] = Ab

    # Place the full COO triple on rank 0 only and let convert_coo_to_csr
    # redistribute to the rank that owns each row.
    M_coo = sp.sparse.coo_matrix(M_np)
    if comm.getRank() == 0:
        rows = np.asarray(M_coo.row, dtype=PETSc.IntType)
        cols = np.asarray(M_coo.col, dtype=PETSc.IntType)
        vals = np.asarray(M_coo.data, dtype=PETSc.ScalarType)
    else:
        rows = np.empty(0, dtype=PETSc.IntType)
        cols = np.empty(0, dtype=PETSc.IntType)
        vals = np.empty(0, dtype=PETSc.ScalarType)

    sizes = ((full_nl, full_n), (full_nl, full_n))
    rows_ptr, cols_csr, vals_csr = res4py.convert_coo_to_csr(
        [rows, cols, vals], sizes,
    )
    M = PETSc.Mat().createAIJ(sizes, comm=PETSc.COMM_WORLD)
    M.setPreallocationCSR((rows_ptr, cols_csr))
    M.setValuesCSR(rows_ptr, cols_csr, vals_csr, True)
    M.assemble(False)

    ksp = res4py.create_gmres_bjacobi_solver(
        M, nblocks, rtol=1e-12, atol=1e-12,
    )

    b = res4py.generate_random_petsc_vector((full_nl, full_n))
    x = b.duplicate()
    ksp.solve(b, x)
    n_iters = ksp.getIterationNumber()

    # Residual ||b - M x|| / ||b||
    r = b.duplicate()
    M.mult(x, r)
    r.aypx(-1.0, b)
    residual = r.norm() / b.norm()

    r.destroy()
    b.destroy()
    x.destroy()
    ksp.destroy()
    M.destroy()

    assert n_iters == 1, (
        f"Expected GMRES to converge in 1 iteration on a block-diagonal "
        f"matrix preconditioned by matching block-Jacobi, got {n_iters}"
    )
    assert residual < 1e-10, f"residual = {residual:.3e}"


def test_gmres_bjacobi_solver_custom_tolerances(comm):
    r"""Tighter tolerances should give smaller (or equal) residual.

    Uses the same block-aligned ``nblocks`` choice as
    :func:`test_gmres_bjacobi_block_diagonal_one_iter` (``nblocks = 5``
    on 1 rank, ``nblocks = comm.size`` otherwise) so the bjacobi blocks
    line up with rank ownership.  The matrix here is *not* block
    diagonal — it's a generic random sparse matrix — so GMRES genuinely
    needs more than one step and the residual is sensitive to ``rtol``.
    """
    nprocs = comm.getSize()
    nblocks = 5 if nprocs == 1 else nprocs
    n = 8
    full_n = nblocks * n
    full_nl = res4py.compute_local_size(full_n)

    Apetsc, _ = pytest_utils.generate_random_matrix(comm, (full_n, full_n))
    b = res4py.generate_random_petsc_vector((full_nl, full_n))

    residuals = []
    for tol in [1e-8, 1e-12]:
        ksp = res4py.create_gmres_bjacobi_solver(
            Apetsc, nblocks, rtol=tol, atol=tol
        )
        x = b.duplicate()
        ksp.solve(b, x)
        r = b.duplicate()
        Apetsc.mult(x, r)
        r.aypx(-1.0, b)
        residuals.append(r.norm() / b.norm())
        r.destroy()
        x.destroy()
        ksp.destroy()

    b.destroy()
    Apetsc.destroy()
    # Tighter tolerance should give smaller (or equal) residual
    assert residuals[1] <= residuals[0] + 1e-15
