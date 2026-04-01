import numpy as np
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


def test_gmres_bjacobi_solver_converges(comm, square_random_matrix):
    r"""Test that GMRES + block-Jacobi converges with positive reason"""
    Apetsc, _ = square_random_matrix
    nblocks = comm.getSize()
    ksp = res4py.create_gmres_bjacobi_solver(Apetsc, nblocks)
    # Should not raise (checks convergence reason >= 0)
    res4py.check_gmres_bjacobi_solver(Apetsc, ksp)
    ksp.destroy()


def test_gmres_bjacobi_solver_residual(comm, square_random_matrix):
    r"""Test that the GMRES solution satisfies ||b - Ax|| / ||b|| < tol"""
    Apetsc, _ = square_random_matrix
    N = Apetsc.getSizes()[0][-1]
    nblocks = comm.getSize()
    ksp = res4py.create_gmres_bjacobi_solver(Apetsc, nblocks)

    Nl = res4py.compute_local_size(N)
    b = res4py.generate_random_petsc_vector((Nl, N))
    x = b.duplicate()
    ksp.solve(b, x)

    # Compute residual r = b - A*x
    r = b.duplicate()
    Apetsc.mult(x, r)
    r.aypx(-1.0, b)
    residual = r.norm() / b.norm()

    r.destroy()
    b.destroy()
    x.destroy()
    ksp.destroy()
    assert residual < 1e-8


def test_gmres_bjacobi_solver_multiple_rhs_residual(
    comm, square_random_matrix
):
    r"""Test GMRES + block-Jacobi residual is small for multiple RHS"""
    Apetsc, _ = square_random_matrix
    N = Apetsc.getSizes()[0][-1]
    nblocks = comm.getSize()
    ksp = res4py.create_gmres_bjacobi_solver(Apetsc, nblocks)

    Nl = res4py.compute_local_size(N)
    residuals = []
    for _ in range(3):
        b = res4py.generate_random_petsc_vector((Nl, N))
        x = b.duplicate()
        ksp.solve(b, x)
        r = b.duplicate()
        Apetsc.mult(x, r)
        r.aypx(-1.0, b)
        residuals.append(r.norm() / b.norm())
        r.destroy()
        b.destroy()
        x.destroy()

    ksp.destroy()
    assert max(residuals) < 1e-8


def test_gmres_bjacobi_solver_custom_tolerances(comm, square_random_matrix):
    r"""Test GMRES + block-Jacobi with tighter tolerances gives smaller
    residual"""
    Apetsc, _ = square_random_matrix
    N = Apetsc.getSizes()[0][-1]
    nblocks = comm.getSize()
    Nl = res4py.compute_local_size(N)
    b = res4py.generate_random_petsc_vector((Nl, N))

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
    # Tighter tolerance should give smaller (or equal) residual
    assert residuals[1] <= residuals[0] + 1e-15
