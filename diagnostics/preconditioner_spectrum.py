r"""
Preconditioner Spectrum Diagnostic
===================================

Computes and plots the eigenvalues of:

- The harmonic-balance operator :math:`T`
- The preconditioned operator :math:`S^{-1} T`

where :math:`S` is a preconditioner.

For an effective preconditioner, the eigenvalues of :math:`S^{-1} T`
should be clustered near 1.

Usage::

    mpiexec -n <nprocs> python preconditioner_spectrum.py \
        --T <LinearOperator> --A <PETSc.Mat> --M <PETSc.Mat> \
        --omega <float> --s <complex> --krylov_dim <int> --n_evals <int>

This script is meant to be imported and called from a user script
that already has T, A, M assembled.  See :func:`plot_spectrum` below.
"""

import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

import resolvent4py as res4py
from resolvent4py.linear_operators import (
    ProductLinearOperator,
    LinearOperator,
)


def compute_spectrum(
    T: LinearOperator,
    S: LinearOperator,
    krylov_dim: int,
    n_evals: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r"""
    Compute eigenvalues of :math:`T` and :math:`S^{-1} T`.

    :param T: harmonic-balance linear operator
    :type T: LinearOperator
    :param S: preconditioner linear operator (must have ``.solve()``)
    :type S: LinearOperator
    :param krylov_dim: Krylov subspace dimension for Arnoldi
    :type krylov_dim: int
    :param n_evals: number of eigenvalues to compute
    :type n_evals: int

    :return: ``(evals_T, evals_SinvT, cond_T, cond_SinvT)`` —
        eigenvalues and condition numbers of :math:`T` and
        :math:`S^{-1} T`
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float, float]
    """
    comm = T.get_comm()
    nblocks = T.get_nblocks()

    # Eigenvalues of T (via T^{-1}, mapped back by lambda -> -1/lambda)
    res4py.petscprint(comm, "Computing eigenvalues of T...")
    evals_T, _ = res4py.linalg.eig(
        T,
        T.solve,
        krylov_dim,
        n_evals,
        process_evals=lambda x: -1.0 / x,
    )
    evals_T = np.diag(evals_T)

    # Eigenvalues of S^{-1} T
    res4py.petscprint(comm, "Computing eigenvalues of S^{-1} T...")
    SinvT = ProductLinearOperator(
        [S, T],
        [S.solve, T.apply],
        nblocks,
    )
    evals_SinvT, _ = res4py.linalg.eig(
        SinvT,
        SinvT.apply,
        krylov_dim,
        n_evals,
        process_evals=lambda x: x,
    )
    evals_SinvT = np.diag(evals_SinvT)

    # Condition numbers: ratio of largest to smallest eigenvalue magnitude
    cond_T = np.max(np.abs(evals_T)) / np.min(np.abs(evals_T))
    cond_SinvT = np.max(np.abs(evals_SinvT)) / np.min(np.abs(evals_SinvT))
    res4py.petscprint(comm, f"cond(T)      = {cond_T:.4e}")
    res4py.petscprint(comm, f"cond(S^{{-1}}T) = {cond_SinvT:.4e}")

    return evals_T, evals_SinvT, cond_T, cond_SinvT


def gmres_convergence(
    T: LinearOperator,
    S: LinearOperator,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    maxiter: int = 500,
    restart: int = None,
) -> tuple[list[float], list[float]]:
    r"""
    Solve :math:`T x = b` with GMRES (unpreconditioned and preconditioned
    by :math:`S`) and return the residual histories.

    :param T: harmonic-balance linear operator (must have ``T.A``)
    :type T: LinearOperator
    :param S: preconditioner linear operator (must have ``.solve()``)
    :type S: LinearOperator
    :param rtol: relative tolerance for GMRES
    :type rtol: float
    :param atol: absolute tolerance for GMRES
    :type atol: float
    :param maxiter: maximum number of GMRES iterations
    :type maxiter: int
    :param restart: GMRES restart parameter.  If ``None``, defaults to
        ``nN`` (full GMRES, no restart).
    :type restart: Optional[int]

    :return: ``(residuals_no_pc, residuals_pc)`` — residual norms at
        each iteration for unpreconditioned and preconditioned GMRES
    :rtype: tuple[list[float], list[float]]
    """
    from resolvent4py.linear_operators.petsc_python import _PCContext

    comm = T.get_comm()
    sizes = T.A.getSizes()[0]
    nN = sizes[-1]
    gmres_restart = restart if restart is not None else nN

    res4py.petscprint(
        comm, f"GMRES restart = {gmres_restart}, maxiter = {maxiter}"
    )

    # Random RHS
    b = res4py.generate_random_petsc_vector(sizes)

    # --- Unpreconditioned GMRES ---
    res4py.petscprint(comm, "Running GMRES (no preconditioner)...")
    residuals_no_pc = []

    def monitor_no_pc(ksp, its, rnorm):
        residuals_no_pc.append(rnorm)

    ksp_no_pc = PETSc.KSP().create(comm=comm)
    ksp_no_pc.setOperators(T.A)
    ksp_no_pc.setType("gmres")
    ksp_no_pc.setGMRESRestart(gmres_restart)
    ksp_no_pc.setTolerances(rtol=rtol, atol=atol, max_it=maxiter)
    ksp_no_pc.setMonitor(monitor_no_pc)
    pc_no = ksp_no_pc.getPC()
    pc_no.setType("none")
    ksp_no_pc.setUp()

    x_no_pc = b.duplicate()
    ksp_no_pc.solve(b, x_no_pc)
    res4py.petscprint(
        comm,
        f"  converged in {ksp_no_pc.getIterationNumber()} iterations, "
        f"reason = {ksp_no_pc.getConvergedReason()}",
    )

    # --- Preconditioned GMRES (shell PC) ---
    res4py.petscprint(comm, "Running GMRES (preconditioned)...")

    residuals_pc = []

    def monitor_pc(ksp, its, rnorm):
        residuals_pc.append(rnorm)

    ksp_pc = PETSc.KSP().create(comm=comm)
    ksp_pc.setOperators(T.A)
    ksp_pc.setType("gmres")
    ksp_pc.setGMRESRestart(gmres_restart)
    ksp_pc.setTolerances(rtol=rtol, atol=atol, max_it=maxiter)
    ksp_pc.setMonitor(monitor_pc)
    pc = ksp_pc.getPC()
    pc.setType("python")
    pc.setPythonContext(_PCContext(S))
    ksp_pc.setUp()

    x_pc = b.duplicate()
    ksp_pc.solve(b, x_pc)
    res4py.petscprint(
        comm,
        f"  converged in {ksp_pc.getIterationNumber()} iterations, "
        f"reason = {ksp_pc.getConvergedReason()}",
    )

    # Cleanup
    b.destroy()
    x_no_pc.destroy()
    x_pc.destroy()
    ksp_no_pc.destroy()
    ksp_pc.destroy()

    return residuals_no_pc, residuals_pc


def plot_convergence(
    residuals_no_pc: list[float],
    residuals_pc: list[float],
    save_path: str = None,
) -> None:
    r"""
    Plot GMRES residual convergence curves.

    :param residuals_no_pc: residual history without preconditioner
    :type residuals_no_pc: list[float]
    :param residuals_pc: residual history with preconditioner
    :type residuals_pc: list[float]
    :param save_path: if provided, save figure to this path
    :type save_path: Optional[str]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(
        range(len(residuals_no_pc)),
        residuals_no_pc,
        "k-o",
        markersize=3,
        label="No preconditioner",
    )
    ax.semilogy(
        range(len(residuals_pc)),
        residuals_pc,
        "r-s",
        markersize=3,
        label="Superoptimal block-circulant",
    )
    ax.set_xlabel("GMRES iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title("GMRES convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_spectrum(
    evals_T: np.ndarray,
    evals_SinvT: np.ndarray,
    cond_T: float = None,
    cond_SinvT: float = None,
    save_path: str = None,
) -> None:
    r"""
    Plot eigenvalues of :math:`T` and :math:`S^{-1} T` on the complex
    plane.

    :param evals_T: eigenvalues of :math:`T`
    :type evals_T: numpy.ndarray
    :param evals_SinvT: eigenvalues of :math:`S^{-1} T`
    :type evals_SinvT: numpy.ndarray
    :param cond_T: condition number of :math:`T`
    :type cond_T: Optional[float]
    :param cond_SinvT: condition number of :math:`S^{-1} T`
    :type cond_SinvT: Optional[float]
    :param save_path: if provided, save figure to this path instead of
        showing it
    :type save_path: Optional[str]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Eigenvalues of T
    ax = axes[0]
    ax.plot(evals_T.real, evals_T.imag, "ko", markersize=4)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    title_T = r"Eigenvalues of $T$"
    if cond_T is not None:
        title_T += f"\n$\\kappa = {cond_T:.2e}$"
    ax.set_title(title_T)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Eigenvalues of S^{-1} T
    ax = axes[1]
    ax.plot(evals_SinvT.real, evals_SinvT.imag, "ro", markersize=4)
    ax.plot(1.0, 0.0, "k+", markersize=12, markeredgewidth=2)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    title_SinvT = r"Eigenvalues of $S^{-1}T$"
    if cond_SinvT is not None:
        title_SinvT += f"\n$\\kappa = {cond_SinvT:.2e}$"
    ax.set_title(title_SinvT)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
