import scipy as sp
import numpy as np
import random
import resolvent4py as res4py
import resolvent4py.linalg.resolvent_analysis_time_stepping as res_ts
from petsc4py import PETSc
from slepc4py import SLEPc
from .. import pytest_utils


def _evaluate_forcing(Fhat, omegas, t, tf):
    tau = t if tf < 0 else tf - t
    q = np.exp(1j * omegas * tau)
    if np.min(omegas) == 0:
        c = 2 * np.ones(len(omegas))
        c[0] = 1.0
        q *= c
        f = Fhat.dot(q).real
    else:
        f = Fhat.dot(q)
    return f


def _evaluate_dynamics(t, x, A, Fhat, omegas, tf):
    return A.dot(x) + _evaluate_forcing(Fhat, omegas, t, tf)


def test_time_stepping_forced(comm, square_matrix_size):
    r"""Test time stepping function against scipy.integrate.solve_ivp"""

    np.random.seed(0)
    N, _ = square_matrix_size
    complex_lst = [False, True]
    adjoint_lst = [False, True]
    error_lst = []
    for complex in complex_lst:
        for adjoint in adjoint_lst:
            # Generate the operators
            Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
                comm, (N, N), complex
            )
            linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
            Apython = Apython.conj().T if adjoint else Apython

            # Generate frequency vector
            T = 2 * np.pi
            omega = 2 * np.pi / T
            omegas = (
                omega * np.arange(-2, 3, 1)
                if complex
                else omega * np.arange(0, 3, 1)
            )
            sz = (Apython.shape[0], len(omegas))
            Fpetsc, Fpython = pytest_utils.generate_random_bv(comm, sz, True)
            if not complex:
                Fpython[:, 0] = Fpython[:, 0].real
                f = Fpetsc.getColumn(0)
                f = res4py.vec_real(f, True)
                Fpetsc.restoreColumn(0, f)

            # Generate initial condition
            vpetsc, vpython = pytest_utils.generate_random_vector(
                comm, N, complex
            )

            nsteps = 10000
            t_eval = (T / (nsteps - 1)) * np.arange(0, nsteps, 1)
            tf = T if adjoint else -1
            sol_python = sp.integrate.solve_ivp(
                _evaluate_dynamics,
                [0, T],
                vpython,
                t_eval=t_eval,
                rtol=1e-13,
                atol=1e-13,
                args=(Apython, Fpython, omegas, tf),
            ).y
            sol_python = np.fliplr(sol_python) if adjoint else sol_python
            sol_petsc = res4py.solve_ivp(
                vpetsc,
                linop,
                0,
                T,
                nsteps,
                method="RK3",
                m=1,
                adjoint=adjoint,
                periodic_forcing=(Fpetsc, omegas),
            )

            sol_petsc_mat = sol_petsc.getMat()
            sol_petsc_mat_seq = res4py.distributed_to_sequential_matrix(
                sol_petsc_mat
            )
            error = np.linalg.norm(
                sol_python - sol_petsc_mat_seq.getDenseArray()
            )
            error /= np.linalg.norm(sol_python)
            sol_petsc.restoreMat(sol_petsc_mat)
            sol_petsc_mat_seq.destroy()
            error_lst.append(error)

            sol_petsc.destroy()
            linop.destroy()
            Apetsc.destroy()
            Fpetsc.destroy()
            vpetsc.destroy()

    assert np.max(np.asarray(error_lst)) < 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# compute_post_transient_solution(method='gmres') vs method='donothing'
# ──────────────────────────────────────────────────────────────────────────────


def _identity_op(comm, N):
    Nl = res4py.compute_local_size(N)
    Id_mat = res4py.create_AIJ_identity(comm, ((Nl, N), (Nl, N)))
    return res4py.linear_operators.MatrixLinearOperator(Id_mat), Id_mat


def _make_stable_periodic_op(comm, N, freqs, seed):
    r"""Stable real T-periodic operator with one-sided Fourier
    coefficients ``[A_0, A_1, A_2]``.  ``A_0`` is shifted so all
    eigenvalues have ``Re(lambda) <= -0.5``; AC blocks are scaled by
    ``epsilon = 1e-1`` to keep the perturbation modest."""
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((N, N))
    evals, V = np.linalg.eig(A0)
    shift = max(0.0, evals.real.max() + 0.5)
    A0 = (V @ np.diag(evals - shift) @ np.linalg.inv(V)).real.astype(
        np.complex128
    )
    eps = 1e-1
    A1 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    A2 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    Apetsc_lst = [
        pytest_utils.numpy_to_petsc(comm, Ak) for Ak in (A0, A1, A2)
    ]
    linop_lst = [
        res4py.linear_operators.MatrixLinearOperator(A) for A in Apetsc_lst
    ]
    op = res4py.linear_operators.TimePeriodicMatrixLinearOperator(
        linop_lst, freqs, time=0.0
    )
    return op, Apetsc_lst


def _gather_bv(bv):
    Mat = bv.getMat()
    seq = res4py.distributed_to_sequential_matrix(Mat)
    bv.restoreMat(Mat)
    arr = seq.getDenseArray().copy()
    seq.destroy()
    return arr


def _alloc_post_transient_buffers(comm, N, omegas, tsim, nsave):
    state_sz = (res4py.compute_local_size(N), N)
    Fhat, _ = pytest_utils.generate_random_bv(comm, (N, len(omegas)), True)
    Yhat = Fhat.duplicate()
    X = SLEPc.BV().create(comm=comm)
    X.setSizes(state_sz, len(tsim[::nsave]))
    X.setType("mat")
    x_init = Fhat.createVec()
    return Fhat, Yhat, X, x_init


def _run_post_transient_both_methods(
    comm, L, adjoint, omegas, tsim, nsave, N, real_signal
):
    r"""Run ``compute_post_transient_solution`` with both ``method``
    values on the same operator/forcing.  Returns the two Yhat numpy
    arrays for comparison.  ``real_signal=True`` projects the random
    forcing onto a real time signal (one-sided omegas)."""
    Idop, Id_mat = _identity_op(comm, N)

    Fhat, Yhat_dn, X_dn, x_dn = _alloc_post_transient_buffers(
        comm, N, omegas, tsim, nsave
    )
    # If the spectrum is one-sided, the DC mode must be real-valued for
    # the inverse-FFT to give a real time-domain signal.
    if real_signal:
        col0 = Fhat.getColumn(0)
        col0 = res4py.vec_real(col0, True)
        Fhat.restoreColumn(0, col0)

    Yhat_gm = Yhat_dn.duplicate()
    X_gm = X_dn.duplicate()
    x_gm = x_dn.duplicate()

    Yhat_dn = res4py.compute_post_transient_solution(
        L, Idop, Idop, adjoint,
        tsim, nsave, 500, omegas, x_dn,
        Fhat, Yhat_dn, X_dn,
        tol=1e-10, time_stpper="RK3", method="donothing",
    )
    Yhat_gm = res4py.compute_post_transient_solution(
        L, Idop, Idop, adjoint,
        tsim, nsave, 0, omegas, x_gm,
        Fhat, Yhat_gm, X_gm,
        time_stpper="RK3", method="gmres", gmres_rtol=1e-12,
    )

    Y_dn = _gather_bv(Yhat_dn)
    Y_gm = _gather_bv(Yhat_gm)

    for obj in (
        Fhat, Yhat_dn, Yhat_gm, X_dn, X_gm, x_dn, x_gm, Idop, Id_mat
    ):
        obj.destroy()
    return Y_dn, Y_gm


def test_post_transient_gmres_vs_donothing_LTI(comm, square_matrix_size):
    r"""GMRES IC and post-transient iteration must agree on a stable
    LTI system, for both forward and adjoint integration."""
    N, _ = square_matrix_size
    T = 2 * np.pi
    omega = 2 * np.pi / T
    dt = 1e-2
    n_omegas = 5

    errs = []
    for adjoint in [False, True]:
        Apetsc, _ = pytest_utils.generate_stable_random_matrix(
            comm, (N, N), complex=False
        )
        L = res4py.linear_operators.MatrixLinearOperator(Apetsc)
        # Real operator + one-sided spectrum → real time-domain signal.
        tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
            dt, omega, n_omegas, real=True
        )
        Y_dn, Y_gm = _run_post_transient_both_methods(
            comm, L, adjoint, omegas, tsim, nsave, N, real_signal=True
        )
        rel = np.linalg.norm(Y_dn - Y_gm) / np.linalg.norm(Y_dn)
        errs.append(rel)
        L.destroy()
        Apetsc.destroy()
    assert max(errs) < 1e-6, (
        f"LTI: |Y_donothing - Y_gmres| / |Y_donothing| = {errs} "
        f"(forward, adjoint)"
    )


def test_post_transient_gmres_vs_donothing_LTP(comm, square_matrix_size):
    r"""Same check on a stable :class:`TimePeriodicMatrixLinearOperator`
    built from one-sided Fourier coefficients ``[A_0, A_1, A_2]``."""
    N, _ = square_matrix_size
    T = 2 * np.pi
    omega = 2 * np.pi / T
    dt = 1e-2
    n_omegas = 5
    A_freqs = omega * np.array([0.0, 1.0, 2.0])

    errs = []
    for adjoint in [False, True]:
        L, A_petsc_lst = _make_stable_periodic_op(
            comm, N, A_freqs, seed=17 + int(adjoint)
        )
        # Real LTP operator + one-sided spectrum (matches the operator's
        # _real_A flag) → real time-domain signal.
        tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
            dt, omega, n_omegas, real=True
        )
        Y_dn, Y_gm = _run_post_transient_both_methods(
            comm, L, adjoint, omegas, tsim, nsave, N, real_signal=True
        )
        rel = np.linalg.norm(Y_dn - Y_gm) / np.linalg.norm(Y_dn)
        errs.append(rel)
        for linop in L.Alst:
            linop.destroy()
        L.destroy()
        for Ak in A_petsc_lst:
            Ak.destroy()
    assert max(errs) < 1e-6, (
        f"LTP: |Y_donothing - Y_gmres| / |Y_donothing| = {errs} "
        f"(forward, adjoint)"
    )
