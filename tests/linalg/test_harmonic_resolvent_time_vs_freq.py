r"""Cross-check time-domain post-transient integration of a forced
time-periodic LTV system against the corresponding harmonic-resolvent
frequency-domain solve.

For real T-periodic operators :math:`A(t), B(t), C(t)` and a real
periodic forcing :math:`f(t) = f_\omega e^{i k \omega t} +
\mathrm{c.c.}`, the steady-state output of

.. math::

    \dot{x}(t) = A(t)\,x(t) + B(t)\,f(t),
    \qquad y(t) = C^*(t)\,x(t)

(matching the :math:`C^*` projection convention in
:func:`compute_post_transient_solution`) satisfies the
harmonic-balance equation

.. math::

    \hat{y} \;=\; C_{\rm HB}^{*}\,
        (i \Omega - A_{\rm HB})^{-1}\, B_{\rm HB}\, \hat{f},

where :math:`M_{\rm HB}` is the block-Toeplitz Fourier matrix of
:math:`M(t)` and :math:`\Omega` is the diagonal of perturbation
frequencies.  The adjoint problem
:math:`-\dot{x} = A^*(t) x + B(t) f(t)` replaces
:math:`(i \Omega - A_{\rm HB})^{-1}` with the hermitian-transpose
solve.

Four tests, all single-harmonic real forcings:

1. forward, :math:`B = C = I` — pure state-response check;
2. adjoint, :math:`B = C = I` — pure state-response check;
3. forward, time-periodic :math:`B(t), C(t)` — full input/output;
4. adjoint, time-periodic :math:`B(t), C(t)` — full input/output.
"""

import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from slepc4py import SLEPc
from .. import pytest_utils


# ── helpers ────────────────────────────────────────────────────────────────


def _make_periodic_A_coeffs(N, seed):
    r"""One-sided coefficients of a real T-periodic
    :math:`A(t) = A_0 + 2\mathrm{Re}[A_1 e^{i\omega t} + A_2 e^{i 2\omega t}]`
    with ``omega = 1.2``.  ``A_0`` is shifted to be stable
    (eigenvalues with real part :math:`\le -0.5`), AC blocks are scaled
    by :math:`\varepsilon = 10^{-1}`."""
    rng = np.random.default_rng(seed)
    freqs = np.array([0.0, 1.2, 2.4])
    A0 = rng.standard_normal((N, N))
    evals, V = np.linalg.eig(A0)
    shift = max(0.0, evals.real.max() + 0.5)
    A0 = (V @ np.diag(evals - shift) @ np.linalg.inv(V)).real.astype(np.complex128)
    eps = 1e-1
    A1 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    A2 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    return [A0, A1, A2], freqs


def _build_time_periodic_op(comm, Alst_np, freqs, time):
    r"""Wrap the numpy Fourier coefficients into a
    :class:`TimePeriodicMatrixLinearOperator`.

    Returns the operator along with the constituent PETSc matrices and
    ``MatrixLinearOperator`` objects, which the caller owns and must
    destroy: the TimePeriodicMatrixLinearOperator only frees its own
    work vectors, and each MatrixLinearOperator frees only its internal
    Hermitian-transpose copy (not the matrix passed in)."""
    Apetsc_lst = [pytest_utils.numpy_to_petsc(comm, Ak) for Ak in Alst_np]
    linop_lst = [
        res4py.linear_operators.MatrixLinearOperator(A) for A in Apetsc_lst
    ]
    op = res4py.linear_operators.TimePeriodicMatrixLinearOperator(
        linop_lst, freqs, time
    )
    return op, linop_lst, Apetsc_lst


def _destroy_time_periodic_op(op, linop_lst, Apetsc_lst):
    r"""Destroy a TimePeriodicMatrixLinearOperator together with its
    constituent operators and matrices (see :func:`_build_time_periodic_op`)."""
    op.destroy()
    for linop in linop_lst:
        linop.destroy()
    for A in Apetsc_lst:
        A.destroy()


def _block_toeplitz_HB(Mlst_np, n_pert):
    r"""Numpy block-Toeplitz matrix
    :math:`M_{\rm HB}\bigl[(i, j)\bigr] = M_{i-j}` of total size
    :math:`(2 n_{\rm pert}+1)\, M_r \times (2 n_{\rm pert}+1)\, M_c`.
    ``Mlst_np`` is one-sided (``Mlst_np[k]`` for
    ``k = 0, 1, …, nfb``); negative-frequency blocks are reconstructed
    as :math:`M_{-k} = \overline{M_k}` (real-operator assumption).
    Blocks more than ``nfb`` away from the diagonal are zero."""
    Mr, Mc = Mlst_np[0].shape
    nfb = len(Mlst_np) - 1
    total_r = Mr * (2 * n_pert + 1)
    total_c = Mc * (2 * n_pert + 1)

    def coeff_at_k(k):
        if abs(k) > nfb:
            return None
        return Mlst_np[k] if k >= 0 else Mlst_np[-k].conj()

    H = np.zeros((total_r, total_c), dtype=complex)
    for i in range(-n_pert, n_pert + 1):
        for j in range(-n_pert, n_pert + 1):
            blk = coeff_at_k(i - j)
            if blk is not None:
                H[(i + n_pert) * Mr : (i + n_pert + 1) * Mr,
                  (j + n_pert) * Mc : (j + n_pert + 1) * Mc] = blk
    return H


def _build_HB_T_op(comm, Alst_np, n_pert, omega_base):
    r"""Build the harmonic-resolvent generator
    :math:`T = i \Omega - A_{\rm HB}` with two-sided perturbation
    frequencies :math:`\omega \cdot \{-n_{\rm pert}, \dots, n_{\rm pert}\}`,
    wrap it in a MUMPS LU solver, and return both the operator and the
    perturbation-frequency array."""
    perts_freqs = omega_base * np.arange(-n_pert, n_pert + 1, dtype=float)
    A_hb_np = _block_toeplitz_HB(Alst_np, n_pert)
    A_hb = pytest_utils.numpy_to_petsc(comm, A_hb_np)
    T = res4py.assemble_harmonic_resolvent_generator(A_hb, perts_freqs)
    T.scale(-1.0)
    A_hb.destroy()

    ksp = res4py.create_direct_solver(T)
    res4py.check_solver(T, ksp)
    Top = res4py.linear_operators.MatrixLinearOperator(
        T, ksp, 2 * n_pert + 1
    )
    # Top.destroy() frees only the internal Hermitian-transpose copy; T and
    # ksp are created here and returned so the caller can destroy them.
    return Top, perts_freqs, T, ksp


def _make_HB_force_vec(comm, N, n_pert, f_omega_np, harmonic_index):
    r"""Build the HB-format forcing vector (two-sided, size
    ``N·(2 n_pert + 1)``) with block at ``+harmonic_index`` set to
    ``f_omega`` and block at ``-harmonic_index`` set to its conjugate
    (other blocks zero)."""
    total = N * (2 * n_pert + 1)
    F_np = np.zeros(total, dtype=complex)
    pos = (n_pert + harmonic_index) * N
    neg = (n_pert - harmonic_index) * N
    F_np[pos : pos + N] = f_omega_np
    F_np[neg : neg + N] = f_omega_np.conj()

    F_vec = PETSc.Vec().create(comm=comm)
    F_vec.setSizes((res4py.compute_local_size(total), total))
    F_vec.setUp()
    r0, r1 = F_vec.getOwnershipRange()
    F_vec.setValues(np.arange(r0, r1, dtype=PETSc.IntType), F_np[r0:r1])
    F_vec.assemble()
    return F_vec


def _make_one_sided_force_BV(comm, N, n_omegas, f_omega_np, harmonic_index):
    r"""Build a one-sided forcing BV with ``n_omegas + 1`` columns —
    column ``harmonic_index`` carries ``f_omega``, all other columns
    are zero.  This is the format
    :func:`compute_post_transient_solution` expects for a real signal."""
    bv = SLEPc.BV().create(comm=comm)
    bv.setSizes((res4py.compute_local_size(N), N), n_omegas + 1)
    bv.setType("mat")
    bvm = bv.getMat()
    bvm.zeroEntries()
    bv.restoreMat(bvm)

    col = bv.getColumn(harmonic_index)
    r0, r1 = col.getOwnershipRange()
    col.setValues(np.arange(r0, r1, dtype=PETSc.IntType), f_omega_np[r0:r1])
    col.assemble()
    bv.restoreColumn(harmonic_index, col)
    return bv


def _hb_solve_and_gather(comm, Top, F_hb_vec, n_pert, N, adjoint):
    r"""Solve :math:`T \hat{x} = \hat{f}` (or the hermitian-transpose
    variant) and gather the result into a numpy array of shape
    ``(2 n_pert + 1, N)`` indexed by harmonic block."""
    x_hat = (
        Top.solve_hermitian_transpose(F_hb_vec) if adjoint
        else Top.solve(F_hb_vec)
    )
    x_seq = res4py.distributed_to_sequential_vector(x_hat)
    x_np = x_seq.getArray().copy().reshape(2 * n_pert + 1, N)
    x_seq.destroy()
    x_hat.destroy()
    return x_np


def _gather_BV_columns(BV):
    r"""Gather a distributed BV's underlying dense matrix to a numpy
    array of shape ``(N, ncols)`` replicated on every rank."""
    Mat = BV.getMat()
    Mat_seq = res4py.distributed_to_sequential_matrix(Mat)
    BV.restoreMat(Mat)
    arr = Mat_seq.getDenseArray().copy()
    Mat_seq.destroy()
    return arr


def _make_periodic_BC_coeffs(N, seed, eps=1e-1):
    r"""One-sided coefficients of a real T-periodic matrix-valued
    operator (``B(t)`` or ``C(t)``) sharing the
    :math:`\omega \in \{0, 1.2, 2.4\}` set with the state operator.
    Mean block is order one; AC blocks are scaled by ``eps``.  No
    stability shift — these are passive input/output maps."""
    rng = np.random.default_rng(seed)
    freqs = np.array([0.0, 1.2, 2.4])
    M0 = rng.standard_normal((N, N)).astype(np.complex128)
    M1 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    M2 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    return [M0, M1, M2], freqs


def _numpy_vec_to_distributed(comm, total, arr_np):
    r"""Build a distributed PETSc Vec of length ``total`` from a
    rank-replicated numpy array."""
    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes((res4py.compute_local_size(total), total))
    vec.setUp()
    r0, r1 = vec.getOwnershipRange()
    vec.setValues(np.arange(r0, r1, dtype=PETSc.IntType), arr_np[r0:r1])
    vec.assemble()
    return vec


def _gather_to_numpy(petsc_vec):
    r"""Sequentialize a distributed PETSc Vec to a rank-replicated numpy
    array."""
    seq = res4py.distributed_to_sequential_vector(petsc_vec)
    arr = seq.getArray().copy()
    seq.destroy()
    return arr


def _run_time_vs_freq_check(comm, adjoint, seed):
    r"""Shared kernel for both forward and adjoint tests.  Builds the
    time-periodic operator and the harmonic-resolvent solver, runs the
    time-domain post-transient integration with a single-harmonic
    forcing, performs the matching frequency-domain solve, and
    compares.  Returns the per-harmonic relative errors."""
    N = 8
    omega_base = 1.2
    # n_omegas sizes the save grid (2*(n_omegas+2) samples per period →
    # Nyquist mode n_omegas+2), and n_pert sizes the HB system.  We pick
    # both well above n_compare so neither Nyquist aliasing in the FFT
    # nor HB-boundary truncation pollutes the modes we actually check.
    n_omegas = 24
    n_pert = 24
    n_compare = 10
    harmonic_index = 1
    dt = 1e-3
    nperiods = 30
    tol = 1e-9

    Alst_np, A_freqs = _make_periodic_A_coeffs(N, seed=seed)
    rng = np.random.default_rng(seed + 100)
    f_omega_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    # ── time-domain post-transient integration ────────────────────────
    Atop, Atop_linops, Atop_mats = _build_time_periodic_op(
        comm, Alst_np, A_freqs, time=0.0
    )
    Id_mat = res4py.create_AIJ_identity(
        comm,
        (
            (res4py.compute_local_size(N), N),
            (res4py.compute_local_size(N), N),
        ),
    )
    Idop = res4py.linear_operators.MatrixLinearOperator(Id_mat)

    tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
        dt, omega_base, n_omegas, real=True
    )

    F_BV = _make_one_sided_force_BV(
        comm, N, n_omegas, f_omega_np, harmonic_index
    )
    Y_BV = F_BV.duplicate()
    X = SLEPc.BV().create(comm=comm)
    X.setSizes((res4py.compute_local_size(N), N), len(tsim[::nsave]))
    X.setType("mat")

    x0 = F_BV.createVec()
    x0.zeroEntries()

    Y_BV = res4py.compute_post_transient_solution(
        Atop, Idop, Idop, adjoint,
        tsim, nsave, nperiods, omegas, x0,
        F_BV, Y_BV, X,
        tol, "RK3", 0,
    )

    Y_np = _gather_BV_columns(Y_BV)   # shape (N, n_omegas + 1)

    # ── frequency-domain harmonic-resolvent solve ────────────────────
    Top, _, T_mat, T_ksp = _build_HB_T_op(comm, Alst_np, n_pert, omega_base)
    F_hb_vec = _make_HB_force_vec(comm, N, n_pert, f_omega_np, harmonic_index)
    x_hb_np = _hb_solve_and_gather(comm, Top, F_hb_vec, n_pert, N, adjoint)

    # ── compare per-harmonic on the well-resolved range only ─────────
    errors = []
    for k in range(n_compare + 1):
        y_k = Y_np[:, k]
        x_k = x_hb_np[n_pert + k]
        ref = np.linalg.norm(x_k)
        if ref > 1e-14:
            errors.append(np.linalg.norm(y_k - x_k) / ref)
        else:
            errors.append(np.linalg.norm(y_k - x_k))

    # ── cleanup ───────────────────────────────────────────────────────
    x0.destroy()
    F_BV.destroy()
    Y_BV.destroy()
    X.destroy()
    F_hb_vec.destroy()
    Top.destroy()
    T_mat.destroy()
    T_ksp.destroy()
    _destroy_time_periodic_op(Atop, Atop_linops, Atop_mats)
    Idop.destroy()
    Id_mat.destroy()
    return errors


# ── tests ──────────────────────────────────────────────────────────────────


def test_harmonic_resolvent_forward(comm):
    r"""Forward post-transient response of :math:`\dot{x} = A(t) x + f(t)`
    matches the harmonic-resolvent solve
    :math:`\hat{x} = (i \Omega - A_{\rm HB})^{-1} \hat{f}`."""
    errors = _run_time_vs_freq_check(comm, adjoint=False, seed=11)
    msg = "\n".join(
        f"  harmonic {k}: rel error = {e:.2e}" for k, e in enumerate(errors)
    )
    max_err = max(errors)
    assert max_err < 1e-4, (
        f"forward time-vs-freq max rel error = {max_err:.2e}\n{msg}"
    )


def test_harmonic_resolvent_adjoint(comm):
    r"""Adjoint post-transient response of
    :math:`-\dot{x} = A^*(t) x + f(t)` matches
    :math:`\hat{x} = (i \Omega - A_{\rm HB})^{-*} \hat{f}`."""
    errors = _run_time_vs_freq_check(comm, adjoint=True, seed=29)
    msg = "\n".join(
        f"  harmonic {k}: rel error = {e:.2e}" for k, e in enumerate(errors)
    )
    max_err = max(errors)
    assert max_err < 1e-4, (
        f"adjoint time-vs-freq max rel error = {max_err:.2e}\n{msg}"
    )


# ── Time-periodic B(t) and C(t) ───────────────────────────────────────────


def _run_time_periodic_BC_check(comm, adjoint, seed):
    r"""Like :func:`_run_time_vs_freq_check` but with time-periodic
    input and output operators :math:`B(t)` and :math:`C(t)`.

    The matching frequency-domain solve is

    .. math::

        \hat{y} \;=\; C_{\rm HB}^{*}\,
            (i\Omega - A_{\rm HB})^{-1}\, B_{\rm HB}\, \hat{f}

    for the forward case (and the hermitian-transpose variant of
    :math:`T = i\Omega - A_{\rm HB}` for the adjoint), where
    :math:`B_{\rm HB}` and :math:`C_{\rm HB}` are the block-Toeplitz
    Fourier matrices of :math:`B(t)` and :math:`C(t)`.  The
    :math:`C_{\rm HB}^{*}` factor matches the convention in
    :func:`compute_post_transient_solution`, which projects state
    snapshots through ``C.apply_hermitian_transpose``."""
    N = 8
    omega_base = 1.2
    n_omegas = 24
    n_pert = 24
    n_compare = 10
    harmonic_index = 1
    dt = 1e-3
    nperiods = 30
    tol = 1e-9

    Alst_np, freqs = _make_periodic_A_coeffs(N, seed=seed)
    Blst_np, _ = _make_periodic_BC_coeffs(N, seed=seed + 200)
    Clst_np, _ = _make_periodic_BC_coeffs(N, seed=seed + 400)
    rng = np.random.default_rng(seed + 100)
    f_omega_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    # ── time-domain post-transient integration ────────────────────────
    Atop, Atop_linops, Atop_mats = _build_time_periodic_op(
        comm, Alst_np, freqs, time=0.0
    )
    Btop, Btop_linops, Btop_mats = _build_time_periodic_op(
        comm, Blst_np, freqs, time=0.0
    )
    Ctop, Ctop_linops, Ctop_mats = _build_time_periodic_op(
        comm, Clst_np, freqs, time=0.0
    )

    tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
        dt, omega_base, n_omegas, real=True
    )

    F_BV = _make_one_sided_force_BV(
        comm, N, n_omegas, f_omega_np, harmonic_index
    )
    Y_BV = F_BV.duplicate()
    X = SLEPc.BV().create(comm=comm)
    X.setSizes((res4py.compute_local_size(N), N), len(tsim[::nsave]))
    X.setType("mat")

    x0 = F_BV.createVec()
    x0.zeroEntries()

    Y_BV = res4py.compute_post_transient_solution(
        Atop, Btop, Ctop, adjoint,
        tsim, nsave, nperiods, omegas, x0,
        F_BV, Y_BV, X,
        tol, "RK3", 0,
    )

    Y_np = _gather_BV_columns(Y_BV)   # shape (N, n_omegas + 1)

    # ── frequency-domain harmonic-resolvent solve ────────────────────
    Top, _, T_mat, T_ksp = _build_HB_T_op(comm, Alst_np, n_pert, omega_base)
    B_HB_np = _block_toeplitz_HB(Blst_np, n_pert)
    C_HB_np = _block_toeplitz_HB(Clst_np, n_pert)
    Cstar_HB_np = C_HB_np.conj().T   # matches C.apply_hermitian_transpose

    total = N * (2 * n_pert + 1)
    F_hb_vec = _make_HB_force_vec(comm, N, n_pert, f_omega_np, harmonic_index)
    F_hb_np = _gather_to_numpy(F_hb_vec)
    F_hb_vec.destroy()
    rhs_np = B_HB_np @ F_hb_np
    rhs_vec = _numpy_vec_to_distributed(comm, total, rhs_np)

    x_hat = (
        Top.solve_hermitian_transpose(rhs_vec) if adjoint
        else Top.solve(rhs_vec)
    )
    x_hat_np = _gather_to_numpy(x_hat)
    x_hat.destroy()

    y_hat_np = Cstar_HB_np @ x_hat_np
    y_hat_blocks = y_hat_np.reshape(2 * n_pert + 1, N)

    # ── compare per-harmonic ──────────────────────────────────────────
    errors = []
    for k in range(n_compare + 1):
        y_k_time = Y_np[:, k]
        y_k_freq = y_hat_blocks[n_pert + k]
        ref = np.linalg.norm(y_k_freq)
        if ref > 1e-14:
            errors.append(np.linalg.norm(y_k_time - y_k_freq) / ref)
        else:
            errors.append(np.linalg.norm(y_k_time - y_k_freq))

    # ── cleanup ───────────────────────────────────────────────────────
    x0.destroy()
    F_BV.destroy()
    Y_BV.destroy()
    X.destroy()
    rhs_vec.destroy()
    Top.destroy()
    T_mat.destroy()
    T_ksp.destroy()
    _destroy_time_periodic_op(Atop, Atop_linops, Atop_mats)
    _destroy_time_periodic_op(Btop, Btop_linops, Btop_mats)
    _destroy_time_periodic_op(Ctop, Ctop_linops, Ctop_mats)
    return errors


def test_harmonic_resolvent_forward_periodic_BC(comm):
    r"""Forward post-transient response with time-periodic
    :math:`A(t), B(t), C(t)` matches
    :math:`\hat{y} = C_{\rm HB}^{*} (i\Omega - A_{\rm HB})^{-1}
    B_{\rm HB}\, \hat{f}`."""
    errors = _run_time_periodic_BC_check(comm, adjoint=False, seed=17)
    msg = "\n".join(
        f"  harmonic {k}: rel error = {e:.2e}" for k, e in enumerate(errors)
    )
    max_err = max(errors)
    assert max_err < 1e-4, (
        f"forward (time-periodic B, C) max rel error = "
        f"{max_err:.2e}\n{msg}"
    )


def test_harmonic_resolvent_adjoint_periodic_BC(comm):
    r"""Adjoint post-transient response with time-periodic
    :math:`A(t), B(t), C(t)` matches
    :math:`\hat{y} = C_{\rm HB}^{*} (i\Omega - A_{\rm HB})^{-*}
    B_{\rm HB}\, \hat{f}`."""
    errors = _run_time_periodic_BC_check(comm, adjoint=True, seed=31)
    msg = "\n".join(
        f"  harmonic {k}: rel error = {e:.2e}" for k, e in enumerate(errors)
    )
    max_err = max(errors)
    assert max_err < 1e-4, (
        f"adjoint (time-periodic B, C) max rel error = "
        f"{max_err:.2e}\n{msg}"
    )
