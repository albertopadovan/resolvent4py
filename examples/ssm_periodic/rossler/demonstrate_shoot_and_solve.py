r"""
Shoot-and-solve test for ``(s I - L_HB) x = b`` on Rössler, driven
through :func:`compute_post_transient_solution` with
``method='gmres'``.

For periodic ``A(t)`` and ``T``-periodic forcing ``f(t)``, the
``T``-periodic steady state of

.. math::

    \dot{x} = (A(t) - s I)\, x + f(t)

satisfies the BVP

.. math::

    (I - \Phi(T, 0))\, x(0) \;=\; \int_0^T \Phi(T, \tau)\, f(\tau)\, d\tau,

where ``\Phi`` is the monodromy of the *homogeneous* shifted system.
The ``'gmres'`` branch of
:func:`compute_post_transient_solution` wraps ``(I - \Phi)`` as a
PETSc shell, GMRES-solves for ``x(0)``, integrates one more period
to fill the snapshot buffer, and FFTs into HB-ordered Fourier
coefficients.  Unlike the post-transient iteration (``'donothing'``),
this path does **not** require ``Re(s)`` to be in the stable half
plane — it only needs ``I - \Phi(T, 0)`` to be non-singular.

We run two shifts:

1. a **stable** shift (``s = 2``) where the post-transient iteration
   would also converge; and
2. an **SSM-style** shift (``s = 2 \lambda_{\rm master}`` with
   ``Re(\lambda_{\rm master}) < 0``) where post-transient diverges
   but shoot-and-solve still works.
"""

import numpy as np
from scipy.signal import resample

from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py
from resolvent4py.utils.vector import reshape_harmonic_balanced_vector_into_bv
from resolvent4py.utils.bv import reshape_bv_into_harmonic_balanced_vector

from rossler_differential_equation import RosslerPeriodic


comm = PETSc.COMM_WORLD


# ── Parameters ──────────────────────────────────────────────────────────────
# Period-doubled construction (matches save_ssm_2T.py): tile c_star to
# 2·T_base and declare ``T = 2·T_base``, so the HB basis lives at
# multiples of ``omega_base/2``.  Shooting over "one period" then IS
# shooting over 2·T_base, and the FFT picks up the subharmonics needed
# for a 2T-periodic SSM response.
nf = 40        # 2T orbit needs more harmonics for the same physical bandwidth
nfb = 20
ts_dt = 1e-2
gmres_rtol = 1e-10
gmres_max_it = 200
time_stepper = "RK3"


# ── Load periodic orbit, tile to 2T ─────────────────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T_base = float(data["T"])
C_periodic_1T = data["C"][:, :-1]

T = 2.0 * T_base
C_periodic_2T = np.tile(C_periodic_1T, (1, 2))

n_time = 2 * (nf + nfb) + 1
C_periodic = resample(C_periodic_2T, n_time, axis=1)
time_orbit = np.linspace(0, T, n_time, endpoint=False)
omega = 2 * np.pi / T          # = omega_base / 2

res4py.petscprint(
    comm,
    f"Rössler (2T basis): c={c}, T=2·T_base={T:.6f}, "
    f"nf={nf}, nfb={nfb}",
)


# ── Equations: one for the algebraic reference, one for A(t) access ────────
eq_alg = RosslerPeriodic(
    c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit
)
eq_ts = RosslerPeriodic(
    c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit,
    use_time_stepping=True, ts_dt=ts_dt, ts_method=time_stepper,
)

A_t_op = eq_ts._A_t_op            # TimePeriodicMatrixLinearOperator of A(t)
tsim = eq_ts._ts_tsim             # one-period sim grid (uniform dt)
nsave = eq_ts._ts_nsave           # snapshot stride
omegas_HB = eq_ts._ts_omegas      # HB-ordered pertb_freqs


# ── Conjugate-symmetric HB rhs (so x is conj-symmetric for real s) ──────────
N_hb = eq_alg.n * (2 * nf + 1)
rng = np.random.default_rng(0)
b_pos = (
    rng.standard_normal((nf + 1, eq_alg.n))
    + 1j * rng.standard_normal((nf + 1, eq_alg.n))
)
b_pos[0] = b_pos[0].real
b_full = np.zeros((2 * nf + 1, eq_alg.n), dtype=complex)
b_full[nf:] = b_pos
b_full[:nf] = b_pos[1:][::-1].conj()
rhs_np = b_full.ravel()

rhs = PETSc.Vec().create(comm=comm)
rhs.setSizes((res4py.compute_local_size(N_hb), N_hb))
rhs.setUp()
r0, r1 = rhs.getOwnershipRange()
rhs.setValues(np.arange(r0, r1, dtype=PETSc.IntType), rhs_np[r0:r1])
rhs.assemble()


# ── The shoot-and-solve routine ─────────────────────────────────────────────
state_dim = A_t_op.get_dimensions()[0]
Id_mat = res4py.create_AIJ_identity(comm, (state_dim, state_dim))
Idop = res4py.linear_operators.MatrixLinearOperator(Id_mat)


def shoot_and_solve(s, b_vec, verbose=False):
    r"""Solve ``(s I - L_HB) x = b_vec`` via
    :func:`compute_post_transient_solution` with ``method='gmres'``.
    Builds the shifted operator ``A(t) - s I`` on top of
    ``eq_ts._A_t_op`` and lets the library do the shoot-and-solve.
    Returns the HB long vector of the steady-state response
    (HB-ordered)."""
    shifted = res4py.linear_operators.ShiftAndScaleLinearOperator(
        A_t_op, alpha=-s, beta=1.0
    )

    # Reshape b_vec into BV form, HB-ordered to match omegas_HB.
    F_BV = SLEPc.BV().create(comm=comm)
    F_BV.setSizes(state_dim, 2 * nf + 1)
    F_BV.setType("mat")
    reshape_harmonic_balanced_vector_into_bv(b_vec, 2 * nf + 1, F_BV)

    Y_BV = F_BV.duplicate()
    X_BV = SLEPc.BV().create(comm=comm)
    X_BV.setSizes(state_dim, len(tsim[::nsave]))
    X_BV.setType("mat")
    x_init = F_BV.createVec()
    x_init.zeroEntries()

    Y_BV = res4py.compute_post_transient_solution(
        shifted, Idop, Idop, False,
        tsim, nsave, 0, omegas_HB, x_init,
        F_BV, Y_BV, X_BV,
        time_stpper=time_stepper,
        harmonic_balancing_ordering=True,
        method="gmres",
        gmres_rtol=gmres_rtol,
        gmres_max_it=gmres_max_it,
        verbose=2 if verbose else 0,
    )

    x = reshape_bv_into_harmonic_balanced_vector(Y_BV)

    for obj in (x_init, F_BV, Y_BV, X_BV):
        obj.destroy()
    return x


# ── Compare against algebraic for two shifts ────────────────────────────────
def compare(shift, label):
    res4py.petscprint(comm, f"\n=== {label}: shift = {shift} ===")
    res4py.petscprint(comm, "  Algebraic solve …")
    x_alg = eq_alg.solve_linear_system(shift, rhs)
    res4py.petscprint(comm, "  Shoot-and-solve …")
    x_ts = shoot_and_solve(shift, rhs, verbose=True)

    a = res4py.distributed_to_sequential_vector(x_alg).getArray().copy()
    t = res4py.distributed_to_sequential_vector(x_ts).getArray().copy()
    abs_err = np.linalg.norm(a - t)
    rel_err = abs_err / np.linalg.norm(a)
    res4py.petscprint(
        comm,
        f"  ||x_ts - x_alg|| = {abs_err:.3e},  rel = {rel_err:.3e}",
    )

    a_blk = a.reshape(2 * nf + 1, eq_alg.n)
    t_blk = t.reshape(2 * nf + 1, eq_alg.n)
    res4py.petscprint(comm, "  per-mode rel err:")
    for k in range(2 * nf + 1):
        mode = k - nf
        ref = np.linalg.norm(a_blk[k])
        err = (
            np.linalg.norm(a_blk[k] - t_blk[k]) / ref
            if ref > 1e-14
            else 0.0
        )
        res4py.petscprint(
            comm, f"    HB[{k:3d}]  mode {mode:+3d}  rel err = {err:.2e}"
        )


# 1) Stable shift — both paths should agree to RK3 truncation.
compare(2.0, "STABLE shift")

# 2) SSM-style shift — in the 2T-doubled basis the master Floquet
#    exponent is real (the original ±i·omega_base/2 collapses to 0),
#    so SSM shifts s_k = k·lam_master are real negative.  Post-transient
#    iteration would diverge; shoot-and-solve only needs I - Phi to be
#    invertible.
lam_master = -0.00345287                                  # from cache_2T
compare(2.0 * lam_master, "SSM k=2 shift (s = 2·lam_master, real)")
compare(3.0 * lam_master, "SSM k=3 shift")
