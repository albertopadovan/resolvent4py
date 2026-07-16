"""
Compute the Floquet eigendecomposition and the neutral-projection
operator for the KSE periodic orbit using a 2T harmonic-balanced
embedding.

Rationale: the orbit is T-periodic, but the leading non-neutral Floquet
multiplier approaches μ = -1 (period-doubling).  Its eigenvector is 2T-
periodic.  Embedding everything in a 2T-HB framework lets that mode
live cleanly at k = ±1 of the doubled-frequency grid:

    HB fundamental:  ω_HB = 2π/(2T) = ω/2
    physical freq at HB index k: k · ω_HB = k · ω / 2

A(t) is T-periodic, so its FFT over 2T has zero coefficients at odd
k (= half-integer ω).  The HB matrix block-decouples into even-k
and odd-k subspaces.  The period-doubling Floquet exponent λ_T = iω/2
folds to λ_2T = 0 in the principal 2T-HB strip, so we shift-invert
at σ = 0.  Eigenvectors are saved as the eigensolver returns them;
the even-k zeroing for the period-doubling slot is done at load time
in save_ssm.py (and *not* on the neutral mode).

Run with:
    mpiexec -n <nprocs> python save_eigendecomp.py
"""

import os

import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py

from kse_differential_equation import KuramotoSivashinskyPeriodic
from eigendecomp_kse import compute_eigendecomposition, save


# ── Parameters ───────────────────────────────────────────────────────────────
# nf, nfb are derived from the HB-Newton ``nf_HB`` (T-periodic harmonics
# retained when refining the orbit).  In the 2T-cover, every T-harmonic
# index k maps to a 2T-harmonic index 2k, so to capture the same physical
# bandwidth we need nfb = 2 · nf_HB.  ``nf`` is the eigenproblem
# bandwidth — keep it ≥ nfb plus a small margin for clean resolution.
n_evals = 20
krylov_dim = 100
# Deflate the orbit-tangent neutral direction at multiple harmonics of
# ω_HB (k=0, ±1, ±2, ±3, ±4 → 9 directions total).  This (a) lets the
# σ=0 shift-invert return the period-doubling mode cleanly at index 0
# (instead of mixed with the λ=0 neutral), and (b) gives the
# downstream rom.neutral_project a multi-column basis that wipes ALL
# orbit-tangent harmonics from an off-manifold IC — single-column
# projection (the λ=0 mode only) leaks the higher harmonics through
# and causes the truth to drift along the orbit.  Matches the rossler
# 2T setup.
n_neutral_strips = 6

out_path = "data/eigendecomp_cache.npz"


comm = PETSc.COMM_WORLD


# ── Load the T-periodic orbit and embed it into the 2T-HB time grid ─────────
data = np.load("data/periodic_orbit.npz")
nu = float(data["nu"])
n = int(data["n"])
n_pts = int(data["n_pts"])
T = float(data["T"])
nf_HB = int(data["nf"])                 # HB-Newton's T-periodic nf
C_periodic_full = data["C"][:, :-1]

# 2T-HB bandwidth derived from the HB-Newton truncation
nfb = 2 * nf_HB                         # 2T-HB: each T-harmonic k → 2T k' = 2k
nf = nfb + 10                           # margin above nfb for eigenproblem

T_HB = 2.0 * T                          # harmonic-balance period
omega_orbit = 2.0 * np.pi / T           # ω of the T-orbit
omega_HB = 2.0 * np.pi / T_HB           # = ω/2 — fundamental of 2T-HB

# n_time samples over [0, 2T): split evenly so that each T contains
# n_time / 2 points and the second T is an exact copy of the first.
n_time_per_T = nf + nfb + 1
n_time = 2 * n_time_per_T

C_periodic_one_T = resample(C_periodic_full, n_time_per_T, axis=1)
C_periodic = np.concatenate([C_periodic_one_T, C_periodic_one_T], axis=1)
time_orbit = np.linspace(0.0, T_HB, n_time, endpoint=False)

omegas_one_sided = omega_HB * np.arange(nf + 1)
periodic_diffeq = (omegas_one_sided, time_orbit, False)

# Period-doubling shift.  Use a TINY offset from zero (rather than σ = 0
# exactly) so that L_HB − σ·I is non-singular even when the orbit is
# exact to machine precision: the orbit-tangent direction is a true kernel
# mode of L_HB at λ = 0, and MUMPS factorisation of an exactly-singular
# matrix returns garbage that breaks both the neutral-strip deflation and
# the principal shift-invert.  A 1e-8 offset is far below any meaningful
# eigvalue, so the recovered eigenvalues are accurate to ~1e-8.
sigma = 1e-8

res4py.petscprint(
    comm,
    f"KSE periodic orbit (2T-HB):  nu={nu}, n={n}, T={T:.6f},  "
    f"T_HB=2T={T_HB:.6f},  ω_HB={omega_HB:.6f}",
)
res4py.petscprint(
    comm,
    f"  n_time={n_time}  (= 2 × {n_time_per_T}),  nf={nf},  nfb={nfb}",
)


# ── Build the equation ─────────────────────────────────────────────────────
eq = KuramotoSivashinskyPeriodic(
    n=n,
    nu=nu,
    nf=nf,
    nfb=nfb,
    c_star=C_periodic,
    time=time_orbit,
    n_pts=n_pts,
    periodic_diffeq=periodic_diffeq,
)

res4py.petscprint(comm, f"omega         = {eq.omega:.6f}  (= ω/2)")
res4py.petscprint(comm, f"HB state dim  = {n * (2 * nf + 1)}")


# ── Compute eigendecomposition ─────────────────────────────────────────────
# neutral_omega = omega_orbit (= 2 * eq.omega = 2 * omega_HB):  the
# T-periodic orbit-tangent neutrals sit at integer multiples of the
# *base* orbit frequency ω_orbit, which in the 2T basis is 2·ω_HB.
# Using the default eq.omega would deflate at half-integer multiples
# of ω_orbit — *including* the period-doubling mode at λ = i·ω_orbit/2.
L, Phi, Psi, neutral_proj = compute_eigendecomposition(
    eq,
    n_evals=n_evals,
    krylov_dim=krylov_dim,
    sigma=sigma,
    n_neutral_strips=n_neutral_strips,
    neutral_omega=omega_orbit,
)

res4py.petscprint(comm, f"# eigvals     = {L.shape[0]}")


# ── Enforce HB conj-symmetry on the eigenvectors ─────────────────────────
# `eig()` already calls `enforce_complex_conjugacy` on its Krylov basis
# (when the operator's `block_cc_flag = True`), but the SLEPc normalisation
# step at the end of the Arnoldi can leave each *returned* eigenvector
# with an arbitrary global phase `e^{iθ}` — destroying the conj-symmetric
# block structure.  We re-impose it here on every column of Phi and Psi
# (and the underlying neutral-projection bases) BEFORE saving the cache,
# so that downstream SSM iterates produce conj-symmetric ps[j] / gs[j].
from resolvent4py.utils.vector import enforce_complex_conjugacy

n_harmonics_hb = 2 * nf + 1


def _enforce_cc_on_bv(bv, name=""):
    ncols = bv.getSizes()[-1]
    for k in range(ncols):
        col = bv.getColumn(k)
        enforce_complex_conjugacy(col, n_harmonics_hb)
        bv.restoreColumn(k, col)
    res4py.petscprint(
        comm, f"  enforced conj-symmetry on {name} ({ncols} cols)"
    )


res4py.petscprint(comm, "Enforcing HB conj-symmetry on eigenvectors:")
_enforce_cc_on_bv(Phi, "Phi")
_enforce_cc_on_bv(Psi, "Psi")
# Also re-impose on the neutral projection's right/left bases (used to
# wipe orbit-tangent directions from off-manifold ICs downstream).
_enforce_cc_on_bv(neutral_proj.L.L.U, "neutral V")
_enforce_cc_on_bv(neutral_proj.L.L.V, "neutral W")


# ── Save Φ, Ψ as-is.  The period-doubling slice gets zeroed at load
# time in save_ssm.py — not here, and not on the neutral mode.
save(out_path, L, Phi, Psi, neutral_proj, eq)


# ── Plot Floquet multipliers in the 2T cover (μ_2T = exp(λ · 2T)) ──────────
# Period-doubling mode lands at μ_2T ≈ +1 (because exp(iω/2 · 2T) = +1).
if comm.getRank() == 0:
    mults = np.exp(L * T_HB)
    order = np.argsort(-np.abs(mults))
    print(
        f"Floquet multipliers in 2T cover  μ_2T = exp(λ · 2T)  "
        f"(sorted by |μ| descending, {mults.size} total):"
    )
    for i, k in enumerate(order):
        print(
            f"  [{i:3d}] λ = {L[k].real:+.6e} {L[k].imag:+.6e}j   "
            f"μ_2T = {mults[k].real:+.6e} {mults[k].imag:+.6e}j   "
            f"|μ_2T| = {abs(mults[k]):.6e}"
        )
    theta = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(
        np.cos(theta), np.sin(theta),
        color="0.5", lw=1, ls="--", label="unit circle",
    )
    ax.scatter(
        mults.real, mults.imag,
        s=30, c="#E85D04", marker="o",
        edgecolors="black", linewidths=0.5, zorder=3,
        label=r"$\mu_{2T} = e^{\lambda \cdot 2T}$",
    )
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\mu_{2T})$")
    ax.set_ylabel(r"$\mathrm{Im}(\mu_{2T})$")
    ax.set_title(
        f"Floquet multipliers — 2T cover ({mults.size} eigenpairs, "
        f"2T = {T_HB:.4f})"
    )

    r_max = max(1.1, 1.05 * np.max(np.abs(mults)))
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        "results/floquet_multipliers.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig("results/floquet_multipliers.pdf", bbox_inches="tight")
    print(
        "Saved Floquet-multiplier plot → results/floquet_multipliers.{png,pdf}"
    )
    plt.show()


os._exit(0)
