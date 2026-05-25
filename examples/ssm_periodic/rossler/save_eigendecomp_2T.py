"""
Compute the Floquet eigendecomposition for a *2T-periodic* representation
of the Rössler system around its T-periodic limit cycle, for use in a
period-doubling SSM construction.

The base flow has period T (so omega_base = 2*pi/T), but we work on the
2T-periodic basis where the period-doubled Floquet mode
(mu = exp(lambda * T) ≈ -1, i.e. lambda = a + i*omega_base/2 with
|a| << 1) folds to a *real* eigenvalue lambda_real ≈ a in the 2T
principal strip |Im lambda| <= omega_base/4.

Run with:
    mpiexec -n <nprocs> python save_eigendecomp_2T.py
"""

import os

import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py

from rossler_differential_equation import RosslerPeriodic
from eigendecomp_rossler import compute_eigendecomposition, save


# ── Parameters ───────────────────────────────────────────────────────────────
# Doubled vs the 1T config (nf=50, nfb=35) so the 2T basis covers the
# same physical-frequency range:
#   pertb_freqs go up to nf * omega_2T = nf * omega_base/2.
nf = 100
nfb = 70

n_evals = 5
krylov_dim = 100
n_neutral_strips = 4

out_path = "data/eigendecomp_cache_2T.npz"

comm = PETSc.COMM_WORLD


# ── Load the T-periodic orbit and tile to span 2T ────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T_base = float(data["T"])
C_periodic_1T = data["C"][:, :-1]               # (3, n_orbit), T-periodic

T = 2.0 * T_base
C_periodic_2T = np.tile(C_periodic_1T, (1, 2))  # (3, 2*n_orbit)

n_time = 2 * (nf + nfb) + 1
C_periodic = resample(C_periodic_2T, n_time, axis=1)
time_orbit = np.linspace(0, T, n_time, endpoint=False)

sigma = 0.0

res4py.petscprint(
    comm,
    f"Rossler 2T-periodic orbit: c={c}, T=2*T_base={T:.6f}, "
    f"n_time={n_time}, nf={nf}, nfb={nfb}",
)


# ── Build the equation (period = 2T) ─────────────────────────────────────────
eq = RosslerPeriodic(c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit)

omega_base = 2.0 * np.pi / T_base    # T-periodic base-flow frequency
res4py.petscprint(comm, f"omega_2T   (eq.omega) = {eq.omega:.6f}")
res4py.petscprint(comm, f"omega_base (= 2*eq.omega) = {omega_base:.6f}")
res4py.petscprint(comm, f"HB state dim = {3 * (2 * nf + 1)}")


# ── Compute and save ─────────────────────────────────────────────────────────
# neutral_omega = omega_base: in the 2T basis, the T-periodic neutrals
# (orbit-tangent + its harmonics) sit at lambda = i*k*omega_base for
# integer k. We override eq.omega (= omega_base/2) so the strip loop
# does not accidentally probe the period-doubling sector at i*omega_2T.
L, Phi, Psi, neutral_proj = compute_eigendecomposition(
    eq,
    n_evals=n_evals,
    krylov_dim=krylov_dim,
    sigma=sigma,
    n_neutral_strips=n_neutral_strips,
    neutral_omega=omega_base,
)

res4py.petscprint(comm, f"# eigvals    = {L.shape[0]}")

save(out_path, L, Phi, Psi, neutral_proj, eq)


# ── Print and plot the Floquet multipliers ───────────────────────────────────
if comm.getRank() == 0:
    mults_2T = np.exp(L * T)             # 2T multipliers (T = 2*T_base)
    mults_1T = np.exp(L * T_base)        # equivalent 1T multipliers
    order = np.argsort(-np.abs(mults_2T))

    print(f"Floquet multipliers ({mults_2T.size} total, sorted by |μ_2T| desc.):")
    for i, k in enumerate(order):
        print(
            f"  [{i:3d}] λ = {L[k].real:+.6e} {L[k].imag:+.6e}j   "
            f"μ_2T = {mults_2T[k].real:+.6e} {mults_2T[k].imag:+.6e}j   "
            f"μ_1T = {mults_1T[k].real:+.6e} {mults_1T[k].imag:+.6e}j   "
            f"|μ_2T| = {abs(mults_2T[k]):.6e}"
        )

    theta = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(np.cos(theta), np.sin(theta), color="0.5", lw=1, ls="--",
            label="unit circle")
    ax.scatter(mults_2T.real, mults_2T.imag, s=30, c="#E85D04", marker="o",
               edgecolors="black", linewidths=0.5, zorder=3,
               label=r"$\mu_{2T} = e^{\lambda \cdot 2T_{\rm base}}$")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\mu)$")
    ax.set_ylabel(r"$\mathrm{Im}(\mu)$")
    ax.set_title(f"Floquet multipliers, 2T basis (c={c}, T_base={T_base:.4f})")
    ax.legend(loc="best", fontsize=9)
    r_max = max(1.1, 1.05 * np.max(np.abs(mults_2T)))
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig("results/floquet_multipliers_2T.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/floquet_multipliers_2T.pdf", bbox_inches="tight")
    print("Saved Floquet-multiplier plot → results/floquet_multipliers_2T.{png,pdf}")
    plt.show()


os._exit(0)
