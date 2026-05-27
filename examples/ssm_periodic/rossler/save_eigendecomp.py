"""
Compute the Floquet eigendecomposition and the neutral-projection
operator for the Rössler periodic orbit, and cache them to disk.

Run with:
    mpiexec -n <nprocs> python save_eigendecomp.py
"""

import os

import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py

from rossler_differential_equation import RosslerPeriodic
from eigendecomp_rossler import compute_eigendecomposition, save


# ── Parameters — must match the SSM workflow ─────────────────────────────────
nf = 50
nfb = 35

n_evals = 2
krylov_dim = 100
n_neutral_strips = 6

out_path = "data/eigendecomp_cache.npz"


comm = PETSc.COMM_WORLD


# ── Load and resample the periodic orbit ─────────────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T = float(data["T"])
C_periodic_full = data["C"][:, :-1]  # drop repeated endpoint → (3, n_orbit)

n_time = 2 * (nf + nfb) + 1
C_periodic = resample(C_periodic_full, n_time, axis=1)
time_orbit = np.linspace(0, T, n_time, endpoint=False)

sigma = 1j * 2 * np.pi / T / 2

res4py.petscprint(
    comm,
    f"Rossler periodic orbit: c={c}, T={T:.6f}, "
    f"n_time={n_time}, nf={nf}, nfb={nfb}",
)


# ── Build the equation ────────────────────────────────────────────────────────
eq = RosslerPeriodic(c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit)

res4py.petscprint(comm, f"omega        = {eq.omega:.6f}")
res4py.petscprint(comm, f"HB state dim = {3 * (2 * nf + 1)}")


# ── Compute and save ─────────────────────────────────────────────────────────
L, Phi, Psi, neutral_proj = compute_eigendecomposition(
    eq,
    n_evals=n_evals,
    krylov_dim=krylov_dim,
    sigma=sigma,
    n_neutral_strips=n_neutral_strips,
)

res4py.petscprint(comm, f"# eigvals    = {L.shape[0]}")

save(out_path, L, Phi, Psi, neutral_proj, eq)


# ── Print and plot the Floquet multipliers ────────────────────────────────────
if comm.getRank() == 0:
    mults = np.exp(L * T)
    order = np.argsort(-np.abs(mults))
    print(
        f"Floquet multipliers (sorted by |μ| descending, {mults.size} total):"
    )
    for i, k in enumerate(order):
        print(
            f"  [{i:3d}] λ = {L[k].real:+.6e} {L[k].imag:+.6e}j   "
            f"μ = {mults[k].real:+.6e} {mults[k].imag:+.6e}j   "
            f"|μ| = {abs(mults[k]):.6e}"
        )

    theta = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color="0.5",
        lw=1,
        ls="--",
        label="unit circle",
    )
    ax.scatter(
        mults.real,
        mults.imag,
        s=30,
        c="#E85D04",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
        label=r"Floquet multipliers $\mu = e^{\lambda T}$",
    )
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\mu)$")
    ax.set_ylabel(r"$\mathrm{Im}(\mu)$")
    ax.set_title(f"Floquet multipliers (c={c}, T={T:.4f})")

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
