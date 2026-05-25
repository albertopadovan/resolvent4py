"""
Compute the time-periodic SSM for the Rössler system and dump every
piece needed to reproduce the ROM without PETSc to ``data/ssm_cache.npz``.

Run with ``mpirun -n <nprocs> python save_ssm.py``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from rossler_differential_equation import RosslerPeriodic
from eigendecomp_rossler import load as load_eigendecomp

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold

# ── Parameters (keep in sync with demonstrate_ssm.py) ───────────────────────
nf = 50
nfb = 35
r = 1
m = 15
ssm_scaling = 0.1
manifold_tol = 1e-2

comm = PETSc.COMM_WORLD


# %% Load periodic orbit

data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T = float(data["T"])
C_periodic_full = data["C"][:, :-1]   # drop repeated endpoint → (3, n_orbit)

n_time = 2 * (nf + nfb) + 1
C_periodic = resample(C_periodic_full, n_time, axis=1)
time_orbit = np.linspace(0, T, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"Rossler periodic orbit: c={c}, T={T:.6f}, "
    f"n_time={n_time}, nf={nf}, nfb={nfb}",
)


# %% Build eq and compute SSM

# periodic_diffeq triggers the PeriodicDifferentialEquation mixin in
# DifferentialEquation.__new__, which wraps the per-time evaluate_*
# backends into HB-acting versions that SpectralSubmanifold needs.
omega = 2.0 * np.pi / T
omegas_one_sided = omega * np.arange(nf + 1)
periodic_diffeq = (omegas_one_sided, time_orbit, False)

eq = RosslerPeriodic(
    c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit,
    periodic_diffeq=periodic_diffeq,
)

eq.L, eq.Phi, eq.Psi, eq._neutral_proj = load_eigendecomp(
    "data/eigendecomp_cache.npz", comm=comm,
)

idces = np.arange(r, dtype=np.int32)
L = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

SSM = SpectralSubmanifold(eq, r, m)
SSM.solve(V, W, L, scaling=ssm_scaling, verbose=1)

res4py.petscprint(comm, f"Dominant eigenvalues: {SSM.Lams}")

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
percent_domain, est_error = res4py.proper_radius(manifold_tol, intercept, m)
rho_domain = percent_domain * R
res4py.petscprint(
    comm,
    f"Convergence radius R={R:.3f}, rho_domain={rho_domain:.3f} "
    f"(est error {est_error:.3f})",
)


# ── Plot the geometric decay of the SSM coefficients ─────────────────────────
if comm.getRank() == 0:
    valid = coeff_sums > 0
    fit_line = 10.0 ** (slope * orders + intercept)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(orders[valid], coeff_sums[valid], "o", color="#E85D04",
                markeredgecolor="black", markeredgewidth=0.5,
                label=r"$C_k = \sum_{|j|=k} \|p_j\|_1$")
    ax.semilogy(orders, fit_line, "--", color="0.3",
                label=fr"fit: slope$={slope:.3f}$, $R={R:.3f}$")
    ax.set_xlabel(r"order $k$")
    ax.set_ylabel(r"$C_k$")
    ax.set_title("Geometric decay of SSM polynomial coefficients")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.5)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig("results/ssm_geometric_decay.png", dpi=300, bbox_inches="tight")
    fig.savefig("results/ssm_geometric_decay.pdf", bbox_inches="tight")
    print("Saved SSM geometric-decay plot → results/ssm_geometric_decay.{png,pdf}")
    plt.show()


# %% Gather HB arrays to numpy

n = 3  # Rössler state dimension
n_harmonics = 2 * nf + 1


def _gather_hb_vec(vec_hb):
    vec_seq = res4py.distributed_to_sequential_vector(vec_hb)
    arr = vec_seq.getArray().copy().reshape(n_harmonics, n)
    vec_seq.destroy()
    return arr


def _gather_hb_bv(bv):
    ncols = bv.getSizes()[-1]
    out = np.zeros((n_harmonics, n, ncols), dtype=complex)
    for j in range(ncols):
        col = bv.getColumn(j)
        out[:, :, j] = _gather_hb_vec(col)
        bv.restoreColumn(j, col)
    return out


PS_hb = np.stack([_gather_hb_vec(p) for p in SSM.ps], axis=0)
W_hb = _gather_hb_bv(SSM.W)
V_neut_hb = _gather_hb_bv(eq._neutral_proj.L.L.U)
W_neut_hb = _gather_hb_bv(eq._neutral_proj.L.L.V)

multiindices = np.array(SSM.ssm_multiindices, dtype=np.int64)
Lams = np.asarray(SSM.Lams)
gs = np.asarray(SSM.gs)
conj_to_linear_dynamics = bool(SSM.conj_to_linear_dynamics)


# %% Save

if comm.getRank() == 0:
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/ssm_cache.npz",
        PS_hb=PS_hb,
        W_hb=W_hb,
        V_neut_hb=V_neut_hb,
        W_neut_hb=W_neut_hb,
        multiindices=multiindices,
        Lams=Lams,
        gs=gs,
        conj_to_linear_dynamics=conj_to_linear_dynamics,
        c=c, n=n, T=T,
        nf=nf, nfb=nfb, r=r, m=m,
        rho_domain=rho_domain,
        C_periodic=C_periodic,
        time_orbit=time_orbit,
    )
    print("Saved SSM cache to data/ssm_cache.npz")

comm.Barrier()
os._exit(0)
