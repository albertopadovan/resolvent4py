"""
Compute the time-periodic SSM for the KSE and dump every piece needed
to reproduce the ROM without PETSc to ``data/ssm_cache.npz``.

The companion script ``debug_off_manifold.py`` loads this cache and is
pure numpy — no PETSc/MPI — so the debug iteration is fast.

Run with ``mpirun -n <nprocs> python save_ssm.py``.
"""

import os
import numpy as np
from kse_differential_equation import KuramotoSivashinskyPeriodic

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold

# ── Parameters (keep in sync with demonstrate_ssm.py) ───────────────────────
nf = 17
nfb = 12
r = 2
m = 9
ssm_scaling = 0.2
manifold_tol = 1e-2

comm = PETSc.COMM_WORLD


# %% Load periodic orbit

data = np.load("data/periodic_orbit.npz")
nu = float(data["nu"])
n = int(data["n"])
n_pts = int(data["n_pts"])
T = float(data["T"])
C_orbit = data["C"]
C_periodic = C_orbit[:, :-1]
n_time = C_periodic.shape[1]
time_orbit = np.linspace(0, T, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"KSE periodic orbit: nu={nu}, n={n}, T={T:.6f}, "
    f"n_time={n_time}, nf={nf}, nfb={nfb}",
)


# %% Build eq and compute SSM

eq = KuramotoSivashinskyPeriodic(
    n=n, nu=nu, nf=nf, nfb=nfb,
    c_star=C_periodic, time=time_orbit, n_pts=n_pts,
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


# %% Gather HB arrays to numpy

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

multiindices = np.array(SSM.ssm_multiindices, dtype=np.int64)  # (n_terms, r)
Lams = np.asarray(SSM.Lams)
gs = np.asarray(SSM.gs)                                        # (n_terms, r)
conj_to_linear_dynamics = bool(SSM.conj_to_linear_dynamics)


# %% Save

if comm.getRank() == 0:
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/ssm_cache.npz",
        # SSM HB arrays
        PS_hb=PS_hb,
        W_hb=W_hb,
        V_neut_hb=V_neut_hb,
        W_neut_hb=W_neut_hb,
        multiindices=multiindices,
        Lams=Lams,
        gs=gs,
        conj_to_linear_dynamics=conj_to_linear_dynamics,
        # Parameters
        nu=nu, n=n, n_pts=n_pts, T=T,
        nf=nf, nfb=nfb, r=r, m=m,
        rho_domain=rho_domain,
        # Periodic orbit
        C_periodic=C_periodic,
        time_orbit=time_orbit,
    )
    print("Saved SSM cache to data/ssm_cache.npz")

comm.Barrier()
os._exit(0)
