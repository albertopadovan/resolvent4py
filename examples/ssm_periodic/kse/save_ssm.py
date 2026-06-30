"""
Compute the time-periodic SSM for the KSE and dump every piece needed
to reproduce the ROM without PETSc to ``data/ssm_cache.npz``.

The companion script ``debug_off_manifold.py`` loads this cache and is
pure numpy — no PETSc/MPI — so the debug iteration is fast.

Run with ``mpirun -n <nprocs> python save_ssm.py``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from kse_differential_equation import KuramotoSivashinskyPeriodic
from eigendecomp_kse import load as load_eigendecomp

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold

# ── Parameters (keep in sync with demonstrate_ssm.py) ───────────────────────
# nf, nfb are derived below from the HB-Newton ``nf_HB`` stored in
# data/periodic_orbit.npz, with the SAME formula save_eigendecomp.py uses
# (nfb = 2·nf_HB, nf = nfb + 20) — so the two stay in lock-step automatically.
r = 1
m = 40
ssm_scaling = 0.02
manifold_tol = 1e-2
# Normal-form parametrisation: retain ONLY orders 1 (linear) and 3 (cubic)
# in the latent-space dynamics — the pitchfork normal form.  Every other
# order's contribution is absorbed into the manifold map P(s), so the
# bifurcating fixed point s* = √(-λ/g_3) lives inside the SSM's
# convergence radius and the higher-order coefficients are exactly zero.
latent_space_components = None

comm = PETSc.COMM_WORLD


# %% Load periodic orbit and embed into the 2T-HB time grid
# (same setup as save_eigendecomp.py — orbit is T-periodic but the HB
# framework treats it as 2T-periodic so the period-doubling mode lives
# cleanly at odd k of the doubled-frequency grid)

data = np.load("data/periodic_orbit.npz")
nu = float(data["nu"])
n = int(data["n"])
n_pts = int(data["n_pts"])
T = float(data["T"])
nf_HB = int(data["nf"])                 # HB-Newton's T-periodic nf
C_orbit = data["C"]
C_periodic_full = C_orbit[:, :-1]

# 2T-HB bandwidth — must match save_eigendecomp.py
nfb = 2 * nf_HB
nf = nfb + 20

T_HB = 2.0 * T                          # HB period = 2T
omega_orbit = 2.0 * np.pi / T           # ω of the T-orbit
omega_HB = 2.0 * np.pi / T_HB           # = ω/2 — fundamental of 2T-HB

# n_time samples over [0, 2T): each T contains n_time/2 points, second T
# is an exact copy of the first so the FFT sees exactly T-periodic data
# embedded in the 2T-cover.
n_time_per_T = nf + nfb + 1
n_time = 2 * n_time_per_T

C_periodic_one_T = resample(C_periodic_full, n_time_per_T, axis=1)
C_periodic = np.concatenate([C_periodic_one_T, C_periodic_one_T], axis=1)
time_orbit = np.linspace(0.0, T_HB, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"KSE periodic orbit (2T-HB):  nu={nu}, n={n}, T={T:.6f},  "
    f"T_HB=2T={T_HB:.6f},  ω_HB={omega_HB:.6f}",
)
res4py.petscprint(
    comm,
    f"  n_time={n_time}  (= 2 × {n_time_per_T}),  nf={nf},  nfb={nfb}",
)


# %% Build eq and compute SSM

omegas_one_sided = omega_HB * np.arange(nf + 1)
periodic_diffeq = (omegas_one_sided, time_orbit, False)

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

# Load the precomputed Floquet eigendecomposition and neutral projection.
eq.L, eq.Phi, eq.Psi, eq._neutral_proj = load_eigendecomp(
    "data/eigendecomp_cache.npz",
    comm=comm,
)

# With n_neutral_strips > 0 in save_eigendecomp.py, the orbit-tangent
# neutral subspace is deflated before the eigensolve, so the period-
# doubling mode (Floquet iω/2 → λ_2T ≈ 0) is the leading eigenvalue
# at index 0.  Pick it as the SSM master.
idces = np.arange(r, dtype=np.int32)
L = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

# Re-biorthonormalise V so that Psi^* Phi = I.  The save_eigendecomp.py
# call to `enforce_complex_conjugacy` overwrites each column of Phi/Psi
# with its conj-symmetric projection, which can perturb the biorthonormal
# pairing slightly.  Compute M = W^H V (= Psi^* Phi for the chosen
# columns) and rescale V ← V · M^{-1} so M becomes I.
M = V.dot(W)                              # SLEPc returns W^H V, shape (r, r)
M_arr = M.getDenseArray().copy()
biorth_err_before = np.linalg.norm(M_arr - np.eye(r))
if r == 1:
    V.scale(1.0 / complex(M_arr[0, 0]))
else:
    M_inv = np.linalg.inv(M_arr)
    M_inv_mat = PETSc.Mat().createDense(
        [r, r], array=M_inv.astype(PETSc.ScalarType), comm=PETSc.COMM_SELF,
    )
    V.multInPlace(M_inv_mat, 0, r)
    M_inv_mat.destroy()
M.destroy()
# Verify
M2 = V.dot(W)
biorth_err_after = np.linalg.norm(M2.getDenseArray() - np.eye(r))
M2.destroy()
res4py.petscprint(
    comm,
    f"Biorthonormalisation: |Psi^* Phi - I|  before = {biorth_err_before:.4e},  "
    f"after = {biorth_err_after:.4e}",
)

SSM = SpectralSubmanifold(
    eq, r, m, latent_space_components=latent_space_components
)
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
    ax.semilogy(
        orders[valid],
        coeff_sums[valid],
        "o",
        color="#E85D04",
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=r"$C_k = \sum_{|j|=k} \|p_j\|_1$",
    )
    ax.semilogy(
        orders,
        fit_line,
        "--",
        color="0.3",
        label=rf"fit: slope$={slope:.3f}$, $R={R:.3f}$",
    )
    ax.set_xlabel(r"order $k$")
    ax.set_ylabel(r"$C_k$")
    ax.set_title("Geometric decay of SSM polynomial coefficients")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.5)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        "results/ssm_geometric_decay.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig("results/ssm_geometric_decay.pdf", bbox_inches="tight")
    print(
        "Saved SSM geometric-decay plot → results/ssm_geometric_decay.{png,pdf}"
    )
    plt.show()


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
gs = np.asarray(SSM.gs)  # (n_terms, r)
conj_to_linear_dynamics = bool(SSM.conj_to_linear_dynamics)


# %% Save

if comm.getRank() == 0:
    os.makedirs("data", exist_ok=True)
    save_kwargs = dict(
        # SSM HB arrays
        PS_hb=PS_hb,
        W_hb=W_hb,
        V_neut_hb=V_neut_hb,
        W_neut_hb=W_neut_hb,
        multiindices=multiindices,
        Lams=Lams,
        gs=gs,
        conj_to_linear_dynamics=conj_to_linear_dynamics,
        # Parameters — note T is the HB period (= 2 × T_orbit_phys), since
        # the saved C_periodic / time_orbit cover [0, 2T_phys).
        nu=nu,
        n=n,
        n_pts=n_pts,
        T=T_HB,
        T_orbit_phys=T,
        nf=nf,
        nfb=nfb,
        r=r,
        m=m,
        rho_domain=rho_domain,
        # Periodic orbit (over 2T)
        C_periodic=C_periodic,
        time_orbit=time_orbit,
    )
    np.savez_compressed("data/ssm_cache.npz", **save_kwargs)
    print("Saved SSM cache to data/ssm_cache.npz")

comm.Barrier()
os._exit(0)
