r"""
Time-periodic-g variant of ``save_ssm_2T.py`` for the Rössler system in
the 2T-periodic (period-doubling) representation.

Uses :class:`SpectralSubmanifoldPeriodicG` (pointwise-biorthogonal
projection + T_HB-periodic g_j(t) coefficients) so the along-SSM
dynamics carry their full time-varying structure rather than only the
T_HB-averaged DC part.

Cache layout differs from ``save_ssm_2T.py`` only in the ``g``
coefficients:
    constant-g:  gs shape = (n_terms, r)             — scalar per multi-index
    periodic-g:  gs shape = (n_terms, n_harm_g, r)   — HB coefficients of g_j(t)

Prereqs (mirrors constant-g pipeline):
    data/periodic_orbit.npz         (compute_periodic_orbit.py)
    data/eigendecomp_cache_2T.npz   (save_eigendecomp_2T.py)

Run with ``mpirun -n <nprocs> python save_ssm_2T_periodic_g.py``.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from petsc4py import PETSc
import resolvent4py as res4py

from rossler_differential_equation import RosslerPeriodic
from eigendecomp_rossler import load as load_eigendecomp
from resolvent4py.spectral_submanifold import SpectralSubmanifoldPeriodicG


# ── Parameters ────────────────────────────────────────────────────────
# Keep in sync with save_ssm_2T.py where possible so both caches use
# the same eigendecomposition and Floquet basis.
r = 1
m = 12                                 # SSM polynomial order — kept LOW
                                      # on purpose: periodic-g's time-
                                      # varying g_j(t) provides most of
                                      # the representative power, so
                                      # high m pollutes the latent
                                      # dynamics without adding fidelity.
ssm_scaling = 0.2
manifold_tol = 1e-2
latent_space_components = None

# Periodic-g specific:
#   nf_g = 2·nf_g + 1 Fourier modes retained for g_j(t).  Setting it
#          equal to nf keeps the full internal band; consistent with the
#          KSE periodic-g setup.
#   tol_biorth_pt: solver rescales v(t) pointwise (once) if the
#          max_t |<w(t), v(t)>_pt − 1| exceeds this.
tol_biorth_pt = 1e-6

comm = PETSc.COMM_WORLD


# ── Load periodic orbit and tile to 2T ────────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T_base = float(data["T"])                       # 1T (base) period
nf_orbit = int(data["nf"])
C_periodic_1T = data["C"][:, :-1]                # last column duplicates first

nfb = 2 * nf_orbit
nf = nfb + 70
nf_g = 20

T = 2.0 * T_base                                 # 2T embedding
omega_HB = 2.0 * np.pi / T

# Dealiased quadratic: need n_time >= 4·nf + 2 over the 2T window.
# Since n_time = 2·n_time_per_T, use n_time_per_T = 2·nf + 2 (a
# comfortable margin above 2·nf + 1).  Periodic-g retains aliased
# high-|k| content in g_l(t) that feeds forward through
# _hb_convolve_axpy at every order, so we can't get away with the
# tighter save_ssm_2T.py setting n_time = 2·(nf+nfb)+1 when nf > nfb.
n_time_per_T = 4 * nf + 2
n_time = 2 * n_time_per_T

C_periodic_one_T = resample(C_periodic_1T, n_time_per_T, axis=1)
C_periodic = np.concatenate([C_periodic_one_T, C_periodic_one_T], axis=1)
time_orbit = np.linspace(0.0, T, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"[periodic-g] Rossler 2T-HB:  c = {c},  "
    f"T_base = {T_base:.6f},  T = 2·T_base = {T:.6f}",
)
res4py.petscprint(
    comm,
    f"  n_time = {n_time},  nf = {nf},  nfb = {nfb},  nf_g = {nf_g}  "
    f"(g_j has 2·nf_g + 1 = {2*nf_g + 1} Fourier modes)",
)


# ── Build eq + load Floquet eigendecomposition ────────────────────────
omegas_one_sided = omega_HB * np.arange(nf + 1)
periodic_diffeq = (omegas_one_sided, time_orbit, True)   # True = period-doubling

eq = RosslerPeriodic(
    c=c,
    nf=nf,
    nfb=nfb,
    c_star=C_periodic,
    time=time_orbit,
    periodic_diffeq=periodic_diffeq,
    use_time_stepping=True,
    ts_verbose=1,
)

eq.L, eq.Phi, eq.Psi, eq._neutral_proj = load_eigendecomp(
    "data/eigendecomp_cache_2T.npz",
    comm=comm,
)

idces = np.arange(r, dtype=np.int32)
L = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

# HB-averaged biorthonormalisation:  V ← V · (W^H V)^{-1}
M = V.dot(W)
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
M2 = V.dot(W)
biorth_err_after = np.linalg.norm(M2.getDenseArray() - np.eye(r))
M2.destroy()
res4py.petscprint(
    comm,
    f"Biorthonormalisation |Psi^* Phi - I|:  before = {biorth_err_before:.4e},  "
    f"after = {biorth_err_after:.4e}",
)


# ── Solve SSM with periodic g_j(t) ────────────────────────────────────
SSM = SpectralSubmanifoldPeriodicG(
    eq, r, m, latent_space_components=latent_space_components,
)
SSM.solve(
    V, W, L,
    scaling=ssm_scaling, verbose=1,
    nf_g=nf_g, tol_biorth_pt=tol_biorth_pt,
)

res4py.petscprint(comm, f"[periodic-g] Λ = {SSM.Lams}")

R_conv, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
percent_domain, est_error = res4py.proper_radius(manifold_tol, intercept, m)
rho_domain = percent_domain * R_conv
res4py.petscprint(
    comm,
    f"[periodic-g] R = {R_conv:.3f},  rho_domain = {rho_domain:.3f}  "
    f"(est error {est_error:.3f})",
)


# ── Plot coefficient decay ────────────────────────────────────────────
if comm.getRank() == 0:
    valid = coeff_sums > 0
    fit_line = 10.0 ** (slope * orders + intercept)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(
        orders[valid], coeff_sums[valid], "o",
        color="#E85D04", markeredgecolor="black", markeredgewidth=0.5,
        label=r"$C_k = \sum_{|j|=k} \|p_j\|_1$",
    )
    ax.semilogy(
        orders, fit_line, "--", color="0.3",
        label=fr"fit: slope$={slope:.3f}$, $R={R_conv:.3f}$",
    )
    ax.set_xlabel(r"order $k$")
    ax.set_ylabel(r"$C_k$")
    ax.set_title(
        r"Rössler 2T SSM — geometric decay (periodic-$g$)",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", ls=":", lw=0.5)
    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        "results/ssm_geometric_decay_2T_periodic_g.png",
        dpi=300, bbox_inches="tight",
    )
    fig.savefig(
        "results/ssm_geometric_decay_2T_periodic_g.pdf",
        bbox_inches="tight",
    )
    print(
        "Saved -> results/ssm_geometric_decay_2T_periodic_g.{png,pdf}"
    )


# ── Gather HB arrays to numpy ─────────────────────────────────────────
n = 3                                                # Rössler state dim
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

# gs is a list of length-(2·nf_g+1) complex arrays; stack into
# (n_terms, n_harm_g, r).  For r=1 the inner list has shape (n_harm_g,).
gs_periodic = np.stack(SSM.gs, axis=0)               # (n_terms, n_harm_g)
gs_periodic = gs_periodic[:, :, None]                # (n_terms, n_harm_g, r=1)


# ── Save ──────────────────────────────────────────────────────────────
if comm.getRank() == 0:
    os.makedirs("data", exist_ok=True)
    outpath = "data/ssm_cache_2T_periodic_g.npz"
    np.savez_compressed(
        outpath,
        # SSM HB arrays
        PS_hb=PS_hb,
        W_hb=W_hb,
        V_neut_hb=V_neut_hb,
        W_neut_hb=W_neut_hb,
        multiindices=multiindices,
        Lams=Lams,
        gs_periodic=gs_periodic,                     # (n_terms, n_harm_g, r)
        # Metadata
        c=c,
        n=n,
        T=T,                                         # = 2·T_base (HB period)
        T_base=T_base,
        nf=nf, nfb=nfb, nf_g=nf_g,
        r=r, m=m, rho_domain=rho_domain,
        C_periodic=C_periodic,
        time_orbit=time_orbit,
    )
    print(f"[periodic-g] Saved SSM cache -> {outpath}")
    print(
        f"  gs_periodic shape = {gs_periodic.shape}   "
        f"(n_terms={gs_periodic.shape[0]}, "
        f"n_harm_g={gs_periodic.shape[1]}, r={gs_periodic.shape[2]})"
    )

comm.Barrier()
os._exit(0)
