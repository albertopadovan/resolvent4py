r"""
Diagnostic for the periodic-g SSM cache: check that the pointwise gauge
    w(t)^T p_j(t) = 0
holds for every order j.  Unlike the constant-g gauge (which only zeros
the T_HB-average), periodic-g is *supposed* to enforce this pointwise.

Reads data/ssm_cache_periodic_g.npz, reconstructs w(t) and p_j(t) via
IDFT over [0, T_HB), computes the transpose product on a fine grid, and
plots the traces for the first 10 non-trivial orders.
"""
# Bootstrap: this script was moved into periodic_g_manifold/.  Make the
# parent kse/ directory importable (for spatial_operators, kse_differential_equation,
# eigendecomp_kse) and chdir to it so relative data/... paths still resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import matplotlib.pyplot as plt


# ── Load periodic-g cache ─────────────────────────────────────────────
d = np.load("data/ssm_cache_periodic_g.npz")
PS_hb = d["PS_hb"]                            # (n_terms, n_harm, n)  complex
W_hb = d["W_hb"]                              # (n_harm, n, r)        complex
multiindices = d["multiindices"]              # (n_terms, r)
T_HB = float(d["T"])                          # 2·T_phys
T_phys = float(d["T_orbit_phys"])

n_terms, n_harm, n = PS_hb.shape
nf = (n_harm - 1) // 2
omega = 2.0 * np.pi / T_HB
degs = multiindices[:, 0].astype(int)         # scalar polynomial degrees (r=1)

w_hb = W_hb[:, :, 0]                          # (n_harm, n)

print(f"periodic-g cache:  n_terms = {n_terms}, n_harm = {n_harm} "
      f"(nf = {nf}), T_HB = {T_HB:.6f}")
print(f"polynomial degrees: {degs.min()}..{degs.max()}")


# ── Reconstruct w(t) and p_j(t) on a fine time grid ──────────────────
n_t = 1200
t_arr = np.linspace(0.0, T_HB, n_t, endpoint=False)
k_idx = np.arange(-nf, nf + 1)
weights = np.exp(1j * np.outer(t_arr, k_idx) * omega)     # (n_t, n_harm)

w_t = weights @ w_hb                                       # (n_t, n) complex


# ── Compute w(t)^T p_j(t)  (transpose product, pointwise) ────────────
max_abs = np.zeros(n_terms)
sup_p_norm = np.zeros(n_terms)
mean_val = np.zeros(n_terms, dtype=complex)
traces = np.zeros((n_terms, n_t), dtype=complex)

for j_idx in range(n_terms):
    p_hb = PS_hb[j_idx]                                    # (n_harm, n)
    p_t = weights @ p_hb                                   # (n_t, n)
    wtp = np.sum(w_t * p_t, axis=1)                        # (n_t,) transpose
    traces[j_idx] = wtp
    max_abs[j_idx] = float(np.max(np.abs(wtp)))
    mean_val[j_idx] = complex(np.mean(wtp))
    sup_p_norm[j_idx] = float(np.max(np.linalg.norm(p_t, axis=1)))


# ── Per-order summary ─────────────────────────────────────────────────
print(f"\n{'idx':>3s}  {'deg':>3s}  {'max_t |wᵀp_j|':>15s}  "
      f"{'mean(wᵀp_j)':>22s}  {'sup||p_j||':>13s}  {'ratio':>10s}")
for j_idx in range(min(n_terms, 20)):
    ratio = max_abs[j_idx] / max(sup_p_norm[j_idx], 1e-30)
    print(f"{j_idx:>3d}  {degs[j_idx]:>3d}  {max_abs[j_idx]:15.4e}  "
          f"{mean_val[j_idx].real:+10.3e}{mean_val[j_idx].imag:+10.3e}j  "
          f"{sup_p_norm[j_idx]:13.4e}  {ratio:10.3e}")


# ── Plot traces for the first 10 non-trivial orders ──────────────────
show = [j_idx for j_idx in range(n_terms) if degs[j_idx] >= 1][:10]
ncols = 2
nrows = (len(show) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows),
                         sharex=True)
axes = np.atleast_1d(axes).ravel()

for i, j_idx in enumerate(show):
    ax = axes[i]
    if i == 0:
        # First panel:  p_1 = v (the master mode).  Instead of showing
        # w^T v (which should be 1), plot |1 - w^T v| on a log scale so
        # the deviation from perfect pointwise biorth is visible.
        biorth_dev = np.abs(1.0 - traces[j_idx])
        ax.semilogy(t_arr / T_phys, np.maximum(biorth_dev, 1e-30),
                    "-", color="tab:red", lw=1.2,
                    label=r"$|1 - w(t)^T v(t)|$")
        ax.set_ylabel(r"$|1 - w(t)^T v(t)|$", fontsize=11)
        ax.set_title(
            fr"$j = {j_idx}$ (deg = {degs[j_idx]} $=v$),  "
            fr"$\max_t|1-w^Tv| = {biorth_dev.max():.2e}$",
            fontsize=10,
        )
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=9, loc="best")
    else:
        ax.plot(t_arr / T_phys, traces[j_idx].real, "-",
                color="tab:red", lw=1.2, label=r"$\mathrm{Re}$")
        ax.plot(t_arr / T_phys, traces[j_idx].imag, "-",
                color="tab:blue", lw=1.0, alpha=0.7, label=r"$\mathrm{Im}$")
        ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
        ax.set_ylabel(fr"$w^T p_{{{degs[j_idx]}}}(t)$", fontsize=11)
        ax.set_title(fr"$j = {j_idx}$ (deg = {degs[j_idx]}),  "
                     fr"$\max_t|w^Tp_j| = {max_abs[j_idx]:.2e}$,  "
                     fr"$\sup\|p_j\| = {sup_p_norm[j_idx]:.2e}$",
                     fontsize=10)
        ax.grid(alpha=0.2)

for k in range(len(show), len(axes)):
    axes[k].axis("off")
for k in range(len(axes) - ncols, len(axes)):
    if k < len(show):
        axes[k].set_xlabel(r"$t / T_{\rm phys}$", fontsize=11)

fig.suptitle(
    r"Pointwise gauge $w(t)^T p_j(t)$ for the periodic-$g$ SSM  "
    r"(should be 0 pointwise if periodic-$g$ is working)",
    fontsize=12,
)
fig.tight_layout()
import os
os.makedirs("results", exist_ok=True)
outpath = "results/wtp_j_periodic_g.png"
fig.savefig(outpath, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {outpath}")
plt.show()
