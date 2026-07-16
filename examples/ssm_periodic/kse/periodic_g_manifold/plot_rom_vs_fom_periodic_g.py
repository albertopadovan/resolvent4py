r"""
Periodic-g ROM vs FOM perturbation comparison — analogous to
plot_rom_vs_fom_s01_20T.py but using the SpectralSubmanifoldROMPeriodicG
built from data/ssm_cache_periodic_g.npz.

Prerequisite: run  python save_ssm_periodic_g.py  first to generate the
periodic-g cache at the current ν.
"""
# Bootstrap: this script was moved into periodic_g_manifold/.  Make the
# parent kse/ directory importable (for spatial_operators, kse_differential_equation,
# eigendecomp_kse) and chdir to it so relative data/... paths still resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from petsc4py import PETSc

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROMPeriodicG
from spatial_operators import linear_eigenvalues, nonlinear_rhs


# ── Load periodic-g SSM cache ─────────────────────────────────────────
cache_path = "data/ssm_cache_periodic_g.npz"
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"{cache_path} not found. Run save_ssm_periodic_g.py first."
    )
cache = np.load(cache_path)
T_HB = float(cache["T"])                    # = 2·T_phys
T_phys = float(cache["T_orbit_phys"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
nu = float(cache["nu"])
omega = 2.0 * np.pi / T_HB
rho_domain = float(cache["rho_domain"])
Lam0 = complex(cache["Lams"][0])

rom = SpectralSubmanifoldROMPeriodicG(
    multiindices=cache["multiindices"],
    Lams=cache["Lams"],
    gs=cache["gs_periodic"],                # (n_terms, n_harm_g, r)
    PS=cache["PS_hb"],
    W=cache["W_hb"],
    omega=omega,
)
rom.use_g_spline = False                    # exact IFFT — no interp artefact


# ── c_*(t) interpolant ────────────────────────────────────────────────
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(t_ext, C_ext, axis=1, kind="cubic",
                          assume_sorted=True)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)


# ── Parameters ────────────────────────────────────────────────────────
# s_0 as a fraction of ρ_domain so this script survives changes in m /
# ssm_scaling in save_ssm_periodic_g.py without needing to hand-retune.
s0_fraction = 0.1
s0 = np.array([s0_fraction * rho_domain])
tau = 2.0 * T_phys / 3.0
t_end = tau + 100.0 * T_phys
print(f"periodic-g cache:  |Λ| = {abs(Lam0.real):.4e},  T_phys = {T_phys:.4f},"
      f"  rho_domain = {rho_domain:.3f}")
print(f"tau = {tau:.4f}, s0 = {s0_fraction}·ρ_domain = {s0[0]:.4f}, "
      f"t_end = {t_end:.4f} = tau + 100 T_phys")
print(f"gs shape = {cache['gs_periodic'].shape}")


# ── ROM latent + reconstruction ────────────────────────────────────────
n_t = 8000
t_eval = np.linspace(tau, t_end, n_t)


def rom_rhs_real(t, s):
    r"""Force ds/dt to be real to kill roundoff-driven Im(s) blow-up.

    The SSM around a real orbit with real Λ has purely-real s dynamics
    by construction, but the periodic-g decode returns g_t as complex
    (via exp(+1jkωt)).  In exact arithmetic Im(g_t) = 0; in double
    precision it's ~1e-15.  Multiplied by s^j (j up to m=50) at large
    s that tiny imaginary noise becomes O(1) and blows up the RHS."""
    return rom.latent_space_dynamics(t, s).real


sol = sp.integrate.solve_ivp(
    rom_rhs_real, [tau, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-12,
)
# ROM may terminate before t_end if |s| leaves rho_domain.  Use ACTUAL
# integrated times (sol.t) rather than the requested t_eval so shapes
# stay consistent when the integration is truncated.
t_rom = sol.t
S_rom = sol.y[0]
n_t_rom = len(t_rom)
if n_t_rom < n_t:
    print(f"WARNING: ROM integration terminated at t = "
          f"{t_rom[-1] / T_phys:.3f} T_phys  ({n_t_rom} of {n_t} samples). "
          f"Last |s| = {abs(S_rom[-1]):.4f}, rho_domain = {rho_domain:.4f}")
print(f"|s_ROM(0)| = {abs(S_rom[0]):.4f},  "
      f"|s_ROM({t_rom[-1] / T_phys:.2f} T_phys)| = {abs(S_rom[-1]):.4f}")

X_ROM = np.zeros((n, n_t_rom))
CSTAR_ROM = np.zeros((n, n_t_rom))
for i in range(n_t_rom):
    CSTAR_ROM[:, i] = c_star_at(t_rom[i])
    X_ROM[:, i] = CSTAR_ROM[:, i] + rom.decode(t_rom[i], S_rom[i:i+1]).real
DELTA_ROM = X_ROM - CSTAR_ROM


# ── FOM from IC = c_*(τ) + decode(τ, s0), integrated via RK45 ─────────
delta_0 = rom.decode(tau, s0).real
x0_fom = c_star_at(tau) + delta_0
print(f"|delta_0| = {np.linalg.norm(delta_0):.4e}")


def kse_rhs(_t, q):
    return lam * q + N_fn(q)


sol_fom = sp.integrate.solve_ivp(
    kse_rhs, [tau, t_end], x0_fom,
    method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12,
)
print(f"FOM: RK45,  nfev = {sol_fom.nfev}")
t_fom = sol_fom.t
X_FOM_r = sol_fom.y
CSTAR_FOM = np.zeros((n, len(t_fom)))
for i in range(len(t_fom)):
    CSTAR_FOM[:, i] = c_star_at(t_fom[i])
DELTA_FOM = X_FOM_r - CSTAR_FOM


# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
labels = [r"$\delta_1(t)$", r"$\delta_2(t)$"]
for j in range(2):
    ax = axes[j]
    ax.plot(t_rom / T_phys, DELTA_ROM[j], "-", color="tab:red", lw=1.4,
            label=r"ROM (periodic-$g$)")
    ax.plot(t_fom / T_phys, DELTA_FOM[j], "--", color="black", lw=1.0,
            alpha=0.9, label=r"FOM")
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_ylabel(labels[j], fontsize=14)
    ax.grid(alpha=0.2)
    if j == 0:
        ax.legend(loc="best", fontsize=11)

# Third panel: ROM latent s(t) and the FOM's along-SSM projection
# s_FOM(t) = w(t)ᵀ (q(t) - c_*(t)).  On the SSM they should overlap;
# any drift is polynomial-invariance error or the FOM leaving the SSM.
print("\nProjecting FOM onto SSM latent  s_FOM(t) = w(t)ᵀ (q(t) - c_*(t)) ...")
s_fom = np.zeros(len(t_fom))
for i, t in enumerate(t_fom):
    s_fom[i] = rom.encode(t, DELTA_FOM[:, i])[0].real

ax_s = axes[2]
ax_s.plot(t_rom / T_phys, S_rom.real, "-", color="tab:red", lw=1.2,
          label=r"$s_{\rm ROM}(t)$")
if np.max(np.abs(S_rom.imag)) > 1e-10:
    ax_s.plot(t_rom / T_phys, S_rom.imag, "-", color="tab:blue", lw=1.0,
              alpha=0.7, label=r"$\mathrm{Im}\, s_{\rm ROM}(t)$")
ax_s.plot(t_fom / T_phys, s_fom, "--", color="black", lw=1.2,
          alpha=0.85, label=r"$s_{\rm FOM}(t) = w(t)^\top (q - c_*)$")
ax_s.axhline(0.0, color="0.6", lw=0.6, ls=":")
ax_s.set_ylabel(r"$s(t)$", fontsize=13)
ax_s.grid(alpha=0.2)
ax_s.legend(loc="best", fontsize=10)

axes[-1].set_xlabel(r"$t / T_{\rm phys}$", fontsize=13)
fig.suptitle(
    fr"Periodic-$g$ SSM  $\delta_j = a_j - c^*_j$,  "
    fr"$s_0 = {s0[0].real}$,  $t_{{\rm end}} = 20\,T_{{\rm phys}}$",
    fontsize=12,
)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
outpath = "results/rom_vs_fom_periodic_g.png"
fig.savefig(outpath, dpi=140, bbox_inches="tight")
print(f"Saved -> {outpath}")


# ── (a_1, a_2) phase portrait of the 2T-periodic limit cycle ──────────
# Take the last T_HB = 2·T_phys window of each trajectory — by then
# both should have relaxed onto their respective 2T attractors.
mask_rom_tail = t_rom >= (t_rom[-1] - T_HB)
mask_fom_tail = t_fom >= (t_fom[-1] - T_HB)
X_ROM_tail = X_ROM[:, mask_rom_tail]
X_FOM_tail = X_FOM_r[:, mask_fom_tail]

# Close the loop visually by repeating the first point at the end.
def _closed(X):
    return np.column_stack([X, X[:, :1]])


X_ROM_tail_c = _closed(X_ROM_tail)
X_FOM_tail_c = _closed(X_FOM_tail)

# 1T base orbit projection for context
t_1T = np.linspace(0.0, T_phys, 400, endpoint=False)
C_1T = np.array([c_star_at(t) for t in t_1T]).T

# Exact 2T orbit from HB-Newton (compute_orbit_hb_newton_2T.py) — the
# reference to which both FOM and ROM should relax.
orbit_2T_path = "data/periodic_orbit_2T.npz"
C_2T_exact = None
if os.path.exists(orbit_2T_path):
    orb2T = np.load(orbit_2T_path)
    C_2T_exact = orb2T["C"]                     # (n, n_time_save+1)
    T_2T_exact = float(orb2T["T"])
    print(f"Loaded exact 2T orbit from {orbit_2T_path}:  "
          f"T_2T = {T_2T_exact:.6f}  (vs T_HB = {T_HB:.6f}, "
          f"drift = {(T_2T_exact - T_HB)/T_HB*1e2:+.4f} %)")
else:
    print(f"WARNING: {orbit_2T_path} not found — skipping exact-orbit "
          f"overlay.  Run compute_orbit_hb_newton_2T.py to generate it.")

fig2, ax = plt.subplots(figsize=(7.5, 7.0))
ax.plot(C_1T[0], C_1T[1], "-", color="0.55", lw=1.4, alpha=0.8,
        label=r"1T base flow  $c_*(t)$")
if C_2T_exact is not None:
    ax.plot(C_2T_exact[0], C_2T_exact[1], "-", color="tab:blue", lw=2.4,
            alpha=0.85, label=r"exact $2T$ orbit  (HB-Newton)")
ax.plot(X_FOM_tail_c[0], X_FOM_tail_c[1], "-", color="black", lw=1.6,
        alpha=0.85, label=r"FOM  (last $2T_{\rm phys}$)")
ax.plot(X_ROM_tail_c[0], X_ROM_tail_c[1], "--", color="tab:red", lw=1.6,
        alpha=0.9, label=r"ROM (periodic-$g$, last $2T_{\rm phys}$)")
ax.set_xlabel(r"$a_1$", fontsize=16, labelpad=6)
ax.set_ylabel(r"$a_2$", fontsize=16, labelpad=6)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.25)
ax.legend(loc="best", fontsize=11)
ax.set_title(
    fr"2T-periodic limit cycle:  ROM vs FOM   "
    fr"($s_0 = {s0_fraction}\,\rho_{{\rm dom}}$,  "
    fr"tail window $[t_{{\rm end}} - 2T_{{\rm phys}},\, t_{{\rm end}}]$)",
    fontsize=12,
)
fig2.tight_layout()
outpath2 = "results/rom_vs_fom_periodic_g_a1_a2.png"
fig2.savefig(outpath2, dpi=140, bbox_inches="tight")
fig2.savefig(outpath2.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved -> {outpath2}")

plt.show()

PETSc.COMM_WORLD.Barrier()
os._exit(0)
