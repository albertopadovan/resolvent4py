r"""
Off-manifold ROM vs FOM comparison, constant-g SSM.

FOM IC:  x_0 = c_*(τ) + decode(τ, s_0) + ε · η,   η ~ N(0, I_n).
ROM IC:  s(τ) = w(τ)ᵀ · (x_0 − c_*(τ))   (encode the off-manifold IC).

Both trajectories start from the same physical IC after the ROM
projects onto the manifold via the biorthogonal encoder.  If the SSM
is attracting, the FOM's transverse component decays on a timescale
~ 1/|Λ| while the ROM tracks the SSM-restricted flow.  Plots δ_j(t) =
a_j(t) − c*_j(t) for j = 1, 2:  initial gap = (I − v·wᵀ) · ε · η
(the transverse part of the kick); any long-term separation signals
ROM error along the manifold direction.
"""
import os
from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM
from spatial_operators import linear_eigenvalues, nonlinear_rhs


# ── Load SSM cache + build ROM ─────────────────────────────────────────
cache = np.load("data/ssm_cache.npz")
T_HB = float(cache["T"])
T_phys = float(cache["T_orbit_phys"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
nu = float(cache["nu"])
omega = 2.0 * np.pi / T_HB
rho_domain = float(cache["rho_domain"])
Lam0 = complex(cache["Lams"][0])

rom = SpectralSubmanifoldROM(
    multiindices=cache["multiindices"],
    Lams=cache["Lams"],
    gs=cache["gs"],
    PS=cache["PS_hb"],
    W=cache["W_hb"],
    conj_to_linear_dynamics=bool(cache["conj_to_linear_dynamics"]),
    omega=omega,
)


# ── c_*(t) interpolant ─────────────────────────────────────────────────
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(
    t_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)


# ── Parameters ─────────────────────────────────────────────────────────
seed = 0
eps = 0.05
s0 = np.array([1.0 + 0.0j], dtype=complex)
tau = 0.5 * T_phys
t_end = tau + 30.0 * T_phys


# ── Off-manifold IC: kick the FOM in a random state-space direction ──
rng = np.random.default_rng(seed)
delta_on = rom.decode(tau, s0).real                       # on-manifold Δ
kick = eps * rng.standard_normal(n)
c_at_tau = c_star_at(tau)
x0_fom = c_at_tau + delta_on + kick

print(f"|Λ| = {abs(Lam0.real):.4e},  T_phys = {T_phys:.4f},  "
      f"rho_domain = {rho_domain:.3f}")
print(f"tau = {tau:.4f} = T_phys/2,  s0 = {s0[0].real},  "
      f"t_end = tau + 30 T_phys")
print(f"eps = {eps},  seed = {seed}")
print(f"||decode(τ,s_0)|| = {np.linalg.norm(delta_on):.4e}")
print(f"||ε · η||         = {np.linalg.norm(kick):.4e}  "
      f"(transverse kick / on-manifold amplitude = "
      f"{np.linalg.norm(kick) / max(np.linalg.norm(delta_on), 1e-30):.3f})")


# ── ROM latent: project the off-manifold IC via encode = w(τ)ᵀ · Δ ────
s_rom_0 = rom.encode(tau, x0_fom - c_at_tau)
print(f"encode(τ, Δ_FOM) = {s_rom_0[0]:+.6f}   "
      f"(vs s_0 = {s0[0]:+.6f}, gap = {abs(s_rom_0[0] - s0[0]):.4e})")

n_t = 8000
t_eval = np.linspace(tau, t_end, n_t)
sol_rom = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [tau, t_end], s_rom_0,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-14,
)
S_rom = sol_rom.y[0]
print(f"ROM: |s(τ)| = {abs(S_rom[0]):.4f},  "
      f"|s(t_end)| = {abs(S_rom[-1]):.4f}")

X_ROM = np.zeros((n, n_t))
CSTAR = np.zeros((n, n_t))
for i in range(n_t):
    CSTAR[:, i] = c_star_at(t_eval[i])
    X_ROM[:, i] = CSTAR[:, i] + rom.decode(t_eval[i], S_rom[i:i+1]).real
DELTA_ROM = X_ROM - CSTAR


# ── FOM from the kicked IC ─────────────────────────────────────────────
def kse_rhs(_t, q):
    return lam * q + N_fn(q)


sol_fom = sp.integrate.solve_ivp(
    kse_rhs, [tau, t_end], x0_fom,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-12,
)
print(f"FOM: RK45,  nfev = {sol_fom.nfev}")
X_FOM_r = sol_fom.y
DELTA_FOM = X_FOM_r - CSTAR


# ── Plot δ_j(t) for j = 1, 2 ───────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
labels = [r"$\delta_1(t)$", r"$\delta_2(t)$"]
for j in range(2):
    ax = axes[j]
    ax.plot(t_eval / T_phys, DELTA_ROM[j], "-", color="tab:red", lw=1.4,
            label=r"ROM ($s(\tau) = w(\tau)^\top \Delta_{\rm FOM}$)")
    ax.plot(t_eval / T_phys, DELTA_FOM[j], "--", color="black", lw=1.0,
            alpha=0.9, label=r"FOM (off-manifold IC)")
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_ylabel(labels[j], fontsize=14)
    ax.grid(alpha=0.2)
    if j == 0:
        ax.legend(loc="best", fontsize=11)
axes[-1].set_xlabel(r"$t / T_{\rm phys}$", fontsize=13)
fig.suptitle(
    fr"Off-manifold ROM vs FOM:  "
    fr"$x_0^{{\rm FOM}} = c_* + P(\tau, s_0) + \varepsilon\,\eta$,   "
    fr"$s_0 = {s0[0].real}$, $\varepsilon = {eps}$, seed $= {seed}$",
    fontsize=12,
)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
outpath = "results/rom_vs_fom_off_manifold.png"
fig.savefig(outpath, dpi=140, bbox_inches="tight")
print(f"Saved -> {outpath}")
plt.show()
