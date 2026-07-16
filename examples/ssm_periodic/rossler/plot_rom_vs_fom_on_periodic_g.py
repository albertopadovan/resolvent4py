r"""
On-manifold ROM vs FOM comparison for the Rössler 2T-periodic SSM
using the time-periodic-g formulation.

FOM IC:  x_0 = c_*(τ) + decode(τ, s_0)   (exactly on the SSM)
ROM IC:  s(τ) = s_0
Both integrated with RK45 at matched tolerances.

Plots δ_j(t) = a_j(t) − c*_j(t)  for j = 1..3, ROM (red) vs FOM (black),
and a 3D phase portrait of the full state a(t) with the T-periodic
base orbit for reference.

Prereq:  data/ssm_cache_2T_periodic_g.npz   (save_ssm_2T_periodic_g.py)
"""
import os
from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from petsc4py import PETSc

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROMPeriodicG
from rossler_rhs import perturbation_linear_action, quadratic_bilinear


# ── Load periodic-g SSM cache ─────────────────────────────────────────
cache_path = "data/ssm_cache_2T_periodic_g.npz"
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"{cache_path} not found. Run save_ssm_2T_periodic_g.py first."
    )
cache = np.load(cache_path)
T_HB = float(cache["T"])                    # = 2·T_base
T_base = float(cache["T_base"])
c = float(cache["c"])
n = int(cache["n"])
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

print(f"periodic-g cache (Rössler):  c = {c},  |Λ| = {abs(Lam0.real):.4e},")
print(f"  T_base = {T_base:.4f},  T_HB = 2·T_base = {T_HB:.4f},  "
      f"rho_domain = {rho_domain:.3f}")


# ── c_*(t) interpolant (2T-tiled 1T orbit) ────────────────────────────
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(
    t_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


# ── FOM perturbation RHS: v̇ = A(t) v + B(v, v) ──────────────────────
def fom_perturbation_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


# ── Parameters ─────────────────────────────────────────────────────────
s0_fraction = 0.3
s0 = s0_fraction * rho_domain * np.array([1.0 + 0.0j], dtype=complex)
tau = 0.5 * T_base                       # start at T_base/2
n_periods = 60                          # integrate for 60 T_base
t_end = tau + n_periods * T_base
rtol, atol = 1e-11, 1e-13

print(f"On-manifold IC:  s_0 = {s0_fraction}·ρ_domain = {s0[0].real:.4f}")
print(f"τ = T_base/2 = {tau:.4f},  t_end = τ + {n_periods}·T_base "
      f"= {t_end:.4f}")


# ── ROM latent (on-manifold, starts at s_0) ────────────────────────────
n_t = 40000
t_eval = np.linspace(tau, t_end, n_t)


def rom_rhs_real(t, s):
    """Kill roundoff-driven Im(s) growth from complex exp weights."""
    return rom.latent_space_dynamics(t, s).real


print("\nIntegrating ROM (Radau — implicit) ...")
# Implicit stepper for the ROM: RK45 blows up near s ≈ s* because the
# truncated polynomial's tail dominates there; Radau's L-stability can
# handle the stiff blow-up regime robustly.  max_step caps the stride
# so the first few periods don't jump straight past R_g.
sol_rom = sp.integrate.solve_ivp(
    rom_rhs_real, [tau, t_end], s0.real,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    max_step=T_base / 20.0,
)
S_rom = sol_rom.y[0]
t_rom = sol_rom.t
n_t_rom = len(t_rom)
if n_t_rom < n_t:
    print(f"  ⚠ ROM terminated early at t/T_base = "
          f"{t_rom[-1] / T_base:.3f}, last |s| = {abs(S_rom[-1]):.3f} "
          f"(rho_domain = {rho_domain:.3f})")

V_rom = np.zeros((n, n_t_rom))
CSTAR_ROM = np.zeros((n, n_t_rom))
for i in range(n_t_rom):
    CSTAR_ROM[:, i] = c_star_at(t_rom[i])
    V_rom[:, i] = rom.decode(t_rom[i], S_rom[i:i+1]).real


# ── FOM from IC = c_*(τ) + decode(τ, s_0) ─────────────────────────────
v0_fom = rom.decode(tau, s0).real
print(f"|v_0| = ||decode(τ, s_0)|| = {np.linalg.norm(v0_fom):.4f}")
print("Integrating FOM (perturbation form) ...")
sol_fom = sp.integrate.solve_ivp(
    fom_perturbation_rhs, [tau, t_end], v0_fom,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
)
V_fom = sol_fom.y
t_fom = sol_fom.t
CSTAR_FOM = np.zeros((n, len(t_fom)))
for i in range(len(t_fom)):
    CSTAR_FOM[:, i] = c_star_at(t_fom[i])
print(f"  FOM: RK45,  nfev = {sol_fom.nfev}")


# ── Plot perturbations δ_j(t) ────────────────────────────────────────
labels = [r"$\delta_x(t)$", r"$\delta_y(t)$", r"$\delta_z(t)$"]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for j in range(3):
    ax = axes[j]
    ax.plot(t_rom / T_base, V_rom[j], "-", color="tab:red", lw=1.4,
            label=r"ROM (periodic-$g$)")
    ax.plot(t_fom / T_base, V_fom[j], "--", color="black", lw=1.0,
            alpha=0.9, label=r"FOM")
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_ylabel(labels[j], fontsize=14)
    ax.grid(alpha=0.2)
    if j == 0:
        ax.legend(loc="best", fontsize=11)
axes[-1].set_xlabel(r"$t / T_{\rm base}$", fontsize=13)
fig.suptitle(
    fr"Rössler on-manifold ROM vs FOM  (c = {c}, "
    fr"$s_0 = {s0_fraction}\rho_{{\rm dom}} = {s0[0].real:.3f}$)",
    fontsize=12,
)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
outpath = "results/rom_vs_fom_on_periodic_g"
fig.savefig(outpath + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath + ".pdf", bbox_inches="tight")
print(f"\nSaved -> {outpath}.png/.pdf")


# ── 3D phase portrait of the full state a = c_* + δ ──────────────────
X_ROM_full = CSTAR_ROM + V_rom
X_FOM_full = CSTAR_FOM + V_fom
# Reference 1T orbit for context
t_one = np.linspace(0.0, T_base, 300)
C_ref = _c_star_interp(t_one % T_HB)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(C_ref[0], C_ref[1], C_ref[2], "-", color="0.55", lw=1.4,
        alpha=0.7, label="1T orbit (base)")
ax.plot(X_FOM_full[0], X_FOM_full[1], X_FOM_full[2], "-",
        color="black", lw=1.0, alpha=0.85, label="FOM")
ax.plot(X_ROM_full[0], X_ROM_full[1], X_ROM_full[2], "--",
        color="tab:red", lw=1.0, alpha=0.85, label=r"ROM (periodic-$g$)")
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_zlabel(r"$z$", fontsize=12)
ax.legend(fontsize=10, loc="best")
ax.set_title(fr"Rössler on-manifold phase portrait  (c = {c})",
             fontsize=12)
fig.tight_layout()
outpath3d = "results/rom_vs_fom_on_periodic_g_3D"
fig.savefig(outpath3d + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath3d + ".pdf", bbox_inches="tight")
print(f"Saved -> {outpath3d}.png/.pdf")


# ── s(t): ROM autonomous vs FOM's along-SSM projection ────────────────
# s_ROM(t) = ROM's integrated latent.
# s_FOM(t) = w(t)^T v_FOM(t) — encode of the FOM perturbation.
# On the SSM they should agree.  Divergence at large t signals
# polynomial-invariance error or the FOM leaving the SSM.
print("\nProjecting FOM onto SSM latent  s_FOM(t) = w(t)ᵀ v_FOM(t) ...")
s_fom = np.zeros(len(t_fom))
for i, t in enumerate(t_fom):
    s_fom[i] = rom.encode(t, V_fom[:, i])[0].real

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)

ax = axes[0]
ax.plot(t_rom / T_base, S_rom.real, "-", color="tab:red", lw=1.4,
        label=r"$s_{\rm ROM}(t)$")
ax.plot(t_fom / T_base, s_fom, "--", color="black", lw=1.0, alpha=0.9,
        label=r"$s_{\rm FOM}(t) = w(t)^\top v_{\rm FOM}(t)$")
ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
ax.set_ylabel(r"$s(t)$", fontsize=15, labelpad=6)
ax.grid(alpha=0.25)
ax.legend(loc="best", fontsize=12)
ax.set_title(fr"Rössler on-manifold latent  ($c = {c}$,  "
             fr"$s_0 = {s0[0].real:.3f}$)",
             fontsize=12)

# Log-scale gap
n_common = min(len(t_rom), len(t_fom))
diff = np.abs(S_rom.real[:n_common] - s_fom[:n_common])
ax = axes[1]
ax.semilogy(t_rom[:n_common] / T_base, np.maximum(diff, 1e-30),
            "-", color="tab:purple", lw=1.4,
            label=r"$|s_{\rm ROM}(t) - s_{\rm FOM}(t)|$")
ax.set_xlabel(r"$t / T_{\rm base}$", fontsize=15, labelpad=6)
ax.set_ylabel(r"$|s_{\rm ROM} - s_{\rm FOM}|$", fontsize=14, labelpad=6)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
ax.legend(loc="best", fontsize=12)

fig.tight_layout()
outpath_s = "results/rom_vs_fom_on_periodic_g_s"
fig.savefig(outpath_s + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath_s + ".pdf", bbox_inches="tight")
print(f"Saved -> {outpath_s}.png/.pdf")

plt.show()


PETSc.COMM_WORLD.Barrier()
os._exit(0)
