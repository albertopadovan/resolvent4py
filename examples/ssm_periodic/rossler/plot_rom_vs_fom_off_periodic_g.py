r"""
Off-manifold ROM vs FOM comparison for the Rössler 2T-periodic SSM.

Works with either SSM cache — the ROM class is dispatched based on the
keys present in the ``.npz``:

    data/ssm_cache_2T.npz            → SpectralSubmanifoldROM       (constant-g)
    data/ssm_cache_2T_periodic_g.npz → SpectralSubmanifoldROMPeriodicG

Change ``cache_path`` below to switch which one is loaded.

FOM IC:  v_0 = decode(τ, s_0) + ε · η,       η random unit direction
ROM IC:  s(τ) = w(τ)ᵀ · v_0            (encode: project onto the SSM)

Both the FOM and ROM see the same physical IC at t = τ; the ROM's
projection kills the transverse component of the kick, so any long-term
mismatch between δ_FOM(t) and δ_ROM(t) measures the transverse decay
rate of the SSM as seen by the FOM.
"""
import os
from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from petsc4py import PETSc

from resolvent4py.spectral_submanifold import (
    SpectralSubmanifoldROM,
    SpectralSubmanifoldROMPeriodicG,
)
from _fig_common import setup_matplotlib, ensure_outdir, C_FOM, C_ROM
from rossler_rhs import perturbation_linear_action, quadratic_bilinear


setup_matplotlib()


# ── Load SSM cache (auto-dispatch on variant) ────────────────────────
cache_path = "data/ssm_cache_2T_periodic_g.npz"
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"{cache_path} not found. Run save_ssm_2T.py or "
        f"save_ssm_2T_periodic_g.py first."
    )
cache = np.load(cache_path)
T_HB = float(cache["T"])                    # = 2·T_base
T_base = float(cache["T_base"])
c = float(cache["c"])
n = int(cache["n"])
omega = 2.0 * np.pi / T_HB
rho_domain = float(cache["rho_domain"])
Lam0 = complex(cache["Lams"][0])

if "gs_periodic" in cache.files:
    ssm_variant = "periodic-g"
    rom = SpectralSubmanifoldROMPeriodicG(
        multiindices=cache["multiindices"],
        Lams=cache["Lams"],
        gs=cache["gs_periodic"],
        PS=cache["PS_hb"],
        W=cache["W_hb"],
        omega=omega,
    )
    rom.use_g_spline = False
elif "gs" in cache.files:
    ssm_variant = "constant-g"
    rom = SpectralSubmanifoldROM(
        multiindices=cache["multiindices"],
        Lams=cache["Lams"],
        gs=cache["gs"],
        PS=cache["PS_hb"],
        W=cache["W_hb"],
        conj_to_linear_dynamics=bool(cache["conj_to_linear_dynamics"]),
        omega=omega,
        v_neutral=(cache["V_neut_hb"][:, :, 0]
                   if "V_neut_hb" in cache.files else None),
        w_neutral=(cache["W_neut_hb"][:, :, 0]
                   if "W_neut_hb" in cache.files else None),
    )
else:
    raise KeyError(
        f"{cache_path} has neither 'gs' (constant-g) nor 'gs_periodic' "
        f"(periodic-g) — unrecognised SSM cache format."
    )

print(f"{ssm_variant} cache (Rössler, {cache_path}):")
print(f"  c = {c},  |Λ| = {abs(Lam0.real):.4e},")
print(f"  T_base = {T_base:.4f},  T_HB = 2·T_base = {T_HB:.4f},  "
      f"rho_domain = {rho_domain:.3f}")


# ── c_*(t) interpolant ────────────────────────────────────────────────
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(
    t_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


def fom_perturbation_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


# ── Parameters ─────────────────────────────────────────────────────────
seed = 0
s0_fraction = 0.1
eps_over_delta = 0.25                      # kick / ||decode(τ, s_0)||
s0 = s0_fraction * rho_domain * np.array([1.0 + 0.0j], dtype=complex)
tau = 0.0
n_periods = 100
t_end = tau + n_periods * T_base
rtol, atol = 1e-11, 1e-13


# ── Off-manifold IC ────────────────────────────────────────────────────
# A small random state in ℝ³ — deliberately unassociated with c_*(t)
# and the SSM's decode(0, ·) direction, so no privileged relationship
# between t = 0 and q_0.  Rossler's transverse mode decays with
# μ ≈ 10⁻¹⁵ per T_base, so the FOM lands on the 2T attractor within
# ≈ 1 T_base and lives there for the rest of the run.  The phase
# search then has a genuine "which t_0 fits the FOM tail" question to
# answer.
rng = np.random.default_rng(seed)
tau_nominal = tau
eps = 0.025
kick_dir = rng.standard_normal(n)
kick_dir /= np.linalg.norm(kick_dir)
q_0 = c_star_at(tau_nominal) + eps * kick_dir
x0_state = q_0.copy()
v0_fom_nominal = q_0 - c_star_at(tau_nominal)  # = eps · unit random direction

print(f"FOM IC:  q_0 = c_*(0) + ε · η,   ε = {eps},  "
      f"η ~ unit-norm random")
print(f"  q_0 = ({q_0[0]:+.4f}, {q_0[1]:+.4f}, {q_0[2]:+.4f})   "
      f"‖q_0‖ = {np.linalg.norm(q_0):.4f}")
print(f"  v(τ_nom) = q_0 − c_*(0)      ‖v‖ = "
      f"{np.linalg.norm(v0_fom_nominal):.4f}")


# ── FOM integration ONCE from τ_nominal with dense output ────────────
# The FOM's state-space trajectory is fixed by x_0 = c_*(τ_nom) + v_0;
# we integrate it here (dense output) and then re-sample at whatever
# absolute times the phase search and final plots need.
def rom_rhs_real(t, s):
    return rom.latent_space_dynamics(t, s).real


print("\nIntegrating FOM once (from τ_nominal, dense output) ...")
t_end_fom = tau_nominal + (n_periods + 2) * T_base           # 2·T_base safety
sol_fom = sp.integrate.solve_ivp(
    fom_perturbation_rhs, [tau_nominal, t_end_fom], v0_fom_nominal,
    method="RK45", rtol=rtol, atol=atol, dense_output=True,
)
print(f"  FOM: RK45,  nfev = {sol_fom.nfev}")


def x_fom_at(times):
    """Full state x(t) = c_*(t) + v_FOM(t) at arbitrary times."""
    v = sol_fom.sol(times)
    if np.ndim(times) == 0:
        return c_star_at(float(times)) + v
    C = np.stack([c_star_at(float(t)) for t in times], axis=1)
    return C + v


# ── Encode at nominal τ (no phase search) ────────────────────────────
tau = tau_nominal
v0_fom = v0_fom_nominal
s0_rom_off = rom.encode(tau, v0_fom).real
t_end = tau + n_periods * T_base

print(f"\nEncoded s(τ) = w(τ)ᵀ v_0 = {s0_rom_off[0]:+.4f}")


# ── Integrate ROM from adopted (τ = t_0*, s_0*) at tight tolerances ─
n_t = 30000
t_eval = np.linspace(tau, t_end, n_t)

print(f"\nIntegrating ROM (RK45, starts from t_0* = {tau:.4f}) ...")
sol_rom = sp.integrate.solve_ivp(
    rom_rhs_real, [tau, t_end], s0_rom_off,
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


# ── Sample FOM at the ROM's t_eval from the dense-output solution ────
V_fom = sol_fom.sol(t_eval)
t_fom = t_eval
CSTAR_FOM = np.zeros((n, len(t_fom)))
for i in range(len(t_fom)):
    CSTAR_FOM[:, i] = c_star_at(t_fom[i])


# ── Along-SSM latent  s_FOM(t) = w(t)ᵀ v_FOM(t) ──────────────────────
# Encode the FOM perturbation at each output time.  On the SSM this
# should track S_rom(t); divergence signals off-manifold content.
print("Projecting FOM onto SSM latent  s_FOM(t) = w(t)ᵀ v_FOM(t) ...")
s_fom = np.zeros(len(t_fom))
for i, t in enumerate(t_fom):
    s_fom[i] = rom.encode(t, V_fom[:, i])[0].real


# ── Plot perturbations δ_j(t) — three windows ────────────────────────
# Each figure has three subpanels (δx, δy, δz).  Windows:
#   (1) full run
#   (2) initial transient        t / T_base ∈ [0, 4]
#   (3) final steady-state 2T    t / T_base ∈ [n_periods − 2, n_periods]
mode_labels = ["x", "y", "z"]
outdir = ensure_outdir()


def _plot_window(t_lo, t_hi, tag):
    """δx/δy/δz ROM vs FOM restricted to t/T_base ∈ [t_lo, t_hi]."""
    m_rom = (t_rom / T_base >= t_lo) & (t_rom / T_base <= t_hi)
    m_fom = (t_fom / T_base >= t_lo) & (t_fom / T_base <= t_hi)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 7.0))
    for j in range(3):
        ax = axes[j]
        ax.plot(t_fom[m_fom] / T_base, V_fom[j][m_fom],
                color=C_FOM, lw=1.6, label="FOM (truth)")
        ax.plot(t_rom[m_rom] / T_base, V_rom[j][m_rom],
                color=C_ROM, lw=1.6, ls="--", label="ROM (SSM)")
        ax.set_ylabel(
            rf"$\delta {mode_labels[j]}(t)$",
            fontsize=20, labelpad=6,
        )
        ax.tick_params(axis="both", which="major", labelsize=15)
    axes[-1].set_xlabel(r"$t / T$", fontsize=22, labelpad=6)
    axes[-1].set_xlim(t_lo, t_hi)
    fig.tight_layout()

    outpath = os.path.join(outdir, f"fig_ts_off_periodic_g_{tag}")
    fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
    print(f"Saved -> {outpath}.png/.pdf")
    return fig


_plot_window(0.0, float(n_periods), tag="full")
_plot_window(0.0, 4.0, tag="transient")
_plot_window(n_periods - 2.0, float(n_periods), tag="steady_2T")


# ── s(t): ROM latent vs FOM's SSM projection ─────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.plot(t_fom / T_base, s_fom, color=C_FOM, lw=1.6,
        label=r"$s_{\rm FOM}(t) = \psi(t)^\top (q(t) - \overline{q}(t))$")
ax.plot(t_rom / T_base, S_rom.real, color=C_ROM, lw=1.6, ls="--",
        label=r"$s_{\rm ROM}(t)$")
ax.set_xlabel(r"$t / T$", fontsize=22, labelpad=6)
ax.set_ylabel(r"$s(t)$", fontsize=20, labelpad=6)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.set_xlim(0.0, float(n_periods))
ax.legend(loc="lower right", fontsize=15)
fig.tight_layout()

outpath = os.path.join(outdir, "fig_ts_off_periodic_g_s")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
print(f"Saved -> {outpath}.png/.pdf")


# ── 3D phase portrait ─────────────────────────────────────────────────
X_ROM_full = CSTAR_ROM + V_rom
X_FOM_full = CSTAR_FOM + V_fom
t_one = np.linspace(0.0, T_base, 300)
C_ref = _c_star_interp(t_one % T_HB)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(C_ref[0], C_ref[1], C_ref[2], "-", color="0.55", lw=1.4, alpha=0.7)
ax.plot(X_FOM_full[0], X_FOM_full[1], X_FOM_full[2], "-",
        color="black", lw=1.0, alpha=0.85)
ax.plot(X_ROM_full[0], X_ROM_full[1], X_ROM_full[2], "--",
        color="tab:red", lw=1.0, alpha=0.85)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_zlabel(r"$z$", fontsize=12)
fig.tight_layout()
outpath3d = "results/rom_vs_fom_off_periodic_g_3D"
fig.savefig(outpath3d + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath3d + ".pdf", bbox_inches="tight")
print(f"Saved -> {outpath3d}.png/.pdf")

plt.show()


PETSc.COMM_WORLD.Barrier()
os._exit(0)
