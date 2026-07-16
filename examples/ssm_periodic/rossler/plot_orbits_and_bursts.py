r"""
3D physical-coordinate snapshot:

    - T-periodic base flow   c_*(t),   t ∈ [0, T_base)      (black)
    - 2T-periodic orbit      c_*(t) + decode(t, +s*),
                              t ∈ [0, T_HB)                 (red)
    - Off-manifold FOM burst  x_FOM(t) = c_*(t) + v_FOM(t),
                              t ∈ [0, T_HB]                 (viridis)
    - On-manifold ROM burst   x_ROM(t) = c_*(t) + decode(t, S_ROM(t)),
                              t ∈ [0, T_HB]                 (plasma)

IC + integrator settings match ``plot_rom_vs_fom_off_periodic_g.py``:
    q_0 = c_*(0) + ε · η,   ε = 0.025,   η ~ unit-norm random  (seed = 0)
    v_FOM(0) = q_0 − c_*(0)
    s_ROM(0) = encode(0, v_FOM(0))

Short burst only (2 T_base) so the transverse collapse and the along-
manifold coast are both visible in one 3D view.
"""
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir,
    C_1T, C_2T,
)
from rossler_rhs import perturbation_linear_action, quadratic_bilinear


setup_matplotlib()
ctx = build_context("data/ssm_cache_2T_periodic_g.npz")
rom = ctx["rom"]
c_star_at = ctx["c_star_at"]
s_star = ctx["s_star"]
T_HB = ctx["T_HB"]
T_base = ctx["T_base"]
rho_domain = ctx["rho_domain"]
n = ctx["n"]
c = ctx["c"]

print(f"|s*| = {s_star:.4f},   T_base = {T_base:.4f},   T_HB = {T_HB:.4f}")


# ── IC (identical to plot_rom_vs_fom_off_periodic_g.py) ──────────────
seed = 0
tau = 0.0
eps = 0.025

rng = np.random.default_rng(seed)
kick_dir = rng.standard_normal(n)
kick_dir /= np.linalg.norm(kick_dir)
q_0 = c_star_at(tau) + eps * kick_dir
v0_fom = q_0 - c_star_at(tau)                # = eps · unit random direction
s0_rom = rom.encode(tau, v0_fom).real

print(f"FOM IC:  ‖v_0‖ = {np.linalg.norm(v0_fom):.4f}")
print(f"ROM IC:  s_0   = {s0_rom[0]:+.4f}   (|s_0|/|s*| = "
      f"{abs(s0_rom[0]) / s_star:.3f})")


# ── Integrate FOM and ROM over [0, 2 T_base] = [0, T_HB] ─────────────
t_end = tau + T_HB
rtol, atol = 1e-11, 1e-13
n_t = 4000
t_eval = np.linspace(tau, t_end, n_t)


def fom_perturbation_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


def rom_rhs_real(t, s):
    return rom.latent_space_dynamics(t, s).real


print(f"\nIntegrating FOM (RK45) over [{tau:.3f}, {t_end:.3f}] ...")
sol_fom = sp.integrate.solve_ivp(
    fom_perturbation_rhs, [tau, t_end], v0_fom,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
)
print(f"  FOM nfev = {sol_fom.nfev}")

print(f"Integrating ROM (RK45) over [{tau:.3f}, {t_end:.3f}] ...")
sol_rom = sp.integrate.solve_ivp(
    rom_rhs_real, [tau, t_end], s0_rom,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    max_step=T_base / 20.0,
)
S_rom = sol_rom.y[0]
t_rom = sol_rom.t

# Physical state = c_*(t) + perturbation
CSTAR = np.stack([c_star_at(t) for t in t_eval], axis=1)      # (n, n_t)
X_fom = CSTAR + sol_fom.y                                     # (n, n_t)

n_rom = len(t_rom)
X_rom = np.zeros((n, n_rom))
for i in range(n_rom):
    X_rom[:, i] = c_star_at(t_rom[i]) + rom.decode(
        t_rom[i], S_rom[i:i + 1],
    ).real


# ── T-periodic base and 2T-periodic orbit (physical) ─────────────────
n_orb = 400
t_T = np.linspace(0.0, T_base, n_orb, endpoint=False)
t_2T = np.linspace(0.0, T_HB, 2 * n_orb, endpoint=False)

C_T = np.stack([c_star_at(t) for t in t_T], axis=1)           # (n, n_orb)
C_2T_orbit = np.stack(
    [
        c_star_at(t)
        + rom.decode(t, np.array([+s_star + 0.0j], dtype=complex)).real
        for t in t_2T
    ],
    axis=1,
)


# ── Time-colored 3D lines via Line3DCollection ───────────────────────
def _colored_line(ax, X, cmap, label=None, lw=2.0, alpha=1.0, zorder=5):
    """Draw a 3D line whose color varies with the sample index."""
    pts = X.T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t_norm = np.linspace(0.0, 1.0, X.shape[1] - 1)
    lc = Line3DCollection(segs, cmap=cmap, alpha=alpha, zorder=zorder)
    lc.set_array(t_norm)
    lc.set_linewidth(lw)
    ax.add_collection3d(lc)
    if label is not None:
        # a single opaque line for the legend proxy
        ax.plot([], [], [], color=cmap(0.6), lw=lw, label=label)
    return lc


fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection="3d")

# T-periodic base (thin, black, closed loop)
ax.plot(
    np.append(C_T[0], C_T[0, 0]),
    np.append(C_T[1], C_T[1, 0]),
    np.append(C_T[2], C_T[2, 0]),
    color=C_1T, lw=1.2, alpha=0.85, zorder=2,
    label=r"$T$-periodic base flow",
)

# 2T-periodic orbit (red, closed loop)
ax.plot(
    np.append(C_2T_orbit[0], C_2T_orbit[0, 0]),
    np.append(C_2T_orbit[1], C_2T_orbit[1, 0]),
    np.append(C_2T_orbit[2], C_2T_orbit[2, 0]),
    color=C_2T, lw=1.8, alpha=0.9, zorder=3,
    label=r"$2T$-periodic orbit",
)

# Off-manifold FOM burst (viridis)
_colored_line(
    ax, X_fom, cmap=plt.cm.viridis, lw=2.4, alpha=1.0, zorder=6,
    label="FOM (off manifold)",
)

# On-manifold ROM burst (plasma) — trimmed to same time grid
_colored_line(
    ax, X_rom, cmap=plt.cm.plasma, lw=2.4, alpha=1.0, zorder=6,
    label="ROM (on manifold)",
)

# Endpoints as markers
ax.scatter(
    [X_fom[0, 0]], [X_fom[1, 0]], [X_fom[2, 0]],
    color=plt.cm.viridis(0.0), s=60, edgecolor="black", linewidth=0.6,
    zorder=8,
)
ax.scatter(
    [X_rom[0, 0]], [X_rom[1, 0]], [X_rom[2, 0]],
    color=plt.cm.plasma(0.0), s=60, edgecolor="black", linewidth=0.6,
    zorder=8,
)

ax.set_xlabel(r"$x$", fontsize=17, labelpad=8)
ax.set_ylabel(r"$y$", fontsize=17, labelpad=8)
ax.set_zlabel(r"$z$", fontsize=17, labelpad=8)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.grid(True)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor("white")
    pane.set_edgecolor("white")
    pane.fill = False

ax.legend(loc="upper right", fontsize=11)

ax.text2D(
    0.03, 0.97,
    rf"burst $t \in [0,\,2T]$   $\varepsilon = {eps}$",
    transform=ax.transAxes, ha="left", va="top", fontsize=13,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white", edgecolor="0.6", linewidth=0.6,
    ),
)

fig.tight_layout()

outdir = ensure_outdir()
outpath = os.path.join(outdir, "plot_orbits_and_bursts")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
print(f"\nSaved -> {outpath}.png/.pdf")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
