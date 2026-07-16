r"""
Movie of an off-manifold FOM trajectory in perturbation coordinates
δ = x − c_*(t).  Same IC + integration setup as
``plot_rom_vs_fom_off_periodic_g.py``:

    q_0 = c_*(τ) + ε · η,      η ~ unit-norm random    (seed = 0)
    v_0 = q_0 − c_*(τ) = ε · η,                        ε = 0.025

No phase search — τ = 0 and both FOM and ROM evolve in the same absolute
time.  The FOM is integrated once with dense output, then re-sampled on
the frame grid.  The ROM (encoded via s_0 = w(τ)ᵀ v_0) is not drawn on
these frames; the movie's job is to show the FOM dot collapsing onto
the yellow SSM slice.

At each frame time t_k:
    - yellow curve : SSM slice { P(t_k, s) : s ∈ [0, +s*] }
    - red dot      : 2T-orbit projection  P(t_k, +s*)
    - black origin : T-periodic base flow  (δ = 0 by construction)
    - blue dot     : FOM(t_k) − c_*(t_k) — one dot per frame, no trace

Frames go into <outdir>/fig5_offmanifold_movie/frame_XXX.png; join with

    ffmpeg -r 30 -i frame_%03d.png -pix_fmt yuv420p offmanifold_movie.mp4
"""
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir,
    C_1T, C_2T, C_SSM,
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
cache = ctx["cache"]

print(f"|s*| = {s_star:.4f},   T_base = {T_base:.4f},   T_HB = {T_HB:.4f}")


# ── Off-manifold IC (identical to plot_rom_vs_fom_off_periodic_g.py) ─
seed = 0
tau = 0.0
eps = 0.025

rng = np.random.default_rng(seed)
kick_dir = rng.standard_normal(n)
kick_dir /= np.linalg.norm(kick_dir)
q_0 = c_star_at(tau) + eps * kick_dir
v0_fom = q_0 - c_star_at(tau)                # = eps · unit random direction

print(f"FOM IC:  q_0 = c_*(0) + ε · η,   ε = {eps},  "
      f"η ~ unit-norm random")
print(f"  q_0 = ({q_0[0]:+.4f}, {q_0[1]:+.4f}, {q_0[2]:+.4f})   "
      f"‖q_0‖ = {np.linalg.norm(q_0):.4f}")
print(f"  v(τ)  = q_0 − c_*(0)      ‖v‖ = "
      f"{np.linalg.norm(v0_fom):.4f}")


# ── Integrate FOM (RK45, tight tolerances, dense output) ─────────────
n_periods_movie = 4                          # in T_HB
t_end = tau + n_periods_movie * T_HB
rtol, atol = 1e-11, 1e-13


def fom_perturbation_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


print(f"\nIntegrating FOM  {tau:.3f} → {n_periods_movie} T_HB "
      f"({n_periods_movie * 2:.1f} T_base) ...")
sol_fom = sp.integrate.solve_ivp(
    fom_perturbation_rhs, [tau, t_end], v0_fom,
    method="RK45", rtol=rtol, atol=atol, dense_output=True,
)
print(f"  nfev = {sol_fom.nfev}")


# ── Frame grid ────────────────────────────────────────────────────────
n_frames = 400
t_frames = np.linspace(tau, t_end, n_frames, endpoint=False)


# ── Precompute manifold slice + 2T-orbit position at each frame ──────
n_s = 200
s_grid = np.linspace(0.0, s_star, n_s)
V_manifold = np.zeros((n_s, 3, n_frames))
V_pos = np.zeros((3, n_frames))
for k, t_k in enumerate(t_frames):
    for i, s_i in enumerate(s_grid):
        v = rom.decode(t_k, np.array([s_i + 0.0j], dtype=complex)).real
        V_manifold[i, :, k] = v[:3]
    V_pos[:, k] = rom.decode(
        t_k, np.array([+s_star + 0.0j], dtype=complex),
    ).real[:3]


# ── FOM perturbation at each frame  (δ = v_FOM(t)) ───────────────────
V_fom_frames = sol_fom.sol(t_frames)         # (n, n_frames)


# ── Fixed axis limits across all frames ──────────────────────────────
def _lim(arr):
    return 1.10 * float(np.max(np.abs(arr)))


lim_x = _lim(np.concatenate([
    V_manifold[..., 0].ravel(), V_pos[0], V_fom_frames[0]
]))
lim_y = _lim(np.concatenate([
    V_manifold[..., 1].ravel(), V_pos[1], V_fom_frames[1]
]))
lim_z = _lim(np.concatenate([
    V_manifold[..., 2].ravel(), V_pos[2], V_fom_frames[2]
]))


# ── Render every frame ───────────────────────────────────────────────
outdir = ensure_outdir()
movie_dir = os.path.join(outdir, "fig5_offmanifold_movie")
os.makedirs(movie_dir, exist_ok=True)

for k, t_k in enumerate(t_frames):
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # SSM slice at t_k (yellow, thick)
    ax.plot(
        V_manifold[:, 0, k], V_manifold[:, 1, k], V_manifold[:, 2, k],
        color=C_SSM, lw=3.0, zorder=3,
        label=r"Spectral submanifold $\mathcal{W}(\mathcal{V})$",
    )

    # Base flow (origin) — δ = 0
    ax.scatter(
        [0.0], [0.0], [0.0], color=C_1T, s=110,
        edgecolor="black", linewidth=0.6, zorder=6,
        label=r"$T$-periodic base flow",
    )

    # Instantaneous 2T-orbit position on the SSM slice
    ax.scatter(
        [V_pos[0, k]], [V_pos[1, k]], [V_pos[2, k]],
        color=C_2T, s=110, edgecolor="black", linewidth=0.6,
        marker="o", zorder=5, label=r"$2T$-periodic orbit",
    )

    # FOM's current perturbation δ(t_k) — one dot, no trailing trace
    ax.scatter(
        [V_fom_frames[0, k]],
        [V_fom_frames[1, k]],
        [V_fom_frames[2, k]],
        color="#1f77b4", s=95, edgecolor="black", linewidth=0.8,
        marker="o", zorder=7, label=r"FOM  (off-manifold IC)",
    )

    ax.set_xlabel(r"$\delta x$", fontsize=17, labelpad=8)
    ax.set_ylabel(r"$\delta y$", fontsize=17, labelpad=8)
    ax.set_zlabel(r"$\delta z$", fontsize=17, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_zlim(-lim_z, lim_z)

    # Clean 3D style: white background panes, grid on.
    ax.grid(True)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("white")
        pane.fill = False

    ax.text2D(
        0.03, 0.97, rf"$t / T = {t_k / T_base:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=15,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="0.6", linewidth=0.6,
        ),
    )

    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()

    fpath = os.path.join(movie_dir, f"frame_{k:03d}.png")
    fig.savefig(fpath, bbox_inches="tight", pad_inches=0.5, dpi=200)
    plt.close(fig)
    if k % 20 == 0 or k == n_frames - 1:
        print(f"  frame {k+1:3d} / {n_frames}   ({fpath})")

print(f"\nMovie frames saved to {movie_dir}/")
print("Assemble with:  ffmpeg -r 30 -i frame_%03d.png -pix_fmt yuv420p "
      "offmanifold_movie.mp4")


PETSc.COMM_WORLD.Barrier()
os._exit(0)
