"""
Movie of an off-manifold FOM trajectory approaching the SSM and then
continuing on it — same style as fig3_manifold_movie.py, but the faded
2T-orbit trace is replaced by a live FOM trajectory.

At each frame time t_k:
    - yellow curve : SSM slice { P(t_k, s) : s in [0, +s*] }
    - red dot      : 2T-orbit projection  P(t_k, +s*)
    - black dot    : base flow (origin in perturbation coords)
    - blue trace   : off-manifold FOM trajectory in (δx, δy, δz) up to t_k
    - blue dot     : FOM's current position at t_k

Frames go into <outdir>/fig3_offmanifold_movie/frame_XXX.png; join with

    ffmpeg -r 20 -i frame_%03d.png -pix_fmt yuv420p ssm_offmanifold_movie.mp4
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
ctx = build_context()
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


# ── Off-manifold IC (neutral-projected transverse kick) ──────────────
seed = 0
rng = np.random.default_rng(seed)

s0_fraction = 0.1
s0 = s0_fraction * s_star * np.array([1.0 + 0.0j], dtype=complex)
delta_on = rom.decode(0.0, s0).real          # on-manifold part at t = 0

# Neutral projection at t = 0 (strip the orbit-tangent direction so the
# kick doesn't accumulate a slow linear drift along c_*(t)).
V_neut_hb = cache["V_neut_hb"][:, :, 0]      # (n_harm, n) complex
W_neut_hb = cache["W_neut_hb"][:, :, 0]
nf_cache = (V_neut_hb.shape[0] - 1) // 2
k_idx_neut = np.arange(-nf_cache, nf_cache + 1)
omega = 2.0 * np.pi / T_HB
w = np.exp(1j * k_idx_neut * omega * 0.0)
v_neut_0 = (w @ V_neut_hb).real
w_neut_0 = (w @ W_neut_hb).real
biorth = float(np.dot(w_neut_0, v_neut_0))

kick_dir = rng.standard_normal(n)
kick_dir = kick_dir - (float(np.dot(w_neut_0, kick_dir)) / biorth) * v_neut_0
kick_dir /= np.linalg.norm(kick_dir)

eps_over_delta = 5.0
eps = eps_over_delta * float(np.linalg.norm(delta_on))
v0_fom = delta_on + eps * kick_dir
print(f"  ||decode(0, s_0)|| = {np.linalg.norm(delta_on):.4f},  "
      f"ε = {eps:.4f}")


# ── Integrate FOM over movie duration ────────────────────────────────
n_periods_movie = 4                          # in T_HB
t_end = n_periods_movie * T_HB


def fom_perturbation_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


print(f"\nIntegrating FOM  0 → {n_periods_movie} T_HB = "
      f"{t_end:.3f} = {n_periods_movie * 2:.1f} T_base ...")
sol_fom = sp.integrate.solve_ivp(
    fom_perturbation_rhs, [0.0, t_end], v0_fom,
    method="RK45", rtol=1e-10, atol=1e-12, dense_output=True,
)
print(f"  nfev = {sol_fom.nfev}")


# ── Frame grid ────────────────────────────────────────────────────────
n_frames = 400
t_frames = np.linspace(0.0, t_end, n_frames, endpoint=False)


# ── Precompute SSM slice and 2T-orbit position at each frame ─────────
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


# ── FOM trajectory in perturbation coordinates on the frame grid ─────
V_fom_frames = sol_fom.sol(t_frames)        # (n, n_frames)


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
movie_dir = os.path.join(outdir, "fig3_offmanifold_movie")
os.makedirs(movie_dir, exist_ok=True)

C_FOM_TRACE = "#2166ac"                      # steel blue

for k, t_k in enumerate(t_frames):
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # SSM slice at t_k (yellow, thick)
    ax.plot(
        V_manifold[:, 0, k], V_manifold[:, 1, k], V_manifold[:, 2, k],
        color=C_SSM, lw=3.0, zorder=3,
        label=r"Spectral submanifold $\mathcal{W}(\mathcal{V})$",
    )

    # Base flow (origin)
    ax.scatter(
        [0.0], [0.0], [0.0], color=C_1T, s=110,
        edgecolor="black", linewidth=0.6, zorder=6,
        label=r"$T$-periodic base flow",
    )

    # Instantaneous 2T-orbit position
    ax.scatter(
        [V_pos[0, k]], [V_pos[1, k]], [V_pos[2, k]],
        color=C_2T, s=110, edgecolor="black", linewidth=0.6,
        marker="o", zorder=5, label=r"$2T$-periodic orbit",
    )

    # Off-manifold FOM trajectory  0 → t_k
    if k > 0:
        ax.plot(
            V_fom_frames[0, :k + 1],
            V_fom_frames[1, :k + 1],
            V_fom_frames[2, :k + 1],
            color=C_FOM_TRACE, lw=1.4, alpha=0.9, zorder=4,
            label=r"FOM (off-manifold IC)",
        )
    ax.scatter(
        [V_fom_frames[0, k]], [V_fom_frames[1, k]], [V_fom_frames[2, k]],
        color=C_FOM_TRACE, s=90, edgecolor="black", linewidth=0.6,
        marker="o", zorder=7,
        label=(None if k > 0 else r"FOM (off-manifold IC)"),
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
      "ssm_offmanifold_movie.mp4")


PETSc.COMM_WORLD.Barrier()
os._exit(0)
