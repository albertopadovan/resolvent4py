"""
Movie (200 frames over t in [0, T_HB) = [0, 2·T_base)) of the periodic
SSM slice for the Rössler system, drawn in perturbation coordinates so
that the base flow sits at (0, 0, 0) in every frame.

At each frame time t_k:
    - yellow curve : SSM slice { P(t_k, s) : s in [0, +s*] }
    - red dot      : 2T-orbit projection  P(t_k, +s*)
    - black dot    : base flow (origin in perturbation coords)
    - faded red    : full 2T-orbit trace (from all frames), for
                     spatial context

200 PNGs into <outdir>/fig3_movie/frame_XXX.png; join with

    ffmpeg -r 20 -i frame_%03d.png -pix_fmt yuv420p ssm_movie.mp4
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir,
    C_1T, C_2T, C_SSM,
)


setup_matplotlib()
ctx = build_context()
rom = ctx["rom"]
s_star = ctx["s_star"]
T_HB = ctx["T_HB"]
T_base = ctx["T_base"]
n = ctx["n"]

print(f"|s*| = {s_star:.4f},   T_base = {T_base:.4f},   T_HB = {T_HB:.4f}")


# ── Grids ────────────────────────────────────────────────────────────────
n_frames = 200
t_frames = np.linspace(0.0, T_HB, n_frames, endpoint=False)

# Draw the manifold slice from s = 0 (base flow at the origin) out to
# s = +s* (the 2T-orbit position at time t_k).  The yellow curve
# visually starts at the black dot and ends at the red dot.
n_s = 200
s_grid = np.linspace(0.0, s_star, n_s)


# ── Precompute the manifold slice and the +s* point at each frame ────────
# Rössler is 3D, so we render (δx, δy, δz) — a 3D scatter/line plot.
V_manifold = np.zeros((n_s, 3, n_frames))
V_pos = np.zeros((3, n_frames))
for k, t_k in enumerate(t_frames):
    for i, s_i in enumerate(s_grid):
        s_arr = np.array([s_i + 0.0j], dtype=complex)
        v = rom.decode(t_k, s_arr).real
        V_manifold[i, :, k] = v[:3]
    V_pos[:, k] = rom.decode(
        t_k, np.array([+s_star + 0.0j], dtype=complex),
    ).real[:3]


# ── Fixed axis limits across all frames ──────────────────────────────────
def _lim(arr):
    return 1.10 * float(np.max(np.abs(arr)))


lim_x = _lim(np.concatenate([V_manifold[..., 0].ravel(), V_pos[0]]))
lim_y = _lim(np.concatenate([V_manifold[..., 1].ravel(), V_pos[1]]))
lim_z = _lim(np.concatenate([V_manifold[..., 2].ravel(), V_pos[2]]))


# ── Render every frame ───────────────────────────────────────────────────
outdir = ensure_outdir()
movie_dir = os.path.join(outdir, "fig3_movie")
os.makedirs(movie_dir, exist_ok=True)

for k, t_k in enumerate(t_frames):
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # Full 2T-orbit trace (faded) for context
    ax.plot(
        V_pos[0], V_pos[1], V_pos[2],
        color=C_2T, lw=1.4, alpha=0.35, zorder=2,
    )

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

    # In-plot text with the current time (in units of T_base)
    ax.text2D(
        0.03, 0.97, rf"$t / T = {t_k / T_base:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=15,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="0.6", linewidth=0.6,
        ),
    )

    # Legend only on the first frame
    ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout()

    fpath = os.path.join(movie_dir, f"frame_{k:03d}.png")
    fig.savefig(fpath, bbox_inches="tight", pad_inches=0.5, dpi=200)
    plt.close(fig)
    if k % 10 == 0 or k == n_frames - 1:
        print(f"  frame {k+1:3d} / {n_frames}   ({fpath})")

print(f"\nMovie frames saved to {movie_dir}/")
print("Assemble with:  ffmpeg -r 20 -i frame_%03d.png -pix_fmt yuv420p ssm_movie.mp4")


PETSc.COMM_WORLD.Barrier()
os._exit(0)
