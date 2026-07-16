"""
Movie (100 frames over t in [0, 2T_phys) = [0, T_HB)) of the periodic
SSM slice, drawn in perturbation coordinates so that the base flow
sits at (0, 0) in every frame.

At each frame time t_k:
    - yellow curve : SSM slice { P(t_k, s) : s in [-1.1 s*, +1.1 s*] }
    - two red dots : 2T-orbit fixed-point projections P(t_k, ±s*)
    - black dot    : base flow (origin in perturbation coords)
    - faded red    : full 2T-orbit trace over the whole period,
                     shown for spatial context

100 PNGs into <outdir>/fig3_movie/frame_XXX.png; you can join them
with ffmpeg afterwards, e.g.:

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
T_phys = ctx["T_phys"]

print(f"|s*| = {s_star:.4f},   T_phys = {T_phys:.4f},   T_HB = {T_HB:.4f}")


# ── Grids ────────────────────────────────────────────────────────────────
n_frames = 200
t_frames = np.linspace(0.0, T_HB, n_frames, endpoint=False)

# Draw the manifold slice from s = 0 (base flow at the origin) out to
# s = +s* (the 2T-orbit position at time t_k).  This way the yellow
# curve visually starts at the black dot and ends at the red dot.
n_s = 200
s_grid = np.linspace(0.0, s_star, n_s)


# ── Precompute the manifold slice and the +s* point at each frame time ──
# Stored so we can fix axis limits across the whole movie.  Only the +s*
# branch is kept — the −s* branch traces the same 2T orbit shifted by
# T_phys and would just draw on top of it.
V_manifold = np.zeros((n_s, 2, n_frames))
V_pos = np.zeros((2, n_frames))
for k, t_k in enumerate(t_frames):
    for i, s_i in enumerate(s_grid):
        s_arr = np.array([s_i + 0.0j], dtype=complex)
        v = rom.decode(t_k, s_arr).real
        V_manifold[i, :, k] = v[:2]
    V_pos[:, k] = rom.decode(
        t_k, np.array([+s_star + 0.0j], dtype=complex),
    ).real[:2]


# ── Fixed axis limits across all frames ──────────────────────────────────
all_pts_x = np.concatenate([
    V_manifold[..., 0].ravel(), V_pos[0],
])
all_pts_y = np.concatenate([
    V_manifold[..., 1].ravel(), V_pos[1],
])
lim = 1.10 * max(float(np.max(np.abs(all_pts_x))),
                 float(np.max(np.abs(all_pts_y))))


# ── Render every frame ───────────────────────────────────────────────────
outdir = ensure_outdir()
movie_dir = os.path.join(outdir, "fig3_movie")
os.makedirs(movie_dir, exist_ok=True)

for k, t_k in enumerate(t_frames):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Full 2T-orbit trace (the red dot's path over all frames), faded.
    # Every point on this curve is on manifold(t_j) at its own t_j; it
    # does NOT sit on the current slice at t_k except where the red dot
    # is right now — but visually it lets the reader see the whole orbit
    # from the very first frame while the dot travels along it.
    ax.plot(
        V_pos[0], V_pos[1],
        color=C_2T, lw=1.4, alpha=0.35, zorder=2,
    )

    # SSM slice at t_k (yellow, thick)
    ax.plot(
        V_manifold[:, 0, k], V_manifold[:, 1, k],
        color=C_SSM, lw=3.0, zorder=3,
        label=r"Spectral submanifold $\mathcal{W}(\mathcal{V})$",
    )

    # Base flow (origin)
    ax.scatter(
        [0.0], [0.0], color=C_1T, s=110,
        edgecolor="black", linewidth=0.6, zorder=6,
        label=r"$T$-periodic base flow",
    )

    # Instantaneous 2T-orbit position
    ax.scatter(
        [V_pos[0, k]], [V_pos[1, k]],
        color=C_2T, s=110, edgecolor="black", linewidth=0.6,
        marker="o", zorder=5, label=r"$2T$-periodic orbit",
    )

    ax.set_xlabel(r"$\delta a_1$", fontsize=22, labelpad=8)
    ax.set_ylabel(r"$\delta a_2$", fontsize=22, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(False)

    # In-plot text box with the current time (in units of T_phys)
    ax.text(
        0.03, 0.97, rf"$t / T = {t_k / T_phys:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=17,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="0.6", linewidth=0.6,
        ),
    )

    # Legend only on the first frame
    if k == 0:
        ax.legend(loc="upper right", fontsize=13)

    fig.tight_layout()

    fpath = os.path.join(movie_dir, f"frame_{k:03d}.png")
    fig.savefig(fpath, bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close(fig)
    if k % 10 == 0 or k == n_frames - 1:
        print(f"  frame {k+1:3d} / {n_frames}   ({fpath})")

print(f"\nMovie frames saved to {movie_dir}/")
print("Assemble with:  ffmpeg -r 20 -i frame_%03d.png -pix_fmt yuv420p ssm_movie.mp4")


PETSc.COMM_WORLD.Barrier()
os._exit(0)
