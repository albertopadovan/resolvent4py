"""
Figure 1: 3D phase portrait of the T-periodic base flow and the
two 2T-periodic (period-doubled) orbits for the Rössler system at
c near period-doubling.

    1T orbit  : c_star(t)              — solid black
    2T orbits : c_star(t) + P(t, ±s*)  — solid + dashed red, the two
                symmetry-related branches of the polynomial fixed point
                s* = root of g(s) = 0.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir,
    C_1T, C_2T,
)


setup_matplotlib()
ctx = build_context()
rom = ctx["rom"]
c_star_at = ctx["c_star_at"]
s_star = ctx["s_star"]
T_HB = ctx["T_HB"]
T_base = ctx["T_base"]
n = ctx["n"]
c = ctx["c"]

print(f"1T-orbit period T_base = {T_base:.4f},   HB period T_HB = {T_HB:.4f}")
print(f"2T-orbit fixed points  s* = ±{s_star:.4f}")


# ── Reconstruct the orbits ────────────────────────────────────────────────
n_1T = 600
t_1T = np.linspace(0.0, T_base, n_1T, endpoint=False)
X_1T = np.array([c_star_at(t) for t in t_1T]).T          # (n, n_1T)

n_2T = 1200
t_2T = np.linspace(0.0, T_HB, n_2T, endpoint=False)
X_2T_pos = np.zeros((n, n_2T))
X_2T_neg = np.zeros((n, n_2T))
for j, t in enumerate(t_2T):
    s_p = np.array([+s_star + 0.0j], dtype=complex)
    s_m = np.array([-s_star + 0.0j], dtype=complex)
    X_2T_pos[:, j] = c_star_at(t) + rom.decode(t, s_p).real
    X_2T_neg[:, j] = c_star_at(t) + rom.decode(t, s_m).real


# ── Plot (3D — Rössler is a 3D system) ───────────────────────────────────
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    X_1T[0], X_1T[1], X_1T[2], color=C_1T, lw=2.2,
    label=r"$T$-periodic base flow",
)
ax.plot(
    X_2T_pos[0], X_2T_pos[1], X_2T_pos[2], color=C_2T, lw=2.2,
    label=r"$2T$-periodic orbit",
)
ax.plot(
    X_2T_neg[0], X_2T_neg[1], X_2T_neg[2],
    color=C_2T, lw=2.2, ls="--", alpha=0.9,
)

ax.set_xlabel(r"$x$", fontsize=22, labelpad=8)
ax.set_ylabel(r"$y$", fontsize=22, labelpad=8)
ax.set_zlabel(r"$z$", fontsize=22, labelpad=8)
ax.tick_params(axis="both", which="major", labelsize=13)
ax.legend(loc="upper left", fontsize=15)

# Clean 3D style: white background panes, grid on.
ax.grid(True)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor("white")
    pane.set_edgecolor("white")
    pane.fill = False

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────
outdir = ensure_outdir()
outpath = os.path.join(outdir, "fig1_orbits")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.5)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.5)
print(f"Saved -> {outpath}.png/.pdf")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
