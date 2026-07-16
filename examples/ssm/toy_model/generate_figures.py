"""
Presentation figures for the Hopf3D SSM example.  Reads
``data/ssm_cache.npz`` (produced by ``save_ssm.py``) and renders
publication-quality figures using matplotlib with LaTeX / Computer
Modern typography — palette mirrors the WCCM26 preset in
~/Documents/Presentations/WCCM26/Figures/.

Figure 1: 3D subspace-vs-manifold overview
    - Fixed point at the origin                  (black)
    - 2D spectral subspace (tangent plane at 0)  (translucent blue)
    - 2D spectral manifold  (SSM surface)        (translucent red)

Output: results/subspace_and_ssm_3d.{png,pdf}
"""

import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from petsc4py import PETSc
from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM


# ── Matplotlib style (mirror of WCCM26/generate_periodic_baseflow.py) ─────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 14,
        "text.usetex": True,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

# Palette (WCCM26)
C_PLANE = "#2166ac"      # cSIM — steel blue (linear approximation)
C_SSM = "#f1c40f"        # goldenrod yellow (nonlinear manifold)
C_FIXED = "black"        # fixed point
C_FOM = "#b2182b"        # brick red — full-order trajectory (truth)
C_SHADOW = "#b2182b"     # brick red (on-manifold ROM trajectory)


# ── Load SSM cache ─────────────────────────────────────────────────────────
cache = np.load("data/ssm_cache.npz")
PS = cache["PS"]                       # (n_terms, n_state=3) complex
W = cache["W"]                         # (n_state, r=2)       complex
multiindices = cache["multiindices"]   # (n_terms, r)         int
Lams = cache["Lams"]                   # (r,)                 complex
gs = cache["gs"]                       # (n_terms, r)         complex
conj_to_linear = bool(cache["conj_to_linear_dynamics"])
r = int(cache["r"])
n_state = int(cache["n_state"])
R = float(cache["R"])
rho_domain = float(cache["rho_domain"])


# ── Numpy-only ROM (encoder / decoder / latent-space dynamics) ────────────
rom = SpectralSubmanifoldROM(
    multiindices=multiindices, Lams=Lams, gs=gs, PS=PS, W=W,
    conj_to_linear_dynamics=conj_to_linear,
)


# Domain radius clipped to the ROM's actual limit-cycle amplitude — found
# by integrating the latent ODE to convergence.  Using the analytic
# |z|² = −Re(Λ)/Re(g_{2,1}) formula overshoots because higher-order latent
# terms (g_(3,2), g_(4,3), ...) contract the true fixed point inward; the
# numerical solve captures all of them.
_s_probe = np.array([0.05 + 0.0j, 0.05 - 0.0j], dtype=complex)
_sol_probe = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [0.0, 500.0], _s_probe,
    method="RK45", rtol=1e-11, atol=1e-13,
).y
z_LC_lat = float(abs(_sol_probe[0, -1]))
rho_domain = z_LC_lat

print(
    f"Loaded SSM cache:  R = {R:.4f}, rho_domain = {rho_domain:.4f} "
    f"(clipped to ROM's numerical |z_LC|), r = {r}, n_state = {n_state}"
)


# ── Identify the right master eigenvector v = PS_(1,0) ────────────────────
# For r=2 with a real Hopf pair, PS_(1,0) and PS_(0,1) are conjugates.  The
# linear (tangent-plane) part of the decoder at s = (z, z̄) collapses to
#     P_(1,0)·z + P_(0,1)·z̄  =  2·Re[v·z]
idx_10 = next(
    i for i in range(len(multiindices))
    if tuple(multiindices[i].tolist()) == (1, 0)
)
v_master = PS[idx_10]


# ── Sample both surfaces on the same polar grid (ρ, θ) ────────────────────
# Both hit the origin at ρ = 0 → tangency is visually obvious.
n_rho = 60
n_th = 160
rho_grid = np.linspace(0.0, rho_domain, n_rho)
th_grid = np.linspace(0.0, 2.0 * np.pi, n_th)

X_ssm = np.zeros((n_rho, n_th, 3))
X_plane = np.zeros((n_rho, n_th, 3))
for i, rho_i in enumerate(rho_grid):
    for j, th_j in enumerate(th_grid):
        z = rho_i * np.exp(1j * th_j)
        s = np.array([z, np.conj(z)], dtype=complex)
        X_ssm[i, j] = rom.decode(0.0, s).real
        X_plane[i, j] = 2.0 * np.real(v_master * z)


# ── Off-manifold FOM trajectory + on-manifold shadow ─────────────────────
# Numpy-only Hopf3D right-hand side (matches Hopf3D.evaluate_dynamics_numpy
# without the PETSc round-trip).
mu_h = float(cache["mu"])
alpha_h = float(cache["alpha"])
beta_h = float(cache["beta"])


def hopf3d_rhs(t, q):
    x, y, z = q
    return np.array([
        mu_h * x - y - alpha_h * x * z - beta_h * x * y,
        x + mu_h * y - alpha_h * y * z + beta_h * x * x,
        -alpha_h * z + alpha_h * (x * x + y * y),
    ])


# Off-manifold IC constructed from an on-manifold reference point.
# Pick a latent state  s_0  well inside the convergence domain, decode to
# get the on-manifold point  q_0 = P(s_0), then perturb by
#     eps · q_0 + eps
# (a compound of a multiplicative and an additive kick, all scaled by eps).
# The manifold shadow starts at s_0 itself — same underlying on-manifold
# point, no perturbation — so the FOM is what strays off, converging back
# onto the shadow at rate governed by the transverse Floquet decay.
eps = 0.0
z_0 = 0.5 * rho_domain * np.exp(1j * np.pi / 4)      # latent IC on the SSM
s_0 = np.array([z_0, np.conj(z_0)], dtype=complex)
q_0 = rom.decode(0.0, s_0).real                       # P(s_0), on-manifold
x0_off = q_0 + eps * q_0 + eps                        # off-manifold FOM IC

# Analytic limit-cycle radii (r_LC_phys used only for the settling print
# at the end; z_LC_lat already computed above and used to set rho_domain).
r_LC_phys = float(np.sqrt(float(cache["mu"]) / float(cache["alpha"])))
print(
    f"Limit cycle:  r_LC (phys) = {r_LC_phys:.4f}   "
    f"|z_LC| (latent) = {z_LC_lat:.4f}   "
    f"(inside R = {R:.4f}: {z_LC_lat/R*100:.1f}% of R)"
)

# Long enough window so the amplitude equation
#   d|z|²/dt ≈ 2·Re(Λ)·|z|² + 2·Re(g_{2,1})·|z|⁴
# saturates near the LC.  Time-scale ≈ 1/(2·Re(Λ)) ≈ 10 for our numbers;
# ~5 e-foldings puts us within a few percent of the fixed point.
t_end_traj = 100.0
n_traj = 2000
t_traj = np.linspace(0.0, t_end_traj, n_traj)

# (a) Full-order trajectory — solve the actual Hopf3D ODE.
sol_fom = sp.integrate.solve_ivp(
    hopf3d_rhs, [0.0, t_end_traj], x0_off,
    method="RK45", t_eval=t_traj, rtol=1e-10, atol=1e-10,
).y                                            # (3, n_traj)

# (b) Shadowing trajectory on the SSM: integrate the latent ODE directly
#     from s_0 (the underlying on-manifold latent state), then decode.
sol_shadow_s = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [0.0, t_end_traj], s_0,
    method="RK45", t_eval=t_traj, rtol=1e-12, atol=1e-12,
).y                                            # complex (r, n_traj)
X_shadow = np.zeros((3, n_traj))
for i in range(n_traj):
    X_shadow[:, i] = rom.decode(0.0, sol_shadow_s[:, i]).real

# Settling check — final radii vs analytic LC targets.
r_fom_end = float(np.hypot(sol_fom[0, -1], sol_fom[1, -1]))
z_rom_end = float(abs(sol_shadow_s[0, -1]))
print(
    f"  FOM final:    r_phys = {r_fom_end:.4f}   (target r_LC = {r_LC_phys:.4f})\n"
    f"  ROM final:    |s|    = {z_rom_end:.4f}   (target |z_LC| = {z_LC_lat:.4f})"
)


# ── Render ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.0, 7.0))
ax = fig.add_subplot(111, projection="3d")

# Spectral subspace — translucent flat disk
ax.plot_surface(
    X_plane[..., 0], X_plane[..., 1], X_plane[..., 2],
    color=C_PLANE, alpha=0.30,
    linewidth=0, antialiased=True, shade=True,
    rstride=1, cstride=2, zorder=1,
)

# Spectral submanifold — translucent curved surface
ax.plot_surface(
    X_ssm[..., 0], X_ssm[..., 1], X_ssm[..., 2],
    color=C_SSM, alpha=0.55,
    linewidth=0, antialiased=True, shade=True,
    rstride=1, cstride=2, zorder=2,
)

# Boundary rings (ρ = rho_domain) — sharpen the read of surface curvature
ax.plot(
    X_ssm[-1, :, 0], X_ssm[-1, :, 1], X_ssm[-1, :, 2],
    color=C_SSM, lw=1.6, zorder=3,
)
ax.plot(
    X_plane[-1, :, 0], X_plane[-1, :, 1], X_plane[-1, :, 2],
    color=C_PLANE, lw=1.0, alpha=0.85, zorder=3,
)

# Fixed point
ax.scatter(
    [0], [0], [0], color=C_FIXED, s=55, depthshade=False, zorder=4,
)

# Off-manifold IC marker (▲) common to both trajectories
ax.scatter(
    [x0_off[0]], [x0_off[1]], [x0_off[2]],
    color=C_FOM, s=55, marker="^", depthshade=False, zorder=6,
)

# Full-order trajectory — falls onto the SSM
ax.plot(
    sol_fom[0], sol_fom[1], sol_fom[2],
    color=C_FOM, lw=1.8, zorder=5,
)


# LaTeX-set axis labels
ax.set_xlabel(r"$x$", fontsize=22, labelpad=12)
ax.set_ylabel(r"$y$", fontsize=22, labelpad=12)
ax.set_zlabel(r"$z$", fontsize=22, labelpad=12)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="z", which="major", labelsize=16)

# Clean pane backgrounds (no grey walls)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.fill = False
    a.pane.set_edgecolor("lightgray")

# No grid — keeps the 3D volume visually clean
ax.grid(False)

# Legend via proxy handles (Axes3D doesn't attach one to plot_surface)
legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=C_FIXED,
        markersize=8, label=r"Fixed point $\overline{q} = (0,0,0)$",
    ),
    Patch(
        facecolor=C_PLANE, alpha=0.55,
        label=r"Spectral subspace $\mathcal{V}$",
    ),
    Patch(
        facecolor=C_SSM, alpha=0.75,
        label=r"Spectral submanifold $\mathcal{W}(\mathcal{V})$",
    ),
]
ax.legend(
    handles=legend_handles, loc="upper left", fontsize=17,
    bbox_to_anchor=(0.0, 1.08), borderaxespad=0.0,
    handlelength=1.6, handletextpad=0.7,
)

# Camera — an off-isometric view that shows both surfaces cleanly
ax.view_init(elev=22, azim=42)
ax.set_box_aspect([1.0, 1.0, 0.9])

# Framing — include both surfaces AND the trajectories (their z reach is
# larger than the SSM sheet since the FOM IC is well off-manifold).
extent_xy = max(
    float(np.max(np.abs(X_ssm[..., :2]))),
    float(np.max(np.abs(sol_fom[:2]))),
)
extent_z = max(
    float(np.max(np.abs(X_ssm[..., 2]))),
    float(np.max(np.abs(sol_fom[2]))),
)
lim_xy = 1.05 * extent_xy
lim_z = 1.05 * extent_z
eps_z = 0.02 * lim_z                                 # tiny negative padding
ax.set_xlim(-lim_xy, lim_xy)
ax.set_ylim(-lim_xy, lim_xy)
ax.set_zlim(0.0 - eps_z, lim_z)

fig.tight_layout()


# ── Interactive display, then save when the window is closed ────────────
# The block=True call captures the elev/azim you dial in with the mouse.
# Close the window (⌘W / red-dot / Esc, depending on backend) to trigger
# the save with your chosen view.
print("Showing figure — close the window to save with the current view.")
fig.tight_layout()
plt.show(block=True)

outdir = "/Users/albertopadovan/Documents/Presentations/WCCM26/Figures/toy_model"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "subspace_and_ssm_3d")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.35)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.35)
print(f"Saved → {outpath}.png/.pdf")


# ── Ensure clean MPI teardown ─────────────────────────────────────────────
# mpi4py's atexit handler sometimes hangs on macOS after matplotlib has
# been used; mirror the pattern from save_ssm.py.
PETSc.COMM_WORLD.Barrier()
os._exit(0)
