"""
Step 4 of the KSE periodic-SSM workflow: load the cached SSM and run
the on-/off-manifold ROM vs truth comparison + plotting.

Inputs (all produced by the upstream scripts):
    data/periodic_orbit.npz      (compute_periodic_orbit.py)
    data/eigendecomp_cache.npz   (save_eigendecomp.py)
    data/ssm_cache.npz           (save_ssm.py)

Outputs (PNG/PDF figures into results/):
    rom_vs_truth          time-series of dominant modes, on-manifold IC
    rom_vs_truth_off      time-series of dominant modes, off-manifold IC
    manifold_on           3D SSM surface with on-manifold trajectory
    manifold_off          3D SSM surface with off-manifold trajectory

No PETSc / MPI — purely numpy + scipy + matplotlib.

Run with:
    python demonstrate_ssm.py
"""

import os
from functools import partial

import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import cm

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM
from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import imex_step, integrate


res_path = "results/"


def style_3d_axes(ax):
    """Remove gray pane backgrounds from 3D axes, keep grid."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")


def savefig(fig, name, is_3d=False):
    """Save figure as both PNG and PDF."""
    kw = dict(dpi=300)
    if not is_3d:
        kw["bbox_inches"] = "tight"
    else:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(res_path + name + ".png", **kw)
    fig.savefig(res_path + name + ".pdf", **kw)
    print(f"Saved to {res_path}{name}.png/.pdf")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 12,
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
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


# %% Parameters that aren't in the cache

# ── ROM / full-system comparison ─────────────────────────────────────────────
t_end_fixed = 100.0   # integrate for 100 time units to check whether the
                     # on-manifold IC at P(s*) stays near the 2T orbit
n_t = 4000           # # time-evaluation points
rtol_rom = 1e-12
atol_rom = 1e-12
dt_truth = 1e-4      # IMEX-SSP3 step for the truth integration (KSE is stiff;
                     # CN-treats the diagonal linear part implicitly, so dt is
                     # set by accuracy on the bilinear, not stability)
# `s0_fraction` is no longer used — the on-manifold IC is now the latent
# fixed point s* (computed automatically from the polynomial dynamics).
scaling_off = 0.3  # off-manifold IC: |s0_off| as fraction of rho_domain

# ── Visualization ────────────────────────────────────────────────────────────
dominant_modes = [1, 2, 3, 4, 5]
figsize = (6, 5)
n_phi = 60
n_t_per_period = 80
alpha_3d = 0.3
clr_truth = "#2D3142"
clr_rom = "#E85D04"


# %% Load SSM cache

cache = np.load("data/ssm_cache.npz")
PS_hb = cache["PS_hb"]  # (n_terms, n_harmonics, n) complex
W_hb = cache["W_hb"]  # (n_harmonics, n, r)       complex
V_neut_hb = cache["V_neut_hb"]  # (n_harmonics, n, n_neut)  complex
W_neut_hb = cache["W_neut_hb"]  # (n_harmonics, n, n_neut)  complex
multiindices = cache["multiindices"]  # (n_terms, r)            int
Lams = cache["Lams"]  # (r,)                      complex
gs = cache["gs"]  # (n_terms, r)              complex
conj_to_linear = bool(cache["conj_to_linear_dynamics"])

nu = float(cache["nu"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
T = float(cache["T"])
nf = int(cache["nf"])
nfb = int(cache["nfb"])
r = int(cache["r"])
m = int(cache["m"])
rho_domain = float(cache["rho_domain"])

C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]

omega = 2.0 * np.pi / T

print(
    f"Loaded SSM cache: nu={nu}, n={n}, T={T:.4f}, "
    f"nf={nf}, nfb={nfb}, r={r}, m={m}, "
    f"rho_domain={rho_domain:.4f}, n_terms={len(multiindices)}"
)


# %% Build the serial, numpy-only ROM (handles encode / decode /
#    latent_space_dynamics / neutral_project).  Neutral mode is the
#    orbit-tangent (λ=0) eigenpair pulled from index 0 of the cached
#    Φ/Ψ; used only to clean up off-manifold ICs, not in the SSM
#    computation itself.
rom = SpectralSubmanifoldROM(
    multiindices=multiindices,
    Lams=Lams,
    gs=gs,
    PS=PS_hb,
    W=W_hb,
    conj_to_linear_dynamics=conj_to_linear,
    omega=omega,
    v_neutral=V_neut_hb[:, :, 0],
    w_neutral=W_neut_hb[:, :, 0],
)


# %% Truth RHS — KSE perturbation around the periodic orbit, pure numpy

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

time_ext = np.append(time_orbit, T)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_ext,
    C_ext,
    axis=1,
    kind="cubic",
    assume_sorted=True,
)


def c_star_at(t):
    return c_star_interp(t % T)


def _evaluate_quadratic_term_numpy(q1, q2):
    """``B(q1, q2)`` for KSE — same formula as the periodic-eq class."""
    j = np.arange(1, n + 1, dtype=float)
    half_N = n_pts / 2.0

    def _to_physical(c):
        spec_u = np.zeros(n_pts, dtype=complex)
        spec_ux = np.zeros(n_pts, dtype=complex)
        spec_u[1 : n + 1] = -1j * half_N * c
        spec_ux[1 : n + 1] = half_N * j * c
        spec_u[n_pts - n : n_pts] = 1j * half_N * c[::-1]
        spec_ux[n_pts - n : n_pts] = half_N * j[::-1] * c[::-1]
        return ifft(spec_u), ifft(spec_ux)

    u1, u1x = _to_physical(q1)
    u2, u2x = _to_physical(q2)
    B_spec = fft(-0.5 * (u1 * u2x + u2 * u1x))
    result = 2j * B_spec[1 : n + 1] / n_pts
    if np.isrealobj(q1) and np.isrealobj(q2):
        return result.real
    return result


def perturbation_rhs(t, v):
    """v̇ = lam*v + 2 B(c*(t), v) + B(v, v) — KSE linearised about c*(t)."""
    cs = c_star_at(t)
    return lam * v + 2.0 * _evaluate_quadratic_term_numpy(cs, v) + N_fn(v)


# %% Common grids and the 3D manifold surface (built once)

t_end = t_end_fixed
t_eval = np.linspace(0, t_end, n_t)
n_periods = max(1, int(np.ceil(t_end / T)))

theta = np.linspace(0, 2 * np.pi, n_phi)
s1 = rho_domain * np.exp(1j * theta)
t_one_period = np.linspace(0, T, n_t_per_period, endpoint=False)

# Manifold-surface evaluation only makes sense for the conjugate-pair r=1
# case where s = (z, z̄).  Skip it (and the unused V1_surf / V2_surf
# arrays) for higher r.
if r == 1:
    V1_one = np.zeros((n_phi, n_t_per_period))
    V2_one = np.zeros((n_phi, n_t_per_period))
    print(
        f"Evaluating manifold surface on {n_phi}x{n_t_per_period} grid "
        f"(tiled over {n_periods} periods) ..."
    )
    for i in range(n_phi):
        for j in range(n_t_per_period):
            s = np.array([s1[i], s1[i].conj()], dtype=complex)
            v = rom.decode(t_one_period[j], s)
            V1_one[i, j] = v[0]
            V2_one[i, j] = v[1]
    V1_surf = np.tile(V1_one, (1, n_periods))
    V2_surf = np.tile(V2_one, (1, n_periods))
    T_surf = np.concatenate(
        [t_one_period + p * T for p in range(n_periods)]
    )[None, :] * np.ones((n_phi, 1))


os.makedirs(res_path, exist_ok=True)


# %% On-manifold IC: pick s0 well inside the SSM domain, integrate ROM + truth

# Plain on-manifold IC at half the convergence radius — guaranteed inside,
# avoids any extrapolation issues at the boundary.
s0_fraction = 0.5
s0 = np.zeros(r, dtype=complex)
s0[0] = s0_fraction * rho_domain
print(
    f"\nOn-manifold IC:  s0 = {s0_fraction} · rho_domain = {s0[0].real:.6f}   "
    f"(rho_domain = {rho_domain:.4f})"
)

print(f"\nROM vs Truth (on-manifold), t_end = {t_end:.3f}")
print("  Integrating ROM ...")
S = sp.integrate.solve_ivp(
    rom.latent_space_dynamics,
    [0, t_end],
    s0,
    method="RK45",
    t_eval=t_eval,
    rtol=rtol_rom,
    atol=atol_rom,
).y

Vapp = np.zeros((n, n_t))
for i in range(n_t):
    Vapp[:, i] = c_star_at(t_eval[i]) + rom.decode(t_eval[i], S[:, i])

v0 = rom.decode(0.0, s0)
print("  Integrating truth (IMEX-SSP3 on full KSE) ...")
# Integrate the AUTONOMOUS full system  dx/dt = lam·x + N(x)  starting from
#   x(0) = v0 + c_star(0)
# using the in-house IMEX-SSP3 stepper.  The CN-treated diagonal linear part
# absorbs the stiffness; ``dt_truth`` is set by accuracy on the bilinear, not
# stability.  Recover the perturbation at each saved time by subtracting the
# orbit:  v(t) = x(t) − c_star(t).  Linearly interpolated to t_eval for plot
# consistency with the ROM.
x0_full = v0 + c_star_at(0.0)
n_steps_imex = int(round(t_end / dt_truth))
dt_truth_exact = t_end / n_steps_imex                  # exact stepping to t_end
save_every = max(1, n_steps_imex // (n_t - 1))
t_imex, X_full = integrate(
    x0_full.copy(), lam, N_fn, dt_truth_exact, n_steps_imex,
    save_every=save_every, stepper=imex_step,
)
Vtruth = np.zeros((n, n_t))
for i in range(n):
    Vtruth[i] = np.interp(t_eval, t_imex, X_full[i])


# ── On-manifold diagnostics ────────────────────────────────────────────────
# Three quick checks to localise where ROM and truth diverge:
#   (1) Did the ROM saturate onto a polynomial fixed point of g(s) = 0?
#   (2) How much larger is the truth's peak amplitude than the ROM's?
#   (3) Is the truth still growing at the end of the window (= it left the
#       1-D SSM along OTHER unstable modes, so the ROM physically cannot
#       track it)?  ROM growth ≈ 1 means it has settled; truth growth >> 1
#       means we've outrun the 1-D invariant manifold.
print("\n" + "=" * 60)
print("ON-MANIFOLD DIAGNOSTICS")
print("=" * 60)

# (1) ROM saturation: does |s(t_end)| match a non-trivial root of g(s) = 0?
s_final = float(np.abs(S[0, -1]))
g_coeffs_low_to_high = np.array(
    [complex(gs[j_idx, 0]) for j_idx in range(len(multiindices))]
)
roots = np.roots(g_coeffs_low_to_high[::-1])
fp_candidates = sorted(
    [z for z in roots if abs(z) > 0.1 * rho_domain],
    key=lambda z: abs(z),
)
nearest_fp = fp_candidates[0] if fp_candidates else None
print("(1) ROM saturation onto a polynomial fixed point of g(s)=0:")
print(f"    |s(t_end)|             = {s_final:.4f}")
if nearest_fp is not None:
    print(f"    nearest non-trivial FP = {abs(nearest_fp):.4f}  "
          f"(s = {nearest_fp.real:+.4f}{nearest_fp.imag:+.4f}j)")
    print(f"    ratio (|s|/|s_FP|)     = {s_final / abs(nearest_fp):.4f}  "
          f"(→ 1.00  ⇒  ROM converged onto the polynomial 2T orbit)")
else:
    print("    no non-trivial roots of g(s) inside the polynomial range.")

# (2) Peak amplitude: truth vs ROM
truth_max = float(np.max(np.abs(Vtruth)))
rom_max = float(np.max(np.abs(Vapp)))
print(f"\n(2) Peak amplitudes over [0, {t_end}]:")
print(f"    max |Vtruth| = {truth_max:.4f}")
print(f"    max |Vapp|   = {rom_max:.4f}")
print(f"    Truth/ROM    = {truth_max / max(rom_max, 1e-30):.3f}  "
      f"(≫ 1  ⇒  truth left the 1-D SSM)")

# (3) Truth envelope growth: late window vs early window
late_max = float(np.max(np.abs(Vtruth[:, -200:])))
early_max = float(np.max(np.abs(Vtruth[:, 200:400])))
print(f"\n(3) Truth envelope growth (late window vs early window):")
print(f"    early max (samples 200-400)  = {early_max:.4f}")
print(f"    late max  (last 200 samples) = {late_max:.4f}")
print(f"    late / early                 = "
      f"{late_max / max(early_max, 1e-30):.3f}  "
      f"(≫ 1  ⇒  still growing, has NOT saturated onto any orbit)")

# (4) Sanity: ROM should be saturated (late / early ≈ 1).
late_rom = float(np.max(np.abs(Vapp[:, -200:])))
early_rom = float(np.max(np.abs(Vapp[:, 200:400])))
print(f"\n(4) ROM envelope growth (sanity — should be ≈ 1):")
print(f"    early max     = {early_rom:.4f}")
print(f"    late max      = {late_rom:.4f}")
print(f"    late / early  = "
      f"{late_rom / max(early_rom, 1e-30):.3f}")
print("=" * 60 + "\n")


dominant_idcs = [jm - 1 for jm in dominant_modes]
fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
for i, idx in enumerate(dominant_idcs):
    ax[i].plot(t_eval, Vtruth[idx], color=clr_truth, lw=1.5, label="Truth")
    ax[i].plot(t_eval, Vapp[idx], color=clr_rom, ls="--", lw=1.5, label="ROM")
    ax[i].set_ylabel(rf"$a_{{{dominant_modes[i]}}}$")
ax[0].legend()
ax[-1].set_xlabel(r"Time $t$")
plt.tight_layout()
savefig(fig, "rom_vs_truth")
plt.show()


# %% Replay one 2T window from the settled final states.
#
# After integrating to t_end_fixed the trajectories have (presumably) landed
# on their respective limit cycles, but their last 2T-segment is not on a
# uniform sample grid (ROM's is solve_ivp + n_t = 4000; FOM's was sub-sampled
# from save_every).  Take the FINAL snapshots and integrate out for exactly
# one 2T window with dense IMEX-step granularity (save every 10 steps for
# the FOM; matching dense grid for the ROM via solve_ivp t_eval).
print("\nReplaying one 2T window from final snapshots for the phase portrait ...")

# ── ROM replay ────────────────────────────────────────────────────────────
# Start from S[:, -1] (latent state at t_end), integrate latent ODE over
# [t_end, t_end + T].  Use a dense t_eval grid so the phase portrait is
# smooth.
n_replay_pts = max(2, int(round(T / dt_truth / 10)) + 1)   # ~ save_every=10
t_rom_replay = t_end + np.linspace(0.0, T, n_replay_pts)
sol_rom_replay = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [t_end, t_end + T], S[:, -1],
    method="RK45", t_eval=t_rom_replay,
    rtol=rtol_rom, atol=atol_rom,
)
S_replay = sol_rom_replay.y
X_rom_orbit = np.zeros((n, n_replay_pts))
for i in range(n_replay_pts):
    X_rom_orbit[:, i] = (
        c_star_at(t_rom_replay[i])
        + rom.decode(t_rom_replay[i], S_replay[:, i]).real
    )

# ── FOM replay ────────────────────────────────────────────────────────────
# Start from X_full[:, -1], IMEX-step over [0, T] (autonomous), save every
# 10th step.
n_steps_fom_replay = int(round(T / dt_truth))
dt_fom_replay = T / n_steps_fom_replay
t_fom_replay, X_fom_orbit = integrate(
    X_full[:, -1].copy(), lam, N_fn, dt_fom_replay, n_steps_fom_replay,
    save_every=10, stepper=imex_step,
)


# %% Phase portrait  a_2 vs a_1  —  T-periodic orbit + 2T orbits (FOM + ROM)
#
#   Blue solid     : T-periodic orbit (= c_star, the chart-1 anchor)
#   Black solid    : FOM 2T orbit  (replay from final IMEX snapshot, 1·T)
#   Orange dashed  : ROM 2T orbit  (replay from final ROM snapshot, 1·T)

fig, ax = plt.subplots(figsize=(6.5, 5.5))

ax.plot(
    C_periodic[0], C_periodic[1],
    color="#1F4E79", lw=1.8,
    label=r"$T$-periodic orbit",
)
ax.plot(
    X_fom_orbit[0], X_fom_orbit[1],
    color="black", lw=1.0,
    label=r"FOM $2T$ orbit",
)
ax.plot(
    X_rom_orbit[0], X_rom_orbit[1],
    color=clr_rom, ls="--", lw=1.5,
    label=r"ROM $2T$ orbit",
)

ax.set_xlabel(r"$a_1$")
ax.set_ylabel(r"$a_2$")
ax.set_title(rf"Phase portrait at $1/\nu = {1/nu:.4f}$")
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "phase_portrait_a1a2")
plt.show()


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(
#     T_surf, V1_surf, V2_surf,
#     rstride=1, cstride=1,
#     cmap=cm.plasma, linewidth=0, antialiased=False, alpha=alpha_3d,
# )
# ax.plot3D(t_eval, Vtruth[0], Vtruth[1], color=clr_truth, lw=1.5, label="Truth")
# ax.plot3D(t_eval, Vapp[0], Vapp[1], color=clr_rom, ls="--", lw=1.5, label="ROM")
# style_3d_axes(ax)
# ax.set_xlabel(r"$t$")
# ax.set_ylabel(r"$v_1$")
# ax.set_zlabel(r"$v_2$")
# ax.legend()
# fig.canvas.mpl_connect(
#     "key_press_event",
#     lambda event: savefig(fig, "manifold_on", is_3d=True)
#     if event.key == "s" else None,
# )
# plt.show()


# # %% Off-manifold IC: random direction

# # Random off-manifold IC, orbit-tangent-projected so the truth doesn't
# # drift along the orbit, then scaled so |s0_off| = scaling_off * rho_domain.
# # Matches the rossler/_2T scheme exactly.
# rng = np.random.default_rng(42)
# x0_off = rng.standard_normal(n)
# x0_off = rom.neutral_project(0.0, x0_off).real
# x0_off /= np.linalg.norm(x0_off)

# s0_off = rom.encode(0.0, x0_off)
# scaling_factor = rho_domain / np.linalg.norm(s0_off) * scaling_off
# s0_off *= scaling_factor
# x0_off *= scaling_factor

# print(f"\nROM vs Truth (off-manifold)")
# print(
#     f"  |s0_off| = {np.linalg.norm(s0_off):.3f}  "
#     f"(rho_domain = {rho_domain:.3f}, ratio = "
#     f"{np.linalg.norm(s0_off) / rho_domain:.2f})"
# )
# print("  Integrating ROM ...")
# S_off = sp.integrate.solve_ivp(
#     rom.latent_space_dynamics,
#     [0, t_end],
#     s0_off,
#     method="RK45",
#     t_eval=t_eval,
#     rtol=rtol_rom,
#     atol=atol_rom,
# ).y
# S_abs_max = np.max(np.abs(S_off))
# print(
#     f"  max |s(t)| during ROM integration = {S_abs_max:.3f}  "
#     f"(rho_domain = {rho_domain:.3f})"
# )
# if S_abs_max > rho_domain:
#     print(
#         "  ⚠ latent trajectory left the convergence domain — "
#         "polynomial truncation is unreliable here."
#     )

# Vapp_off = np.zeros((n, n_t))
# for i in range(n_t):
#     Vapp_off[:, i] = rom.decode(t_eval[i], S_off[:, i])

# print("  Integrating truth ...")
# Vtruth_off = sp.integrate.solve_ivp(
#     perturbation_rhs,
#     [0, t_end],
#     x0_off,
#     method="Radau",
#     t_eval=t_eval,
#     rtol=rtol_truth,
#     atol=atol_truth,
# ).y


# # fig = plt.figure(figsize=(8, 6))
# # ax = fig.add_subplot(111, projection="3d")
# # ax.plot_surface(
# #     T_surf, V1_surf, V2_surf,
# #     rstride=1, cstride=1,
# #     cmap=cm.plasma, linewidth=0, antialiased=False, alpha=alpha_3d,
# # )
# # ax.plot3D(t_eval, Vtruth_off[0], Vtruth_off[1], color=clr_truth, lw=1.5, label="Truth")
# # ax.plot3D(t_eval, Vapp_off[0], Vapp_off[1], color=clr_rom, ls="--", lw=1.5, label="ROM")
# # style_3d_axes(ax)
# # ax.set_xlabel(r"$t$")
# # ax.set_ylabel(r"$v_1$")
# # ax.set_zlabel(r"$v_2$")
# # ax.legend()
# # fig.canvas.mpl_connect(
# #     "key_press_event",
# #     lambda event: savefig(fig, "manifold_off", is_3d=True)
# #     if event.key == "s" else None,
# # )
# # plt.show()
# # plt.close("all")


# fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
# for i, idx in enumerate(dominant_idcs):
#     ax[i].plot(t_eval, Vtruth_off[idx], color=clr_truth, lw=1.5, label="Truth")
#     ax[i].plot(
#         t_eval, Vapp_off[idx], color=clr_rom, ls="--", lw=1.5, label="ROM"
#     )
#     ax[i].set_ylabel(rf"$a_{{{dominant_modes[i]}}}$")
# ax[0].legend()
# ax[-1].set_xlabel(r"Time $t$")
# plt.tight_layout()
# savefig(fig, "rom_vs_truth_off")
# plt.show()
# plt.close("all")

os._exit(0)
