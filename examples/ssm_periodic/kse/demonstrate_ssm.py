import os
import numpy as np
import scipy as sp
from functools import partial
from scipy.interpolate import interp1d
from kse_differential_equation import KuramotoSivashinskyPeriodic
from spatial_operators import linear_eigenvalues, nonlinear_rhs

import matplotlib.pyplot as plt
from matplotlib import cm

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold

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


# %% Parameters

# ── Harmonic-balanced / SSM truncation ──────────────────────────────────────
nf = 17             # number of positive frequencies for the SSM / HB system
nfb = 12            # number of positive base-flow frequencies to retain
r = 2               # SSM dimension (number of master modes)
m = 9              # polynomial expansion order
ssm_scaling = 0.2   # coordinate scaling applied during SSM solve
manifold_tol = 1e-2 # truncation tolerance for estimating the valid SSM domain

# ── ROM / full-system comparison ─────────────────────────────────────────────
n_periods = 10      # number of periods over which to compare ROM vs truth
n_t = 2500          # number of time-evaluation points
rtol_rom = 1e-12
atol_rom = 1e-12
rtol_truth = 1e-10
atol_truth = 1e-10
scaling_off = 0.8   # scaling for off-manifold perturbation (relative to rho_domain)

# ── Visualization ────────────────────────────────────────────────────────────
dominant_modes = [1, 2, 3, 4, 5]      # Fourier mode indices for time-series plots
figsize = (6, 5)
n_rho = 40
n_theta = 40
alpha_3d = 0.3
clr_truth = "#2D3142"           # dark charcoal
clr_rom = "#E85D04"             # burnt orange

comm = PETSc.COMM_WORLD


# %% Load periodic orbit

data = np.load("data/periodic_orbit.npz")
nu = float(data["nu"])
n = int(data["n"])
n_pts = int(data["n_pts"])
T = float(data["T"])
C_orbit = data["C"]                # shape (n, n_orbit + 1), last col ≈ first

# Drop duplicate endpoint so the samples span [0, T)
C_periodic = C_orbit[:, :-1]
n_time = C_periodic.shape[1]
time_orbit = np.linspace(0, T, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"KSE periodic orbit: nu={nu}, n={n}, T={T:.6f}, "
    f"n_time={n_time}, nf={nf}, nfb={nfb}",
)

eq = KuramotoSivashinskyPeriodic(
    n=n, nu=nu, nf=nf, nfb=nfb,
    c_star=C_periodic, time=time_orbit, n_pts=n_pts,
)

res4py.petscprint(comm, f"omega = {eq.omega:.6f}")
res4py.petscprint(comm, f"HB state dim = {n * (2 * nf + 1)}")


# %% SSM computation

idces = np.arange(r, dtype=np.int32)
L = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

SSM = SpectralSubmanifold(eq, r, m)
SSM.solve(V, W, L, scaling=ssm_scaling, verbose=1)

res4py.petscprint(comm, f"Dominant eigenvalues: {SSM.Lams}")

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
res4py.petscprint(comm, f"Estimated convergence radius: {R:.3f}")
percent_domain, est_error = res4py.proper_radius(manifold_tol, intercept, m)
rho_domain = percent_domain * R
res4py.petscprint(
    comm,
    f"Using radius: {rho_domain:.3f} for estimated error {est_error:.3f}",
)

if comm.getRank() == 0:
    os.makedirs(res_path, exist_ok=True)
    res4py.plot_convergence_radius(orders, coeff_sums, slope, intercept, R)
    plt.tight_layout()
    savefig(plt.gcf(), "convergence_radius")
    plt.show()


# %% Helpers: physical-space encode / decode for the time-periodic SSM
#
# ``ssm.decode`` / ``ssm.encode`` work only for time-invariant SSMs.  For a
# time-periodic SSM the polynomial coefficients ``ssm.ps`` and left
# eigenvectors ``ssm.W`` live in HB (Fourier) space — each vector has
# dimension ``n * (2*nf + 1)``.  Let
#
#     E(t) v_hb = sum_k v_k exp(i k omega t)
#
# denote the IFFT (HB → physical at time ``t``).  Then:
#   * physical decoder: x(t) = sum_{idx} s^{j(idx)} * E(t) p_idx
#   * physical encoder at time t: s = W_phys(t)^H x, where
#                                  W_phys(t) = E(t) applied column-wise to W.

n_harmonics = 2 * nf + 1
k_idx = np.arange(-nf, nf + 1)


def _gather_hb_vec(vec_hb):
    """Gather a distributed HB PETSc vec and reshape to ``(n_harmonics, n)``."""
    vec_seq = res4py.distributed_to_sequential_vector(vec_hb)
    arr = vec_seq.getArray().copy().reshape(n_harmonics, n)
    vec_seq.destroy()
    return arr


def _gather_hb_bv(bv):
    """Gather an HB SLEPc BV to a numpy array of shape ``(n_harmonics, n, ncols)``."""
    ncols = bv.getSizes()[-1]
    out = np.zeros((n_harmonics, n, ncols), dtype=complex)
    for j in range(ncols):
        col = bv.getColumn(j)
        out[:, :, j] = _gather_hb_vec(col)
        bv.restoreColumn(j, col)
    return out


# Pre-gather the HB polynomial coefficients and left eigenvectors so we
# only pay the communication cost once.  Shapes:
#   PS_hb   : (n_terms, n_harmonics, n)
#   W_hb    : (n_harmonics, n, r)
PS_hb = np.stack([_gather_hb_vec(p) for p in SSM.ps], axis=0)
W_hb = _gather_hb_bv(SSM.W)

# Pre-gather the neutral Floquet bases (stored inside the projection operator
# as ``L.L.U`` and ``L.L.V`` — they live in HB space).
V_neut_hb = _gather_hb_bv(eq._neutral_proj.L.L.U)
W_neut_hb = _gather_hb_bv(eq._neutral_proj.L.L.V)


def _ifft_weights(t):
    """``exp(i k omega t)`` for k in ``[-nf, ..., nf]``."""
    return np.exp(1j * k_idx * eq.omega * t)


def decode_phys(s, t):
    """Physical-space decoder: ``x(t) = sum_idx s^{j(idx)} E(t) ps[idx]``."""
    s = np.asarray(s)
    coefs = np.array(
        [np.prod(s ** np.asarray(j)) for j in SSM.ssm_multiindices],
        dtype=complex,
    )
    hb = np.einsum("i,ihj->hj", coefs, PS_hb)      # (n_harmonics, n)
    return (_ifft_weights(t) @ hb).real            # (n,)


def encode_phys(x0, t):
    """Physical-space encoder: ``s = W_phys(t)^H x0``."""
    W_phys = np.einsum("h,hjr->jr", _ifft_weights(t), W_hb)  # (n, r)
    return W_phys.conj().T @ x0


# The σ = 0 neutral Floquet mode (orbit-tangent) is stored at column 0 of
# the neutral bases by construction in ``_compute_neutral_eigentriples``.
# The other 10 columns are Floquet modes at shifts ``±kiω`` — in HB they are
# linearly independent, but after IFFT at a single time ``t`` they collapse
# to the same physical direction, making the 11×11 oblique-projection Gram
# rank-deficient.  Projecting with only the σ = 0 column avoids this.
v_neut_hb = V_neut_hb[:, :, 0]
w_neut_hb = W_neut_hb[:, :, 0]


def neutral_project_phys(x0, t):
    """Remove the orbit-tangent direction from ``x0`` at time ``t``.

    Uses only the ``σ = 0`` neutral mode ``(v, w)`` IFFT-ed at time ``t``:
    ``x - v (w^H x) / (w^H v)``.  ``v`` and ``w`` are biorthogonal in HB
    and — for the ``σ = 0`` pair — remain biorthogonal pointwise in
    physical space.
    """
    w_ifft = _ifft_weights(t)
    v_t = w_ifft @ v_neut_hb
    w_t = w_ifft @ w_neut_hb
    return x0 - v_t * (np.vdot(w_t, x0) / np.vdot(w_t, v_t))


# %% Perturbation dynamics around the periodic orbit (truth)

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

time_ext = np.append(time_orbit, T)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return c_star_interp(t % T)


def perturbation_rhs(t, v):
    """v̇ = lam*v + 2*B(c*(t), v) + B(v, v)."""
    cs = c_star_at(t)
    return lam * v + 2.0 * eq._evaluate_quadratic_term_numpy(cs, v) + N_fn(v)


# %% ROM vs Truth comparison (on-manifold IC)

t_end = n_periods * T
t_eval = np.linspace(0, t_end, n_t)

# Stay well inside the SSM convergence radius: ``rho_domain`` is the
# radius where the *estimated* truncation error equals ``manifold_tol``
# (here 1e-2).  At ``|s| = rho_domain`` that ~1% error is visible on the
# plot.  Dropping to ``|s| ≈ 0.3·rho_domain`` shrinks the polynomial
# truncation by ``(0.3)^(m+1)`` (~six orders of magnitude at m=9) and
# the encoder's higher-order leakage by ``O(|s|^2)``.
s0_fraction = 0.3
s0 = s0_fraction * rho_domain * np.array([1.0, 1.0], dtype=complex)

res4py.petscprint(comm, f"\nROM vs Truth (on-manifold), t_end = {t_end:.3f}")
res4py.petscprint(comm, "  Integrating ROM ...")
S = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t_end], s0, method="RK45", t_eval=t_eval,
    rtol=rtol_rom, atol=atol_rom,
).y

Vapp = np.zeros((n, n_t))
for i in range(n_t):
    Vapp[:, i] = decode_phys(S[:, i], t_eval[i])

# Truth: integrate perturbation dynamics in physical space
v0 = decode_phys(s0, 0.0)

res4py.petscprint(comm, "  Integrating truth ...")
Vtruth = sp.integrate.solve_ivp(
    perturbation_rhs, [0, t_end], v0, method="Radau", t_eval=t_eval,
    rtol=rtol_truth, atol=atol_truth,
).y

if comm.getRank() == 0:
    dominant_idcs = [jm - 1 for jm in dominant_modes]
    fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
    for i, idx in enumerate(dominant_idcs):
        ax[i].plot(t_eval, Vtruth[idx], color=clr_truth, lw=1.5, label="Truth")
        ax[i].plot(t_eval, Vapp[idx], color=clr_rom, ls="--", lw=1.5, label="ROM")
        ax[i].set_ylabel(rf"$v_{{{dominant_modes[i]}}}$")
    ax[0].legend()
    ax[-1].set_xlabel(r"Time $t$")
    plt.tight_layout()
    savefig(fig, "rom_vs_truth")
    plt.show()


# %% 3D manifold: evaluate (v_1, v_2, t) perturbation surface over the SSM domain

n_s = 60
n_phi = 60
n_t_per_period = 80

theta = np.linspace(0, 2 * np.pi, n_phi)
s1 = rho_domain * np.exp(1j * theta)

# Evaluate the perturbation SSM surface over one period and tile it across
# ``n_periods`` (``decode_phys(s, t)`` is ``T``-periodic in ``t``).
t_one_period = np.linspace(0, T, n_t_per_period, endpoint=False)

V1_one = np.zeros((n_phi, n_t_per_period))
V2_one = np.zeros((n_phi, n_t_per_period))

res4py.petscprint(
    comm,
    f"\nEvaluating manifold surface on {n_phi}x{n_t_per_period} grid "
    f"(tiled over {n_periods} periods) ...",
)
for i in range(n_phi):
    for j in range(n_t_per_period):
        s = np.array([s1[i], s1[i].conj()], dtype=complex)
        v = decode_phys(s, t_one_period[j])
        V1_one[i, j] = v[0]
        V2_one[i, j] = v[1]

V1_surf = np.tile(V1_one, (1, n_periods))
V2_surf = np.tile(V2_one, (1, n_periods))
T_surf = np.concatenate(
    [t_one_period + p * T for p in range(n_periods)]
)[None, :] * np.ones((n_phi, 1))

if comm.getRank() == 0:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        T_surf, V1_surf, V2_surf,
        rstride=1, cstride=1,
        cmap=cm.plasma, linewidth=0, antialiased=False, alpha=alpha_3d,
    )
    ax.plot3D(t_eval, Vtruth[0], Vtruth[1], color=clr_truth, lw=1.5, label="Truth")
    ax.plot3D(t_eval, Vapp[0], Vapp[1], color=clr_rom, ls="--", lw=1.5, label="ROM")
    style_3d_axes(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v_1$")
    ax.set_zlabel(r"$v_2$")
    ax.legend()
    fig_3d = ax.get_figure()
    fig_3d.canvas.mpl_connect(
        "key_press_event",
        lambda event: savefig(fig_3d, "manifold_on", is_3d=True)
        if event.key == "s"
        else None,
    )
    plt.show()


# %% Off-manifold initial condition
#
# Random physical-space IC.  Project out the neutral Floquet content at
# t = 0 (the neutral projection is defined in HB space; we IFFT its bases
# and apply the oblique projector in physical space) so the truth does
# not drift in orbit phase relative to the ROM, which already lives in
# the complement of the neutral directions.

rng = np.random.default_rng(42)
x0_off = rng.standard_normal(n)
x0_off = neutral_project_phys(x0_off, 0.0).real
x0_off /= np.linalg.norm(x0_off)

# Encode via the physical-space encoder at t = 0, then scale so the
# encoded latent coordinates have norm ``rho_domain * scaling_off``.
s0_off = encode_phys(x0_off, 0.0)
scaling_factor = rho_domain / np.linalg.norm(s0_off) * scaling_off
s0_off *= scaling_factor
x0_off *= scaling_factor

res4py.petscprint(comm, f"\nROM vs Truth (off-manifold)")
res4py.petscprint(comm, "  Integrating ROM ...")
S_off = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t_end], s0_off, method="RK45", t_eval=t_eval,
    rtol=rtol_rom, atol=atol_rom,
).y

Vapp_off = np.zeros((n, n_t))
for i in range(n_t):
    Vapp_off[:, i] = decode_phys(S_off[:, i], t_eval[i])

v0_off = x0_off

res4py.petscprint(comm, "  Integrating truth ...")
Vtruth_off = sp.integrate.solve_ivp(
    perturbation_rhs, [0, t_end], v0_off, method="Radau", t_eval=t_eval,
    rtol=rtol_truth, atol=atol_truth,
).y

if comm.getRank() == 0:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        T_surf, V1_surf, V2_surf,
        rstride=1, cstride=1,
        cmap=cm.plasma, linewidth=0, antialiased=False, alpha=alpha_3d,
    )
    ax.plot3D(t_eval, Vtruth_off[0], Vtruth_off[1], color=clr_truth, lw=1.5, label="Truth")
    ax.plot3D(t_eval, Vapp_off[0], Vapp_off[1], color=clr_rom, ls="--", lw=1.5, label="ROM")
    style_3d_axes(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v_1$")
    ax.set_zlabel(r"$v_2$")
    ax.legend()
    fig_3d_off = ax.get_figure()
    fig_3d_off.canvas.mpl_connect(
        "key_press_event",
        lambda event: savefig(fig_3d_off, "manifold_off", is_3d=True)
        if event.key == "s"
        else None,
    )
    plt.show()
    plt.close('all')

if comm.getRank() == 0:
    dominant_idcs = [jm - 1 for jm in dominant_modes]
    fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
    for i, idx in enumerate(dominant_idcs):
        ax[i].plot(t_eval, Vtruth_off[idx], color=clr_truth, lw=1.5, label="Truth")
        ax[i].plot(t_eval, Vapp_off[idx], color=clr_rom, ls="--", lw=1.5, label="ROM")
        ax[i].set_ylabel(rf"$v_{{{dominant_modes[i]}}}$")
    ax[0].legend()
    ax[-1].set_xlabel(r"Time $t$")
    plt.tight_layout()
    savefig(fig, "rom_vs_truth_off")
    plt.show()
    plt.close('all')


# Force clean termination: synchronise ranks, then kill the process bypassing
# Python/MPI/matplotlib shutdown hooks (these sometimes hang after plt.show()
# with PETSc initialised).
comm.Barrier()
os._exit(0)
