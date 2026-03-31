"""
Instantiate KuramotoSivashinskyPeriodic from the saved periodic orbit,
compute the eigendecomposition, and plot the eigenvalues.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold

from kse_differential_equation import KuramotoSivashinskyPeriodic
from spatial_operators import linear_eigenvalues, nonlinear_rhs

comm = PETSc.COMM_WORLD

# %% Load periodic orbit

data = np.load("data/periodic_orbit.npz")
nu = float(data["nu"])
n = int(data["n"])
n_pts = int(data["n_pts"])
T = float(data["T"])
C = data["C"]            # shape (n, n_orbit + 1), last column ≈ first

# Drop the duplicate endpoint so time spans [0, T)
C_periodic = C[:, :-1]
n_time = C_periodic.shape[1]
time = np.linspace(0, T, n_time, endpoint=False)

# %% Parameters

nf = 15       # positive frequencies for the SSM / HB system
nfb = 10      # positive base-flow frequencies to retain

# %% Instantiate

res4py.petscprint(comm, f"KSE periodic orbit: nu={nu}, n={n}, T={T:.6f}")
res4py.petscprint(comm, f"  n_time={n_time}, nf={nf}, nfb={nfb}")
res4py.petscprint(comm, f"  HB state dimension: {n * (2 * nf + 1)}")

eq = KuramotoSivashinskyPeriodic(
    n=n, nu=nu, nf=nf, nfb=nfb,
    c_star=C_periodic, time=time, n_pts=n_pts,
)

res4py.petscprint(comm, f"  omega = {eq.omega:.6f}")
res4py.petscprint(comm, f"  Eigenvalues computed: {len(eq.L)}")

# %% Plot eigenvalues

eigs = eq.L

if comm.getRank() == 0:
    omega = 2 * np.pi / T
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw horizontal lines at Im = +/- k*omega/2
    im_max = np.max(np.abs(eigs.imag))
    k_max = int(np.ceil(im_max / (omega / 2))) + 1
    for k in range(-k_max, k_max + 1):
        ax.axhline(
            k * omega / 2, color="red", lw=0.8, ls="-", alpha=0.25,
        )

    ax.scatter(eigs.real, eigs.imag, s=15, c="k", zorder=3)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"Re$(\lambda)$")
    ax.set_ylabel(r"Im$(\lambda)$")
    ax.set_title(
        rf"Floquet exponents of KSE periodic orbit ($\nu={nu}$, $n_f={nf}$)"
    )
    ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


# %% SSM computation (1D manifold)

r = 1       # SSM dimension (single master mode)
m = 32      # polynomial expansion order

idces = np.arange(r, dtype=np.int32)
L_ssm = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

res4py.petscprint(comm, f"\nSSM computation: r={r}, m={m}")
res4py.petscprint(comm, f"  Master eigenvalue: {np.diag(L_ssm)}")

WtV = V.dot(W)
res4py.petscprint(comm, f"  W^* V = {WtV.getDenseArray()}")
WtV.destroy()

SSM = SpectralSubmanifold(eq, r, m)
SSM.solve(V, W, L_ssm, scaling=0.2, verbose=1)

res4py.petscprint(comm, f"  SSM solved successfully.")

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
res4py.petscprint(comm, f"  Estimated convergence radius: {R:.3f}")

if comm.getRank() == 0:
    res4py.plot_convergence_radius(orders, coeff_sums, slope, intercept, R)
    plt.show()


# %% Plot the 1D manifold as a surface: s × t → projected perturbation

manifold_tol = 1e-2
percent_domain, est_error = res4py.proper_radius(manifold_tol, intercept, m)
rho_domain = percent_domain * R
res4py.petscprint(comm, f"  Using radius: {rho_domain:.3f} "
                        f"(estimated error {est_error:.3e})")

# ── Helper: decode HB vector → physical perturbation at time t ──────────
def decode_at_time(ssm, s, t):
    """Decode latent coordinate s to physical perturbation v(t)."""
    vec_hb = ssm.decode(np.asarray(s))
    vec_seq = res4py.distributed_to_sequential_vector(vec_hb)
    hb_arr = vec_seq.getArray().copy()
    vec_seq.destroy()
    vec_hb.destroy()

    nf_loc = eq.nf
    n_harmonics = 2 * nf_loc + 1
    hb_blocks = hb_arr.reshape(n_harmonics, n)
    k_idx = np.arange(-nf_loc, nf_loc + 1)
    exp_k = np.exp(1j * k_idx * eq.omega * t)
    return (exp_k @ hb_blocks).real

# ── Evaluate manifold over (s, t) grid ──────────────────────────────────
n_s = 80
n_t = 100
s_vals = np.linspace(-rho_domain, rho_domain, n_s)
t_vals = np.linspace(0, T, n_t, endpoint=False)

S_grid, T_grid = np.meshgrid(s_vals, t_vals, indexing="ij")  # (n_s, n_t)
C1 = np.zeros_like(S_grid)  # sine mode 1
C2 = np.zeros_like(S_grid)  # sine mode 2

res4py.petscprint(comm, f"\nEvaluating 1D manifold on {n_s}x{n_t} grid ...")
for i in range(n_s):
    for j in range(n_t):
        s = np.array([s_vals[i]], dtype=complex)
        v = decode_at_time(SSM, s, t_vals[j])
        C1[i, j] = v[0]
        C2[i, j] = v[1]
    if (i + 1) % 20 == 0:
        res4py.petscprint(comm, f"  s = {i+1}/{n_s}")

# ── Plot: c_1 vs c_2 vs t ───────────────────────────────────────────────
if comm.getRank() == 0:
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        C1, C2, T_grid,
        rstride=1, cstride=1,
        cmap=cm.plasma, linewidth=0, antialiased=True, alpha=0.85,
    )
    ax.set_xlabel(r"$c_1$")
    ax.set_ylabel(r"$c_2$")
    ax.set_zlabel(r"$t$")
    ax.set_title(rf"1D periodic SSM ($\nu={nu}$, $m={m}$)")
    plt.tight_layout()
    plt.savefig("data/ssm_1d_manifold.png", dpi=200)
    plt.show()


# %% ROM vs Truth comparison (on-manifold initial condition)

# ── Perturbation dynamics around the periodic orbit ─────────────────────
# v̇ = lam*v + 2*B(c*(t), v) + B(v,v)
# where c*(t) is interpolated from the stored periodic orbit.

lam = linear_eigenvalues(n, nu)
from functools import partial
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

# Interpolate c*(t) periodically from the stored orbit.
# Append the first sample at t=T to close the period for interpolation.
from scipy.interpolate import interp1d
time_extended = np.append(time, T)
C_extended = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_extended, C_extended, axis=1, kind="cubic", assume_sorted=True,
)

def c_star_at(t):
    """Evaluate c*(t) with periodic wrapping."""
    return c_star_interp(t % T)

def perturbation_rhs(t, v):
    """RHS of the perturbation equation: v̇ = lam*v + 2*B(c*(t),v) + B(v,v)."""
    cs = c_star_at(t)
    Bvv = N_fn(v)
    Bcv = eq._evaluate_quadratic_term_numpy(cs, v)
    return lam * v + 2.0 * Bcv + Bvv

# ── Initial condition on the manifold ───────────────────────────────────
s0 = np.array([rho_domain * 0.05], dtype=complex)
v0_hb = SSM.decode(s0)
v0_seq = res4py.distributed_to_sequential_vector(v0_hb)
v0_full = v0_seq.getArray().copy()
v0_seq.destroy()
v0_hb.destroy()

# IFFT to get v(t=0) in physical space
nf_loc = eq.nf
n_harmonics = 2 * nf_loc + 1
hb_blocks = v0_full.reshape(n_harmonics, n)
k_idx = np.arange(-nf_loc, nf_loc + 1)
v0 = (np.exp(1j * k_idx * eq.omega * 0.0) @ hb_blocks).real

# ── Integration parameters ──────────────────────────────────────────────
t_end = 5.0 * T
n_eval = 500
t_eval = np.linspace(0, t_end, n_eval)

# ── ROM: integrate latent-space dynamics ────────────────────────────────
res4py.petscprint(comm, f"\nROM vs Truth comparison (on-manifold IC)")
res4py.petscprint(comm, f"  s0 = {s0},  t_end = {t_end:.3f}")
res4py.petscprint(comm, "  Integrating ROM ...")

S_rom = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics, [0, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-12,
).y  # (r, n_eval)

# Decode ROM trajectory to physical space at each t
C_rom = np.zeros((n, n_eval))
for i in range(n_eval):
    C_rom[:, i] = decode_at_time(SSM, S_rom[:, i], t_eval[i])

# ── Truth: integrate perturbation dynamics ──────────────────────────────
res4py.petscprint(comm, "  Integrating truth ...")

sol_truth = sp.integrate.solve_ivp(
    perturbation_rhs, [0, t_end], v0,
    method="Radau", t_eval=t_eval, rtol=1e-10, atol=1e-10,
)
C_truth = sol_truth.y  # (n, n_eval)

res4py.petscprint(comm, "  Done.")

# ── Plot 1: time series of first few modes ──────────────────────────────
if comm.getRank() == 0:
    modes_to_plot = [1, 2, 3, 4]
    fig, axes = plt.subplots(len(modes_to_plot), 1, sharex=True, figsize=(10, 8))
    for ax, mode in zip(axes, modes_to_plot):
        ax.plot(t_eval, C_truth[mode - 1], "k", lw=1.5, label="Truth")
        ax.plot(t_eval, C_rom[mode - 1], "r--", lw=1.5, label="ROM")
        ax.set_ylabel(rf"$c_{{{mode}}}$")
    axes[0].legend()
    axes[0].set_title(rf"On-manifold: ROM vs Truth ($\nu={nu}$, $m={m}$)")
    axes[-1].set_xlabel(r"$t$")
    plt.tight_layout()
    plt.savefig("data/ssm_rom_vs_truth_timeseries.png", dpi=200)
    plt.show()

# ── Plot 2: trajectories over the manifold (c1 vs c2 vs t) ─────────────
if comm.getRank() == 0:
    from matplotlib import cm

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Tile the manifold surface over the full integration window
    n_periods = int(np.ceil(t_end / T))
    for p in range(n_periods):
        t_offset = p * T
        T_tile = T_grid + t_offset
        # Clip to t_end
        mask = T_tile <= t_end + 1e-10
        if not mask.any():
            break
        ax.plot_surface(
            T_tile, C1, C2,
            rstride=2, cstride=2,
            cmap=cm.plasma, linewidth=0, antialiased=True, alpha=0.2,
        )

    # Truth trajectory
    ax.plot3D(
        t_eval, C_truth[0], C_truth[1],
        "k", lw=1.5, label="Truth",
    )
    # ROM trajectory
    ax.plot3D(
        t_eval, C_rom[0], C_rom[1],
        "r--", lw=1.5, label="ROM",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$c_1$")
    ax.set_zlabel(r"$c_2$")
    ax.set_title(rf"Trajectories on SSM ($\nu={nu}$, $m={m}$)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/ssm_rom_vs_truth_manifold.png", dpi=200)
    plt.show()
