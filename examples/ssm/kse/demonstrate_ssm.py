import sys
import numpy as np
import scipy as sp
from functools import partial
from kse_differential_equation import KuramotoSivashinsky
from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import integrate

import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py
from resolvent4py.spectral_submanifold import SpectralSubmanifold
import plotting_utils as plotting


# %% Parameters

# ── PDE / spatial discretization ────────────────────────────────────────────
nu = 0.061      # KSE viscosity parameter
n = 64          # number of Fourier sine modes
n_pts = 4 * n   # physical-space grid resolution for nonlinear term evaluation

# ── Time-marching to equilibrium ─────────────────────────────────────────────
ic_seed = 42        # RNG seed for random initial condition
ic_scale = 1e-3     # amplitude of random initial condition
dt = 5e-3           # time step
T = 200.0           # total integration time (long enough to reach equilibrium)
save_interval = 1.0 # time between saved snapshots

# ── SSM computation ──────────────────────────────────────────────────────────
r = 2               # SSM dimension (number of master modes)
m = 32              # polynomial expansion order
ctld=False          # linear latent-space dynamics (True) or nonlinear (False)
ssm_scaling = 0.5   # coordinate scaling applied during SSM solve
manifold_tol = 1e-2 # truncation tolerance for estimating the valid SSM domain

# ── ROM / full-system comparison ─────────────────────────────────────────────
t_end = 5.0         # end time for ROM vs truth integration
n_t = 500           # number of time-evaluation points
rtol_rom = 1e-12    # relative tolerance for ROM ODE solver (RK45)
atol_rom = 1e-12    # absolute tolerance for ROM ODE solver
rtol_truth = 1e-10  # relative tolerance for truth ODE solver (Radau)
atol_truth = 1e-10  # absolute tolerance for truth ODE solver
off_seed = 123      # RNG seed for off-manifold initial condition
scaling_off = 1.65   # scaling for off-manifold perturbation (relative to SSM domain )

# ── Visualization ────────────────────────────────────────────────────────────
dominant_modes = [1, 2, 3, 4]  # Fourier mode indices for time-series plots
figsize = (10, 8)               # figure size for time-series plots
n_rho = 40                      # radial grid points for 3D manifold surface
n_theta = 40                    # angular grid points for 3D manifold surface
alpha_3d = 0.3                  # transparency of 3D manifold surface

comm = PETSc.COMM_WORLD

# %% Time-march to equilibrium

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

rng = np.random.default_rng(ic_seed)
c0 = ic_scale * rng.standard_normal(n)

n_steps = int(T / dt)
save_every = int(save_interval / dt)

print("Time-marching to equilibrium...")
t_march, C_march = integrate(c0, lam, N_fn, dt, n_steps, save_every=save_every)
c_star = C_march[:, -1]

residual = np.linalg.norm(lam * c_star + N_fn(c_star))
print(f"Equilibrium residual ||L*c* + N(c*)||: {residual:.6e}")

eq_tol = 1e-6
if comm.getRank() == 0:
    if residual > eq_tol:
        print(f"ERROR: System did not converge to a steady state for nu={nu}.")
        print(f"Residual {residual:.6e} exceeds tolerance {eq_tol:.6e}.")
        print(
            "Try increasing T or adjusting nu to a regime with a stable fixed point."
        )
        sys.exit(1)

    print("Steady state found.")


# Setup and validation

eq = KuramotoSivashinsky(n=n, nu=nu, n_pts=n_pts, c_star=c_star)


#%% SSM computation

idces = np.arange(r, dtype=np.int32)
L = eq.L[idces]
V = res4py.bv_slice(eq.Phi, idces)
W = res4py.bv_slice(eq.Psi, idces)

SSM = SpectralSubmanifold(eq, r, m, )
SSM.solve(V, W, L, scaling=ssm_scaling)

res4py.petscprint(comm, f"Dominant eigenvalues: {SSM.Lams}")

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
res4py.petscprint(comm, f"Estimated convergence radius: {R:.3f}")
percent_domain,est_error = res4py.proper_radius(manifold_tol, intercept, m)
rho_domain = percent_domain * R
res4py.petscprint(comm, f"Using radius: {rho_domain:.3f} for estimated error {est_error:.3f}")

if comm.getRank() == 0:
    res4py.plot_convergence_radius(orders, coeff_sums, slope, intercept, R)
    plt.show()


# %% ROM vs Truth comparison

s0 = np.array([rho_domain, rho_domain], dtype=complex)
t = np.linspace(0, t_end, num=n_t)

S = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t[-1]],
    s0,
    "RK45",
    t_eval=t,
    rtol=rtol_rom,
    atol=atol_rom,
).y

# Decode ROM: v(t) from latent space, then c(t) = c_star + v(t)
Vapp = np.zeros((n, len(t)))
for i in range(len(t)):
    vec = SSM.decode(S[:, i])
    vec_seq = res4py.distributed_to_sequential_vector(vec)
    Vapp[:, i] = vec_seq.getArray().real
    vec_seq.destroy()
    vec.destroy()
Qapp = c_star[:, None] + Vapp

# Truth: integrate perturbation dynamics, then c(t) = c_star + v(t)
vec0 = SSM.decode(s0)
vec0_seq = res4py.distributed_to_sequential_vector(vec0)
v0 = vec0_seq.getArray().real.copy()
vec0_seq.destroy()
vec0.destroy()
Vtruth = sp.integrate.solve_ivp(
    eq.evaluate_dynamics_numpy,
    [0, t[-1]],
    v0,
    "Radau",
    t_eval=t,
    rtol=rtol_truth,
    atol=atol_truth,
).y
Q = c_star[:, None] + Vtruth

if comm.getRank() == 0:
    dominant_idcs = [jm - 1 for jm in dominant_modes]
    fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
    for i, idx in enumerate(dominant_idcs):
        ax[i].plot(t, Q[idx], "k", lw=1.5, label="Truth")
        ax[i].plot(t, Qapp[idx], "r--", lw=1.5, label="ROM")
        ax[i].set_ylabel(rf"$c_{{{dominant_modes[i]}}}$")
    ax[0].legend()
    ax[-1].set_xlabel(r"Time $t$")
    plt.tight_layout()
    plt.show()

# 3D manifold + trajectories projected onto eigenspace basis
ax, B, labels = plotting.plot_manifold_3d(
    eq.Psi, eq.L, SSM, rho_max=rho_domain,
    n_rho=n_rho, n_theta=n_theta, surface_alpha=alpha_3d,
)

# Project ROM and truth perturbation trajectories onto the same basis
Vapp_proj = np.zeros((3, len(t)))
Vtruth_proj = np.zeros((3, len(t)))
for i in range(len(t)):
    Vapp_proj[:, i] = plotting.project_to_3d(B, Vapp[:, i])
    Vtruth_proj[:, i] = plotting.project_to_3d(B, Vtruth[:, i])

if comm.getRank() == 0:
    ax.plot3D(*Vtruth_proj, color="k", lw=2, label="Truth")
    ax.plot3D(*Vapp_proj, color="r", ls="--", lw=2, label="ROM")
    ax.legend()
    plt.show()


# %% Off-manifold initial condition

v0_dist = res4py.generate_random_petsc_vector(eq.get_state_dimension())
v0_dist.scale(1.0 / v0_dist.norm())
s0_off = SSM.encode(v0_dist)
scaling_factor = rho_domain / np.linalg.norm(s0_off)
s0_off *= scaling_factor
v0_dist.scale(scaling_factor)

S_off = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t[-1]],
    s0_off,
    "RK45",
    t_eval=t,
    rtol=rtol_rom,
    atol=atol_rom,
).y

Vapp_off = np.zeros((n, len(t)))
for i in range(len(t)):
    vec = SSM.decode(S_off[:, i])
    vec_seq = res4py.distributed_to_sequential_vector(vec)
    Vapp_off[:, i] = vec_seq.getArray().real
    vec_seq.destroy()
    vec.destroy()

v0_seq = res4py.distributed_to_sequential_vector(v0_dist)
v0_off = v0_seq.getArray().real.copy()
v0_seq.destroy()
v0_dist.destroy()

Vtruth_off = sp.integrate.solve_ivp(
    eq.evaluate_dynamics_numpy,
    [0, t[-1]],
    v0_off,
    "Radau",
    t_eval=t,
    rtol=rtol_truth,
    atol=atol_truth,
).y

# 3D plot with manifold
Vapp_off_proj = np.zeros((3, len(t)))
Vtruth_off_proj = np.zeros((3, len(t)))
for i in range(len(t)):
    Vapp_off_proj[:, i] = plotting.project_to_3d(B, Vapp_off[:, i])
    Vtruth_off_proj[:, i] = plotting.project_to_3d(B, Vtruth_off[:, i])

ax_off, _, _ = plotting.plot_manifold_3d(
    eq.Psi, eq.L, SSM, rho_max=rho_domain,
    n_rho=n_rho, n_theta=n_theta, surface_alpha=alpha_3d,
)
if comm.getRank() == 0:
    ax_off.plot3D(*Vtruth_off_proj, color="k", lw=2, label="Truth")
    ax_off.plot3D(*Vapp_off_proj, color="r", ls="--", lw=2, label="ROM")
    ax_off.legend()
    plt.show()

Qapp_off = c_star[:, None] + Vapp_off
Q_off = c_star[:, None] + Vtruth_off
if comm.getRank() == 0:
    dominant_idcs = [jm - 1 for jm in dominant_modes]
    fig, ax = plt.subplots(len(dominant_modes), 1, sharex=True, figsize=figsize)
    for i, idx in enumerate(dominant_idcs):
        ax[i].plot(t, Q_off[idx], "k", lw=1.5, label="Truth")
        ax[i].plot(t, Qapp_off[idx], "r--", lw=1.5, label="ROM")
        ax[i].set_ylabel(rf"$c_{{{dominant_modes[i]}}}$")
    ax[0].legend()
    ax[-1].set_xlabel(r"Time $t$")
    plt.tight_layout()
    plt.show()