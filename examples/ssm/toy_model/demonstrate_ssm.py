import os

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from petsc4py import PETSc
import resolvent4py as res4py

from toy_model_differential_equation import Hopf3D
import plotting_utils as plotting

comm = PETSc.COMM_WORLD


diff_eq = Hopf3D(mu=-1 / 20, alpha=0.15, beta=0.0)
L, Phi, Psi = diff_eq.compute_eigendecomposition()


r = 2
m = 30

idces = [0, 1]
V = res4py.bv_slice(Phi, idces)
W = res4py.bv_slice(Psi, idces)
L = np.diag(L)[idces]

# Compute the manifold
SSM = res4py.spectral_submanifold.SpectralSubmanifold(diff_eq, r, m, conj_to_linear_dynamics=False)
SSM.solve(V, W, L, scaling=0.2)

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
res4py.petscprint(comm, f"Estimated convergence radius: {R:.3f}")
if comm.getRank() == 0:
    ax = res4py.plot_convergence_radius(orders, coeff_sums, slope, intercept, R)
    plt.show()


# Compute on-manifold trajectory using
# full system and latent-space dynamics
rho_domain = 0.50 * R
rho = rho_domain
th = np.pi / 2
num = rho * np.exp(1j * th)
s0 = np.asarray([num, np.conj(num)])
t = np.linspace(0, 100, num=400)

S = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t[-1]],
    s0,
    "RK45",
    t_eval=t,
    rtol=1e-12,
    atol=1e-12,
).y

Qapprox = np.zeros((3, len(t)))
for i in range(len(t)):
    q = SSM.decode(S[:, i])
    qseq = res4py.distributed_to_sequential_vector(q)
    Qapprox[:, i] = qseq.getArray().real
    qseq.destroy()
    q.destroy()

q0 = Qapprox[:, 0]
Q = sp.integrate.solve_ivp(
    diff_eq.evaluate_dynamics_numpy,
    [0, t[-1]],
    q0.real,
    "RK45",
    t_eval=t,
    rtol=1e-12,
    atol=1e-12,
).y

# Timeseries plot
if comm.getRank() == 0:

    ylabels = [r'$x$', r'$y$', r'$z$']
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range (3):
        if i == 0:
            ax[i].plot(t, Q[i], 'k', label=r'Truth')
            ax[i].plot(t, Qapprox[i], 'r--', label=r'ROM')
        else:
            ax[i].plot(t, Q[i], 'k')
            ax[i].plot(t, Qapprox[i], 'r--')
        ax[i].set_ylabel(ylabels[i])
        if i == 2:
            ax[i].set_xlabel(r'Time $t$')
    ax[0].legend()
    plt.tight_layout()
    plt.show()

# 3D plot
ax = plotting.plot_manifold_3d(SSM, rho_domain)
if ax is not None:
    ax.plot3D(*Q[:3], color="k", lw=2, label=r"Truth")
    ax.plot3D(*Qapprox[:3], color="r", ls="--", lw=2, label=r"ROM")
    ax.legend()
    plt.show()


# Compute off-manifold trajectory using
# full system and latent-space dynamics

q0d = res4py.generate_random_petsc_vector(diff_eq.get_state_dimension())
q0d.scale(1 / q0d.norm())
s0 = SSM.encode(q0d)
scaling = rho_domain / np.linalg.norm(s0)
s0 *= scaling
q0d.scale(scaling)

S = sp.integrate.solve_ivp(
    SSM.latent_space_dynamics,
    [0, t[-1]],
    s0,
    "RK45",
    t_eval=t,
    rtol=1e-12,
    atol=1e-12,
).y

Qapprox = np.zeros((3, len(t)))
for i in range(len(t)):
    q = SSM.decode(S[:, i])
    qseq = res4py.distributed_to_sequential_vector(q)
    Qapprox[:, i] = qseq.getArray().real
    qseq.destroy()
    q.destroy()

q0 = res4py.distributed_to_sequential_vector(q0d)
Q = sp.integrate.solve_ivp(
    diff_eq.evaluate_dynamics_numpy,
    [0, t[-1]],
    q0.getArray().real,
    "RK45",
    t_eval=t,
    rtol=1e-12,
    atol=1e-12,
).y

# Timeseries plot
if comm.getRank() == 0:

    ylabels = [r'$x$', r'$y$', r'$z$']
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range (3):
        if i == 0:
            ax[i].plot(t, Q[i], 'k', label=r'Truth')
            ax[i].plot(t, Qapprox[i], 'r--', label=r'ROM')
        else:
            ax[i].plot(t, Q[i], 'k')
            ax[i].plot(t, Qapprox[i], 'r--')
        ax[i].set_ylabel(ylabels[i])
        if i == 2:
            ax[i].set_xlabel(r'Time $t$')
    ax[0].legend()
    plt.tight_layout()
    plt.show()

# 3D plot
ax = plotting.plot_manifold_3d(SSM, rho_domain)
if ax is not None:
    ax.plot3D(*Q[:3], color="k", lw=2, label=r"Truth")
    ax.plot3D(*Qapprox[:3], color="r", ls="--", lw=2, label=r"ROM")
    ax.legend()
    plt.show()



res4py.petscprint(comm, "Program is done executing.")
res4py.petscprint(comm, " ")
res4py.petscprint(comm, " ")

os._exit(0)
