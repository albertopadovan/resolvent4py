r"""
Resolvent Analysis Demonstration via Time Stepping
==================================================

Given the linear dynamics :math:`d_t q = Aq`, we perform resolvent analysis
by computing the singular value decomposition (SVD) of the resolvent operator

.. math::

    R(i\omega) = \left(i\omega I - A\right)^{-1}

with :math:`\omega = 0.648` the natural frequency of the linearized CGL
equation. This script demonstrates the following:

- Resolvent analysis using time-stepping via
  :func:`~resolvent4py.linalg.resolvent_analysis_time_stepping.resolvent_analysis_rsvd_dt`

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from petsc4py import PETSc

import cgl

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)

comm = PETSc.COMM_WORLD

# Read the A matrix from file
res4py.petscprint(comm, "Reading matrix from file...")
load_path = "data/"
N = 2000
Nl = res4py.compute_local_size(N)
sizes = ((Nl, N), (Nl, N))
names = [
    load_path + "rows.dat",
    load_path + "cols.dat",
    load_path + "vals.dat",
]
A = res4py.read_coo_matrix(names, sizes)

# Compute the svd
res4py.petscprint(comm, "Running randomized SVD (algebraic)...")
omega = 0.648
n_rand = 2
n_loops = 1
n_svals = 2
Rinv = res4py.create_AIJ_identity(comm, sizes)
Rinv.scale(-1j * omega)
Rinv.axpy(-1.0, A)
ksp = res4py.create_mumps_solver(Rinv)
res4py.check_lu_factorization(Rinv, ksp)
L = res4py.linear_operators.MatrixLinearOperator(Rinv, ksp)
Ua, Sa, Va = res4py.linalg.randomized_svd(
    L, L.solve_mat, n_rand, n_loops, n_svals
)
Sa = np.diag(Sa)

res4py.petscprint(comm, "Running randomized SVD (time stepping)...")
res4py.petscprint(comm, "This may take several minutes...")
n_omegas = 1
n_periods = 100
dt = 1e-4
tol = 1e-3
verbose = 2
L = res4py.linear_operators.MatrixLinearOperator(A)
U, S, V = (
    res4py.linalg.resolvent_analysis_time_stepping.resolvent_analysis_rsvd_dt(
        L,
        dt,
        omega,
        n_omegas,
        n_periods,
        n_rand,
        n_loops,
        n_svals,
        tol,
        verbose,
    )
)
St = np.diag(S[-1])
Ut = U[-1]
Vt = V[-1]

idx = 0
bvs = [Ua, Ut, Va, Vt]
arrays = []
for bv in bvs:
    vec = bv.getColumn(idx)
    vecseq = res4py.distributed_to_sequential_vector(vec)
    bv.restoreColumn(idx, vec)
    arrays.append(vecseq.getArray().copy())
    vecseq.destroy()

if comm.getRank() == 0:
    save_path = "results/"
    os.makedirs(save_path) if not os.path.exists(save_path) else None

    l = 30 * 2
    x = np.linspace(-l / 2, l / 2, num=N, endpoint=True)
    nu = 1.0 * (2 + 0.4 * 1j)
    gamma = 1 - 1j
    mu0 = 0.38
    mu2 = -0.01
    sigma = 0.4
    system = cgl.CGL(x, nu, gamma, mu0, mu2, sigma)

    plt.figure()
    plt.plot(Sa.real, "ko", label="rsvd")
    plt.plot(St.real, "rx", label="rsvd-dt")
    ax = plt.gca()
    ax.set_xlabel(r"Index $j$")
    ax.set_ylabel(r"Singular values $\sigma_j(\omega)$")
    ax.set_title(r"SVD of $R(\omega)$")
    ax.set_yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "singular_values_compare.png")

    plt.figure()
    plt.plot(x, np.abs(arrays[0]), label="rsvd")
    plt.plot(x, np.abs(arrays[1]), "--", label="rsvd-dt")
    ax = plt.gca()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Abs. value of output mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "output_mode_compare.png")

    plt.figure()
    plt.plot(x, np.abs(arrays[2]), label="rsvd")
    plt.plot(x, np.abs(arrays[3]), "--", label="rsvd-dt")
    ax = plt.gca()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Abs. value of input mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "input_mode_compare.png")
