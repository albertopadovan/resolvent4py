r"""
Eigendecomposition Demonstration
================================

Given the linear dynamics :math:`d_t q = Aq`, we compute the eigenvalues of
the matrix :math:`A` closest to the origin using the shift-and-invert technique.
This script demonstrates the following:

- LU decomposition using :func:`~resolvent4py.utils.ksp.create_mumps_solver`
- Eigendecomposition using
  :func:`~resolvent4py.linalg.eigendecomposition.eig`

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
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

# Compute the eigendecomposition of L using shift and invert about s.
# We need to define a matrix M = sI - A, compute its lu decomposition,
# define a corresponding MatrixLinearOperator L and compute its eigendecomp.
res4py.petscprint(comm, "Computing LU decomposition...")
s = 0.0
M = res4py.create_AIJ_identity(comm, sizes)
M.scale(s)
M.axpy(-1.0, A)
ksp = res4py.create_mumps_solver(M)
res4py.check_lu_factorization(M, ksp)
L = res4py.linear_operators.MatrixLinearOperator(M, ksp)

# Compute the eigendecomp.
res4py.petscprint(comm, "Running Arnoldi iteration...")
krylov_dim = 50
n_evals = 10
D, V = res4py.linalg.eig(
    L, L.solve, krylov_dim, n_evals, lambda x: s - 1.0 / x
)

# Check convergence
L.destroy()
L = res4py.linear_operators.MatrixLinearOperator(A)
res4py.linalg.check_eig_convergence(L.apply, D, V)

# Destroy objects
L.destroy()
V.destroy()

# Make some plots
if comm.getRank() == 0:
    l = 30 * 2
    x = np.linspace(-l / 2, l / 2, num=N, endpoint=True)
    nu = 1.0 * (2 + 0.4 * 1j)
    gamma = 1 - 1j
    mu0 = 0.38
    mu2 = -0.01
    sigma = 0.4
    system = cgl.CGL(x, nu, gamma, mu0, mu2, sigma)

    save_path = "results/"
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    evals = system.compute_exact_eigenvalues(n_evals)
    D = np.diag(D)

    plt.figure()
    plt.plot(D.imag, D.real, "ko", label="res4py")
    plt.plot(evals.imag, evals.real, "rx", label="exact")
    ax = plt.gca()
    ax.set_xlabel(r"$\mathrm{Real}(\lambda_j)$")
    ax.set_ylabel(r"$\mathrm{Imag}(\lambda_j)$")
    ax.set_title(r"Eigenvalues $\lambda$")
    ax.axhline(y=0.0, linewidth=1.0, color="blue", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "eigenvalues.png")
