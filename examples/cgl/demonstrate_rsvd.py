import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from mpi4py import MPI

import cgl

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)

comm = MPI.COMM_WORLD

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
A = res4py.read_coo_matrix(comm, names, sizes)

# Compute the SVD of the resolvent operator R = inv(1j*omega*I - A) using
# the randomized SVD algorithm
res4py.petscprint(comm, "Computing LU decomposition...")
s = -1j * 0.648
Rinv = res4py.create_AIJ_identity(comm, sizes)
Rinv.scale(s)
Rinv.axpy(-1.0, A)
ksp = res4py.create_mumps_solver(comm, Rinv)
res4py.check_lu_factorization(comm, Rinv, ksp)
L = res4py.linear_operators.MatrixLinearOperator(comm, Rinv, ksp)

# Compute the svd
res4py.petscprint(comm, "Running randomized SVD...")
n_rand = 40
n_loops = 3
n_svals = 10
U, S, V = res4py.linalg.randomized_svd(
    L, L.solve_mat, n_rand, n_loops, n_svals
)

# Check convergence
res4py.linalg.check_randomized_svd_convergence(L.solve, U, S, V)

# Destroy objects
L.destroy()
V.destroy()
U.destroy()

if comm.Get_rank() == 0:
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

    Id = sp.sparse.identity(N)
    R = sp.linalg.inv((s * Id - system.A).todense())
    _, s, _ = sp.linalg.svd(R)
    S = np.diag(S)

    plt.figure()
    plt.plot(S.real, "ko", label="res4py")
    plt.plot(s[: len(S)].real, "rx", label="exact")
    ax = plt.gca()
    ax.set_xlabel(r"Index $j$")
    ax.set_ylabel(r"Singular values $\sigma_j(\omega)$")
    ax.set_title(r"SVD of $R(\omega)$")
    ax.set_yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "singular_values.png")
