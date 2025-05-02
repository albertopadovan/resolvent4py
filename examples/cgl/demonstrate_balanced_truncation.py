from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
from mpi4py import MPI

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)


def L_generator(omega, A):
    comm = MPI.COMM_WORLD
    Rinv = res4py.create_AIJ_identity(comm, A.getSizes())
    Rinv.scale(1j * omega)
    Rinv.axpy(-1.0, A)
    ksp = res4py.create_mumps_solver(comm, Rinv)
    L = res4py.linear_operators.MatrixLinearOperator(comm, Rinv, ksp)
    return (L, L.solve_mat, (L.destroy,))


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
B = res4py.read_bv(comm, load_path + "B.dat", (A.getSizes()[0], 2))
C = res4py.read_bv(comm, load_path + "C.dat", (A.getSizes()[0], 3))

domega = 0.648 / 2
omegas = np.arange(-30, 30, domega)
weights = domega / (2 * np.pi) * np.ones(len(omegas))
L_gen = partial(L_generator, A=A)
L_generators = [L_gen for _ in range(len(omegas))]

res4py.petscprint(comm, "Computing Gramian factors...")
X, Y = res4py.model_reduction.compute_gramian_factors(
    L_generators, omegas, weights, B, C
)

res4py.petscprint(comm, "Computing balanced projection...")
Phi, Psi, S = res4py.model_reduction.compute_balanced_projection(X, Y, 10)
# res4py.petscprint(comm, np.diag(S))

# Id = Phi.dot(Psi)
# Id_ = Id.getDenseArray()
# res4py.petscprint(comm, np.diag(Id_))
