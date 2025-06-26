r"""
Balanced Truncation Demonstration
=================================

Given the linear input-output dynamics

.. math::

    \begin{aligned}
        \frac{d}{dt}q &= A q + B u \\
        y &= C q
    \end{aligned}

with the input and output matrices :math:`B` and :math:`C` defined as in 
:cite:`Ilak2010model`, we perform balanced truncation in the frequency domain
using the algorithm presented in :cite:`dergham2011`.

- LU decomposition using :func:`~resolvent4py.utils.ksp.create_mumps_solver`
- Balanced truncation in the frequency domain using 
  :func:`~resolvent4py.model_reduction.balanced_truncation`

"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)


def L_generator(omega, A):
    comm = PETSc.COMM_WORLD
    Rinv = res4py.create_AIJ_identity(comm, A.getSizes())
    Rinv.scale(1j * omega)
    Rinv.axpy(-1.0, A)
    ksp = res4py.create_mumps_solver(Rinv)
    L = res4py.linear_operators.MatrixLinearOperator(Rinv, ksp)
    return (L, L.solve_mat, (L.destroy,))


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
B = res4py.read_bv(load_path + "B.dat", (A.getSizes()[0], 2))
C = res4py.read_bv(load_path + "C.dat", (A.getSizes()[0], 3))

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
