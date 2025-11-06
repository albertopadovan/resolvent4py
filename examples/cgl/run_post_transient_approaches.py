import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

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
B = res4py.read_bv(load_path + "B.dat", (sizes[0], 2))
b = B.getColumn(0)
f = b.copy()
B.restoreColumn(0, b)
Aop = res4py.linear_operators.MatrixLinearOperator(A)


omega = 0.648
n_omegas = 3
dt = 1.5e-4
tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
    dt, omega, n_omegas, False
)
n_omegas = len(omegas)

FHat = SLEPc.BV().create(comm=comm)
FHat.setSizes(sizes[0], n_omegas)
FHat.setType("mat")
FHat.scale(0.0)

alphas = np.random.randn(n_omegas)
for j in range(FHat.getSizes()[-1]):
    if j != int((n_omegas - 1) // 2):
        fj = FHat.getColumn(j)
        f.copy(fj)
        fj.scale(np.exp(1j * alphas[j]))
        FHat.restoreColumn(j, fj)

tf = 2 * np.pi / omega
v = f.copy()
v.scale(0)
sol = res4py.solve_ivp(
    v, Aop.apply, 0.0, tf, int(tf // dt), periodic_forcing=(FHat, omegas)
)


# sol_a = sol.getArray()
# plt.figure()
# plt.plot(sol_a.real)
# plt.plot(sol_a.imag)
# plt.show()

Expop = res4py.linear_operators.MatrixExponentialLinearOperator(Aop, tf, dt)
linop = res4py.linear_operators.ShiftAndScaleLinearOperator(Expop, 1.0, -1.0)
L = res4py.linear_operators.PetscPythonLinearOperator.create_shell(linop)


# sz = L.getSizes()[0]
# b = res4py.generate_random_petsc_vector(sz, True)
# x = b.duplicate()

ksp = PETSc.KSP().create()
ksp.setOperators(L)
ksp.setType("gmres")
ksp.getPC().setType("none")
ksp.setTolerances(
    rtol=1e-8,  # relative tolerance
    atol=1e-12,  # absolute tolerance
    divtol=1e4,  # divergence tolerance
    max_it=1000,  # maximum number of iterations
)
ksp.setFromOptions()


def monitor(ksp, its, rnorm):
    res4py.petscprint(comm, f"GMRES iter {its}: residual = {rnorm:.3e}")


ksp.setMonitor(monitor)

x = sol.duplicate()
res4py.petscprint(comm, "Solving linear system...")
ksp.solve(sol, x)
res4py.petscprint(comm, "Solved.")


y = res4py.solve_ivp(
    x, Aop.apply, 0.0, tf, int(tf // dt), periodic_forcing=(FHat, omegas)
)
x.axpy(-1.0, y)
error = x.norm()
res4py.petscprint(comm, "Error = %1.10e" % error)


x.scale(0)
YHat = FHat.duplicate()
X = SLEPc.BV().create(comm)
X.setSizes(sizes[0], len(tsim[::nsave]))
X.setType("mat")

Id = res4py.create_AIJ_identity(comm, sizes)
Idop = res4py.linear_operators.MatrixLinearOperator(Id)
res4py.compute_post_transient_solution(
    Aop,
    Idop,
    Idop,
    Aop.apply,
    tsim,
    nsave,
    200,
    omegas,
    x,
    FHat,
    YHat,
    X,
    1e-5,
    verbose=2,
)


# sol_a = x.getArray()
# plt.figure()
# plt.plot(sol_a.real)
# plt.plot(sol_a.imag)
# plt.show()

# x_ = Aop.solve(b)

# x_.axpy(-1.0, x)
# error = x_.norm()
# res4py.petscprint(comm, "Error = %1.10e"%error)
