import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)


comm = MPI.COMM_WORLD

save_path = "data/"
bflow_freqs = np.load(save_path + 'bflow_freqs.npy')
perts_freqs = np.load(save_path + 'perts_freqs.npy')
fnames_lst = [
    (
        save_path + "rows_%02d.dat" % j,
        save_path + "cols_%02d.dat" % j,
        save_path + "vals_%02d.dat" % j
    )
    for j in range(len(bflow_freqs))
]

N = 3 * len(perts_freqs)
Nl = res4py.compute_local_size(N)
n = 3
nl = res4py.compute_local_size(n)
H = res4py.read_harmonic_balanced_matrix(
    comm,
    fnames_lst,
    True,
    ((nl, n), (nl, n)),
    ((Nl, N), (Nl, N)),
)

x = PETSc.Vec().createWithArray(np.ones(Nl), (Nl, N), None, comm)
y = x.duplicate()
H.mult(x, y)


def gmres_monitor(ksp, its, rnorm):
    print(f"GMRES Iteration {its:3d}, Residual Norm = {rnorm:.3e}")


# opts = PETSc.Options()
# print(opts.getAll())
# opts["pc_type"] = "bjacobi"
# opts["pc_bjacobi_blocks"] = len(perts_freqs)
# opts["sub_ksp_type"] = "preonly"
# opts["sub_pc_type"] = "lu"
# opts["sub_pc_factor_mat_solver_type"] = "mumps"

# ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
# ksp.setOperators(H)
# ksp.setType('gmres')
# ksp.setTolerances(rtol=1e-10, atol=1e-10)
# ksp.setMonitor(gmres_monitor)
# pc = ksp.getPC()
# pc.setFromOptions()
# pc.setUp()
# ksp.setUp()

# pc = ksp.getPC()

# print(pc.getType())
# print(PETSc.Sys.getVersion())
    
ksp = res4py.create_gmres_bjacobi_solver(comm, H, len(perts_freqs))
res4py.check_gmres_bjacobi_solver(comm, H, ksp)

x = res4py.generate_random_petsc_vector(comm, (Nl, N), True)
y = x.duplicate()
ksp.solve(x, y)

x2 = x.duplicate()
H.mult(y, x2)
x2.axpy(-1.0, x)
res4py.petscprint(comm, x2.norm())




