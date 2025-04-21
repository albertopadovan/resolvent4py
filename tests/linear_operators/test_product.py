import scipy as sp
import numpy as np
import sys
import os

sys.path.append("../../")

import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

mats_names = ["A1", "A2", "A3", "A4"]
row_sizes = [10, 30, 30, 20]
col_sizes = row_sizes[1:]
col_sizes.append(7)

s = 9

path = "data/"

fnames = None
if rank == 0:
    os.makedirs(path) if os.path.isdir(path) == False else None

    matrices = []
    for i in range(len(row_sizes)):
        Nr = row_sizes[i]
        Nc = col_sizes[i]
        A = sp.sparse.random(Nr, Nc, 0.1, "csr", np.complex128)
        if Nr == Nc:
            A += sp.sparse.identity(Nr, np.complex128, "csr")
        A = A.tocoo()
        arrays = [A.row, A.col, A.data]
        name = path + mats_names[i] + "_"
        fnames_jac = [name + "rows.dat", name + "cols.dat", name + "vals.dat"]
        for i, array in enumerate(arrays):
            vec = PETSc.Vec().createWithArray(
                array, len(array), None, MPI.COMM_SELF
            )
            res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
            vec.destroy()
        A = A.todense()
        matrices.append(A.copy())

    A = matrices[0]
    for mat in matrices[1:]:
        A = A @ mat

    x = np.random.randn(A.shape[-1]) + 1j * np.random.randn(A.shape[-1])
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)

    X = np.random.randn(A.shape[-1], s) + 1j * np.random.randn(A.shape[-1], s)
    Xmat = PETSc.Mat().createDense((A.shape[-1], s), None, X, MPI.COMM_SELF)
    AX = PETSc.Mat().createDense((A.shape[0], s), None, A @ X, MPI.COMM_SELF)

    objs = [xvec, Ax, Xmat, AX]
    fnames_ = ["xvec", "Axvec", "X", "AX"]
    fnames = [path + root + ".dat" for root in fnames_]
    for k, obj in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

comm.Barrier()

fnames = comm.bcast(fnames, root=0)
fnames_vecs = fnames[:2]
fnames_mats = fnames[2:]

lops = []
actions = []
for i in range(len(mats_names)):
    name = path + mats_names[i] + "_"
    fnames = [name + "rows.dat", name + "cols.dat", name + "vals.dat"]
    Nr = row_sizes[i]
    Nc = col_sizes[i]
    Nrl = res4py.compute_local_size(Nr)
    Ncl = res4py.compute_local_size(Nc)
    A = res4py.read_coo_matrix(comm, fnames, ((Nrl, Nr), (Ncl, Nc)))
    Alop = res4py.MatrixLinearOperator(comm, A)
    lops.append(Alop)
    actions.append(Alop.apply)


linop = res4py.ProductLinearOperator(comm, lops, actions)

x = res4py.read_vector(comm, fnames_vecs[0])
actions = [linop.apply]
strs = ["apply"]
for i in range(1, len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i - 1](x))
    string = f"Error for {strs[i - 1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()

nX = col_sizes[-1]
nXl = res4py.compute_local_size(nX)
X = res4py.read_bv(comm, fnames_mats[0], ((nXl, nX), s))
actions = [linop.apply_mat]
strs = ["apply_mat"]
for i in range(1, len(fnames_vecs)):
    nX = row_sizes[0]
    nXl = res4py.compute_local_size(nX)
    Y = res4py.read_bv(comm, fnames_mats[i], ((nXl, nX), s))
    res4py.bv_add(-1.0, Y, actions[i - 1](X))
    string = f"Error for {strs[i - 1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()
