import os

import numpy as np
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

N = 100
r = 10
q = 7
s = 17

path = "data/"

fnames_factors, fnames = None, None
if rank == 0:
    os.makedirs(path) if os.path.isdir(path) == False else None
    U = np.random.randn(N, r) + 1j * np.random.randn(N, r)
    Sigma = np.random.randn(r, q) + 1j * np.random.randn(r, q)
    V = np.random.randn(N, q) + 1j * np.random.randn(N, q)
    A = U @ Sigma @ V.conj().T

    np.save(path + "Sigma.npy", Sigma)
    Umat = PETSc.Mat().createDense((N, r), None, U, MPI.COMM_SELF)
    Vmat = PETSc.Mat().createDense((N, q), None, V, MPI.COMM_SELF)
    objs = [Umat, Vmat]
    fnames_ = ["U", "V"]
    fnames_factors = [path + root + ".dat" for root in fnames_]
    for k, obj in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames_factors[k], obj)

    x = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)
    ATx = PETSc.Vec().createWithArray(A.conj().T @ x, comm=MPI.COMM_SELF)

    X = np.random.randn(N, s) + 1j * np.random.randn(N, s)
    Xmat = PETSc.Mat().createDense((N, s), None, X, MPI.COMM_SELF)
    AX = PETSc.Mat().createDense((N, s), None, A @ X, MPI.COMM_SELF)
    ATX = PETSc.Mat().createDense((N, s), None, A.conj().T @ X, MPI.COMM_SELF)

    objs = [xvec, Ax, ATx, Xmat, AX, ATX]
    fnames_ = ["xvec", "Axvec", "ATxvec", "X", "AX", "ATX"]
    fnames = [path + root + ".dat" for root in fnames_]
    for k, obj in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

comm.Barrier()

fnames = comm.bcast(fnames, root=0)
fnames_factors = comm.bcast(fnames_factors, root=0)
fnames_vecs = fnames[:3]
fnames_mats = fnames[3:]

Nl = res4py.compute_local_size(N)
rl = res4py.compute_local_size(r)
ql = res4py.compute_local_size(q)
sl = res4py.compute_local_size(s)
Sig = np.load(path + "Sigma.npy")
U = res4py.read_bv(comm, fnames_factors[0], ((Nl, N), r))
V = res4py.read_bv(comm, fnames_factors[1], ((Nl, N), q))
linop = res4py.linear_operators.LowRankLinearOperator(comm, U, Sig, V)
x = res4py.read_vector(comm, fnames_vecs[0])
actions = [linop.apply, linop.apply_hermitian_transpose]
strs = ["apply", "apply_hermitian_transpose"]
for i in range(1, len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i - 1](x))
    string = f"Error for {strs[i - 1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()
X = res4py.read_bv(comm, fnames_mats[0], ((Nl, N), s))
actions = [linop.apply_mat, linop.apply_hermitian_transpose_mat]
strs = ["apply_mat", "apply_hermitian_transpose_mat"]
for i in range(1, len(fnames_vecs)):
    Y = res4py.read_bv(comm, fnames_mats[i], ((Nl, N), s))
    res4py.bv_add(-1.0, Y, actions[i - 1](X))
    string = f"Error for {strs[i - 1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()
