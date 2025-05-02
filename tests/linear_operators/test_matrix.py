import scipy as sp
import numpy as np
import os

import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

string = f"Hello, running test_matrix.py with {size} processors"
res4py.petscprint(comm, string)

N = 100
s = 10

path = "data/"

fnames_jac, fnames = None, None
if rank == 0:
    os.makedirs(path) if os.path.isdir(path) == False else None
    A = sp.sparse.random(N, N, 0.1, "csr", np.complex128)
    A += sp.sparse.identity(N, np.complex128, "csr")
    A = A.tocoo()
    arrays = [A.row, A.col, A.data]
    fnames_jac = [path + "rows.dat", path + "cols.dat", path + "vals.dat"]
    for i, array in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A = A.todense()
    Ainv = sp.linalg.inv(A)

    x = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)
    ATx = PETSc.Vec().createWithArray(A.conj().T @ x, comm=MPI.COMM_SELF)
    Ainvx = PETSc.Vec().createWithArray(Ainv @ x, comm=MPI.COMM_SELF)
    AinvTx = PETSc.Vec().createWithArray(Ainv.conj().T @ x, comm=MPI.COMM_SELF)

    X = np.random.randn(N, s) + 1j * np.random.randn(N, s)
    Xmat = PETSc.Mat().createDense((N, s), None, X, MPI.COMM_SELF)
    AX = PETSc.Mat().createDense((N, s), None, A @ X, MPI.COMM_SELF)
    ATX = PETSc.Mat().createDense((N, s), None, A.conj().T @ X, MPI.COMM_SELF)
    AinvX = PETSc.Mat().createDense((N, s), None, Ainv @ X, MPI.COMM_SELF)
    AinvTX = PETSc.Mat().createDense(
        (N, s), None, Ainv.conj().T @ X, MPI.COMM_SELF
    )

    objs = [xvec, Ax, ATx, Ainvx, AinvTx, Xmat, AX, ATX, AinvX, AinvTX]
    fnames_ = [
        "xvec",
        "Axvec",
        "ATxvec",
        "Ainvxvec",
        "AinvTxvec",
        "X",
        "AX",
        "ATX",
        "AinvX",
        "AinvTX",
    ]
    fnames = [path + root + ".dat" for root in fnames_]
    for i, obj in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[i], obj)

comm.Barrier()

fnames_jac = comm.bcast(fnames_jac, root=0)
fnames = comm.bcast(fnames, root=0)
fnames_vecs = fnames[:5]
fnames_mats = fnames[5:]

Nl = res4py.compute_local_size(N)
sl = res4py.compute_local_size(s)

A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N), (Nl, N)))
ksp = res4py.create_mumps_solver(comm, A)
linop = res4py.linear_operators.MatrixLinearOperator(comm, A, ksp)
x = res4py.read_vector(comm, fnames_vecs[0])
actions = [
    linop.apply,
    linop.apply_hermitian_transpose,
    linop.solve,
    linop.solve_hermitian_transpose,
]
strs = [
    "apply",
    "apply_hermitian_transpose",
    "solve",
    "solve_hermitian_transpose",
]
for i in range(1, len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i - 1](x))
    string = f"Error for {strs[i - 1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()
X = res4py.read_bv(comm, fnames_mats[0], ((Nl, N), s))
actions = [
    linop.apply_mat,
    linop.apply_hermitian_transpose_mat,
    linop.solve_mat,
    linop.solve_hermitian_transpose_mat,
]
strs = [
    "apply_mat",
    "apply_hermitian_transpose_mat",
    "solve_mat",
    "solve_hermitian_transpose_mat",
]
for i in range(1, len(fnames_vecs)):
    Y = res4py.read_bv(comm, fnames_mats[i], ((Nl, N), s))
    res4py.bv_add(-1.0, Y, actions[i - 1](X))
    string = f"Error for {strs[i - 1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()
