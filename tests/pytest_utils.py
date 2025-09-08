from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py
import numpy as np
import scipy as sp


def generate_random_matrix(comm, size, complex=True):
    r"""Create random matrix of size = (Nrows, Ncols)"""
    Nr, Nc = size
    Nrl = res4py.compute_local_size(Nr)
    Ncl = res4py.compute_local_size(Nc)
    Apetsc = res4py.generate_random_petsc_sparse_matrix(
        ((Nrl, Nr), (Ncl, Nc)), int(0.3 * Nr * Nc), complex
    )
    Adense = Apetsc.copy()
    Adense.convert(PETSc.Mat.Type.DENSE)
    Adense_seq = res4py.distributed_to_sequential_matrix(Adense)
    Apython = Adense_seq.getDenseArray().copy()
    Adense.destroy()
    Adense_seq.destroy()
    return Apetsc, Apython


def generate_stable_random_matrix(comm, size, complex=True):
    r"""Create random matrix with eigenvalues with negative real part"""
    Apetsc, Apython = generate_random_matrix(comm, size, complex)
    evals, _ = sp.linalg.eig(Apython)
    shift = np.sort(evals.real)[-1]
    if shift >= 0.0:
        alpha = 1.1
        Id = res4py.create_AIJ_identity(comm, Apetsc.getSizes())
        Id.convert(PETSc.Mat.Type.AIJ)
        Apetsc.axpy(-alpha * shift, Id)
        Id.destroy()
        Apython -= alpha * shift * np.eye(Apython.shape[0])
    return Apetsc, Apython


def generate_random_bv(comm, size, complex=True):
    r"""Create random SLEPc BV of size = (Nrows, Ncols)"""
    Nr, Nc = size
    Nrl = res4py.compute_local_size(Nr)
    X = SLEPc.BV().create(comm=comm)
    X.setSizes((Nrl, Nr), Nc)
    X.setType("mat")
    rand = PETSc.Random().create(comm=comm)
    rand.setType(PETSc.Random.Type.RAND)
    rand.setSeed(round(np.random.randint(1000, 100000) + comm.getRank()))
    X.setRandomContext(rand)
    X.setRandomNormal()
    rand.destroy()
    X = res4py.bv_real(X, True) if not complex else X
    Xm = X.getMat()
    Xmseq = res4py.distributed_to_sequential_matrix(Xm)
    X.restoreMat(Xm)
    Xpython = Xmseq.getDenseArray().copy()
    Xmseq.destroy()
    return X, Xpython


def generate_random_vector(comm, N, complex=True):
    r"""Create random vector of size N"""
    Nl = res4py.compute_local_size(N)
    x = res4py.generate_random_petsc_vector((Nl, N), complex)
    xseq = res4py.distributed_to_sequential_vector(x)
    xpython = xseq.getArray().copy()
    xseq.destroy()
    return x, xpython


def compute_error_vector(comm, linop_action, x, y, python_action, xpython):
    r"""Compute error between the action of a resolvent4py LinearOperator
    and the analogue in scipy on vectors"""
    y = linop_action(x, y)
    ys = res4py.distributed_to_sequential_vector(y)
    ysa = ys.getArray().copy()
    ys.destroy()
    ypython = python_action(xpython)
    return np.linalg.norm(ypython - ysa) / np.linalg.norm(ypython)


def compute_error_vector_shell_operator(
    comm, linop_action, x, y, python_action, xpython
):
    r"""Compute error between the action of a PetscPython (shell) operator
    and the analogue in scipy on vectors"""
    linop_action(x, y)
    ys = res4py.distributed_to_sequential_vector(y)
    ysa = ys.getArray().copy()
    ys.destroy()
    ypython = python_action(xpython)
    return np.linalg.norm(ypython - ysa) / np.linalg.norm(ypython)


def compute_error_bv(comm, linop_action, X, Y, python_action, Xpython):
    r"""Compute error between the action of a resolvent4py LinearOperator
    and the analogue in scipy on BVs"""
    Y = linop_action(X, Y)
    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    Ymsa = Yms.getDenseArray().copy()
    Yms.destroy()
    Ypython = python_action(Xpython)
    return np.linalg.norm(Ypython - Ymsa) / np.linalg.norm(Ypython)
