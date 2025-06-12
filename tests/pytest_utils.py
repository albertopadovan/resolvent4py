from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py
import numpy as np


def generate_random_matrix(comm, size, complex=True):
    r"""Create random matrix of size = (Nrows, Ncols)"""
    Nr, Nc = size
    Nrl = res4py.compute_local_size(Nr)
    Ncl = res4py.compute_local_size(Nc)
    Apetsc = res4py.generate_random_petsc_sparse_matrix(
        comm, ((Nrl, Nr), (Ncl, Nc)), int(0.3 * Nr * Nc), complex
    )
    Adense = Apetsc.copy()
    Adense.convert(PETSc.Mat.Type.DENSE)
    Adense_seq = res4py.distributed_to_sequential_matrix(comm, Adense)
    Apython = Adense_seq.getDenseArray().copy()
    Adense.destroy()
    Adense_seq.destroy()
    return Apetsc, Apython


def generate_negative_semidefinite_matrix(comm, size, complex=True):
    r"""Create negative semidefinite matrix A of shape (N, N), i.e. all eigenvalues ≤ 0."""
    N, _ = size
    Nl = res4py.compute_local_size(N)
    Bpetsc = res4py.generate_random_petsc_sparse_matrix(
        comm, ((Nl, N), (Nl, N)),
        int(0.3 * N * N),
        complex
    )
    Bpetsc.convert(PETSc.Mat.Type.DENSE)
    Bseq = res4py.distributed_to_sequential_matrix(comm, Bpetsc)
    Bpython = Bseq.getDenseArray().copy()
    eigvals = np.sort(np.linalg.eigvals(Bpython).real)

    if eigvals[-1] >= 0:
        I = res4py.create_AIJ_identity(comm, ((Nl, N), (Nl, N)))
        I.scale(eigvals[-1])
        I.convert(PETSc.Mat.Type.DENSE)
        Bpetsc.axpy(-1.1, I)
        I.destroy()

    Btemp = Bpetsc.copy()
    Bseq = res4py.distributed_to_sequential_matrix(comm, Btemp)
    Bpython = Bseq.getDenseArray().copy()

    Bpetsc.convert(PETSc.Mat.Type.AIJ)

    Bseq.destroy()
    Btemp.destroy()
    return Bpetsc, Bpython



def generate_random_bv(comm, size, complex=True):
    r"""Create random SLEPc BV of size = (Nrows, Ncols)"""
    Nr, Nc = size
    Nrl = res4py.compute_local_size(Nr)
    X = SLEPc.BV().create(comm=comm)
    X.setSizes((Nrl, Nr), Nc)
    X.setType("mat")
    X.setRandomNormal()
    X = res4py.bv_real(X, True) if not complex else X
    Xm = X.getMat()
    Xmseq = res4py.distributed_to_sequential_matrix(comm, Xm)
    X.restoreMat(Xm)
    Xpython = Xmseq.getDenseArray().copy()
    Xmseq.destroy()
    return X, Xpython


def generate_random_vector(comm, N, complex=True):
    r"""Create random vector of size N"""
    Nl = res4py.compute_local_size(N)
    x = res4py.generate_random_petsc_vector(comm, (Nl, N), complex)
    xseq = res4py.distributed_to_sequential_vector(comm, x)
    xpython = xseq.getArray().copy()
    xseq.destroy()
    return x, xpython


def compute_error_vector(comm, linop_action, x, y, python_action, xpython):
    r"""Compute error between the action of a resolvent4py LinearOperator
    and the analogue in scipy on vectors"""
    y = linop_action(x, y)
    ys = res4py.distributed_to_sequential_vector(comm, y)
    ysa = ys.getArray().copy()
    ys.destroy()
    ypython = python_action(xpython)
    return np.linalg.norm(ypython - ysa) / np.linalg.norm(ypython)


def compute_error_bv(comm, linop_action, X, Y, python_action, Xpython):
    r"""Compute error between the action of a resolvent4py LinearOperator
    and the analogue in scipy on BVs"""
    Y = linop_action(X, Y)
    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(comm, Ym)
    Y.restoreMat(Ym)
    Ymsa = Yms.getDenseArray().copy()
    Yms.destroy()
    Ypython = python_action(Xpython)
    return np.linalg.norm(Ypython - Ymsa) / np.linalg.norm(Ypython)
