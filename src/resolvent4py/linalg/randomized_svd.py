__all__ = [
    "randomized_svd",
    "check_randomized_svd_convergence",
]

import typing

import numpy as np
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from ..linear_operators import LinearOperator
from ..utils.matrix import create_dense_matrix
from ..utils.miscellaneous import petscprint
from ..utils.vector import enforce_complex_conjugacy


def randomized_svd(
    L: LinearOperator,
    action: typing.Callable[[SLEPc.BV, SLEPc.BV], SLEPc.BV],
    n_rand: int,
    n_loops: int,
    n_svals: int,
) -> typing.Tuple[SLEPc.BV, np.ndarray, SLEPc.BV]:
    r"""
    Compute the singular value decomposition (SVD) of the linear operator 
    specified by :code:`L` and :code:`action` using a randomized SVD algorithm.
    For example, with :code:`L.solve_mat` we compute 

    .. math::

        L^{-1} = U \Sigma V^*.
    
    :param L: instance of the :class:`.LinearOperator` class
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply_mat` or
        :meth:`.LinearOperator.solve_mat`
    :type action: Callable[[SLEPc.BV, SLEPc.BV], SLEPc.BV]
    :param n_rand: number of random vectors
    :type n_rand: int
    :param n_loops: number of randomized svd iterations
    :type n_loops: int
    :param n_svals: number of singular triplets to return
    :type n_svals: int

    :return: a tuple :math:`(U,\,\Sigma,\, V)` with the leading 
        :code:`n_svals` singular values and corresponding left and \
        right singular vectors
    :rtype: (SLEPc.BV with :code:`n_svals` columns, 
        numpy.ndarray of size :code:`n_svals x n_svals`, 
        SLEPc.BV with :code:`n_svals` columns)
    """
    if action != L.apply_mat and action != L.solve_mat:
        raise ValueError(f"action must be L.apply_mat or L.solve_mat.")
    action_adj = (
        L.apply_hermitian_transpose_mat
        if action == L.apply_mat
        else L.solve_hermitian_transpose_mat
    )
    # Assemble random BV (this will be multiplied against L^*)
    rowsizes = L._dimensions[0]
    X = SLEPc.BV().create(comm=L._comm)
    X.setSizes(rowsizes, n_rand)
    X.setType("mat")
    X.setRandomNormal()
    for j in range(n_rand):
        xj = X.getColumn(j)
        if L._real:
            row_offset = xj.getOwnershipRange()[0]
            rows = np.arange(rowsizes[0], dtype=np.int64) + row_offset
            xj.setValues(rows, xj.getArray().real)
            xj.assemble()
        if L._block_cc:
            enforce_complex_conjugacy(L._comm, xj, L._nblocks)
        X.restoreColumn(j, xj)
    X.orthogonalize(None)
    # Perform randomized SVD loop
    Qadj = SLEPc.BV().create(comm=L._comm)
    Qadj.setSizes(L._dimensions[-1], n_rand)
    Qadj.setType("mat")
    Qadj = action_adj(X, Qadj)
    Qadj.orthogonalize(None)
    X.destroy()
    Qfwd = SLEPc.BV().create(comm=L._comm)
    Qfwd.setSizes(L._dimensions[0], n_rand)
    Qfwd.setType("mat")
    R = create_dense_matrix(MPI.COMM_SELF, (n_rand, n_rand))
    for j in range(n_loops):
        Qfwd = action(Qadj, Qfwd)
        Qfwd.orthogonalize(None)
        Qadj = action_adj(Qfwd, Qadj)
        Qadj.orthogonalize(R)
    # Compute low-rank SVD
    u, s, v = sp.linalg.svd(R.getDenseArray())
    R.destroy()
    v = v.conj().T
    s = s[:n_svals]
    u = u[:, :n_svals]
    v = v[:, :n_svals]
    u = PETSc.Mat().createDense((n_rand, n_svals), None, u, comm=MPI.COMM_SELF)
    v = PETSc.Mat().createDense((n_rand, n_svals), None, v, comm=MPI.COMM_SELF)
    Qfwd.multInPlace(v, 0, n_svals)
    Qfwd.setActiveColumns(0, n_svals)
    Qfwd.resize(n_svals, copy=True)
    Qadj.multInPlace(u, 0, n_svals)
    Qadj.setActiveColumns(0, n_svals)
    Qadj.resize(n_svals, copy=True)
    u.destroy()
    v.destroy()
    return (Qfwd, np.diag(s), Qadj)


def check_randomized_svd_convergence(
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    U: SLEPc.BV,
    S: np.ndarray,
    V: SLEPc.BV,
) -> None:
    r"""
    Check the convergence of the singular value triplets by measuring
    :math:`\lVert Av/\sigma - u\rVert` for every triplet :math:`(u, \sigma, v)`.

    :param action: one of :meth:`.LinearOperator.apply` or
        :meth:`.LinearOperator.solve`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param U: left singular vectors
    :type U: SLEPc.BV
    :param D: diagonal 2D numpy array with the singular values
    :type D: numpy.ndarray
    :param V: right singular vectors
    :type V: SLEPc.BV

    :return: None
    """
    petscprint(MPI.COMM_WORLD, " ")
    petscprint(MPI.COMM_WORLD, "Executing SVD triplet convergence check...")
    x = U.createVec()
    n_svals = S.shape[-1]
    for k in range(n_svals):
        v = V.getColumn(k)
        u = U.getColumn(k)
        x = action(v, x)
        x.scale(1.0 / S[k, k])
        x.axpy(-1.0, u)
        error = x.norm()
        str = "Error for SVD triplet %d = %1.15e" % (k + 1, error)
        petscprint(MPI.COMM_WORLD, str)
        U.restoreColumn(k, u)
        V.restoreColumn(k, v)
    x.destroy()
    petscprint(MPI.COMM_WORLD, "Executing SVD triplet convergence check...")
    petscprint(MPI.COMM_WORLD, " ")
