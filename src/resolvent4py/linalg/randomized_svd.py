__all__ = [
    "randomized_svd",
    "check_randomized_svd_convergence",
]

import typing

import numpy as np
import scipy as sp
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
    verbose: typing.Optional[int] = 0,
) -> typing.Tuple[SLEPc.BV, np.ndarray, SLEPc.BV]:
    r"""
    Compute the singular value decomposition (SVD) of the linear operator
    specified by :code:`L` and :code:`action` using a randomized SVD algorithm.
    (See [Halko2011]_.)
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
    :param n_loops: number of randomized svd power iterations
        (see [Ribeiro2020]_ for additional details on this parameter)
    :type n_loops: int
    :param n_svals: number of singular triplets to return
    :type n_svals: int
    :param verbose: defines verbosity of output to terminal (useful to
        monitor progress during time stepping). = 0 no printout to terminal,
        = 1 monitor randomized SVD iterations.
    :type verbose: Optional[int], default is 0

    :return: leading :code:`n_svals` left singular vectors,
        singular values and right singular vectors
    :rtype: Tuple[SLEPc.BV, np.ndarray, SLEPc.BV]

    References
    ----------
    .. [Halko2011] Halko et al., *Finding Structure With Randomness:
        Probabilistic Algorithms For Constructing Approximate Matrix
        Decompositions*, SIAM Review, 2011
    .. [Ribeiro2020] Ribeiro et al., *Randomized resolvent analysis*,
        Physical Review Fluids, 2020
    """
    comm = L.get_comm()
    if action != L.apply_mat and action != L.solve_mat:
        raise ValueError(f"action must be L.apply_mat or L.solve_mat.")
    action_adj = (
        L.apply_hermitian_transpose_mat
        if action == L.apply_mat
        else L.solve_hermitian_transpose_mat
    )
    # Assemble random BV (this will be multiplied against L^*)
    rowsizes = L.get_dimensions()[0]
    # Set seed
    rank = L.get_comm().getRank()
    rand = PETSc.Random().create(comm=comm)
    rand.setType(PETSc.Random.Type.RAND)
    rand.setSeed(round(np.random.randint(1000, 100000) + rank))
    X = SLEPc.BV().create(comm=comm)
    X.setSizes(rowsizes, n_rand)
    X.setType("mat")
    X.setRandomContext(rand)
    X.setRandomNormal()
    rand.destroy()
    for j in range(n_rand):
        xj = X.getColumn(j)
        if L.get_real_flag():
            row_offset = xj.getOwnershipRange()[0]
            rows = np.arange(rowsizes[0], dtype=PETSc.IntType) + row_offset
            xj.setValues(rows, xj.getArray().real)
            xj.assemble()
        if L.get_block_cc_flag():
            enforce_complex_conjugacy(comm, xj, L.get_nblocks())
        X.restoreColumn(j, xj)
    X.orthogonalize(None)
    # Perform randomized SVD loop
    Qadj = SLEPc.BV().create(comm=comm)
    Qadj.setSizes(L.get_dimensions()[-1], n_rand)
    Qadj.setType("mat")
    Qadj = action_adj(X, Qadj)
    Qadj.orthogonalize(None)
    X.destroy()
    Qfwd = SLEPc.BV().create(comm=comm)
    Qfwd.setSizes(L.get_dimensions()[0], n_rand)
    Qfwd.setType("mat")
    R = create_dense_matrix(PETSc.COMM_SELF, (n_rand, n_rand))
    for j in range(n_loops):
        if verbose == 1:
            str = "Loop %d/%d, forward action" % (j + 1, n_loops)
            petscprint(comm, str)
        Qfwd = action(Qadj, Qfwd)
        Qfwd.orthogonalize(None)
        if verbose == 1:
            str = "Loop %d/%d, adjoint action" % (j + 1, n_loops)
            petscprint(comm, str)
        Qadj = action_adj(Qfwd, Qadj)
        Qadj.orthogonalize(R)
    # Compute low-rank SVD
    u, s, v = sp.linalg.svd(R.getDenseArray())
    R.destroy()
    v = v.conj().T
    s = s[:n_svals]
    u = u[:, :n_svals]
    v = v[:, :n_svals]
    u = PETSc.Mat().createDense(
        (n_rand, n_svals), None, u, comm=PETSc.COMM_SELF
    )
    v = PETSc.Mat().createDense(
        (n_rand, n_svals), None, v, comm=PETSc.COMM_SELF
    )
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
    monitor: typing.Optional[bool] = False,
) -> np.array:
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

    :return: Error vector (each entry is the error of the corresponding
        singular triplet)
    :rtype: np.array
    """
    if monitor:
        petscprint(PETSc.COMM_WORLD, " ")
        petscprint(
            PETSc.COMM_WORLD, "Executing SVD triplet convergence check..."
        )
    x = U.createVec()
    n_svals = S.shape[-1]
    error_vec = np.zeros(n_svals)
    for k in range(n_svals):
        v = V.getColumn(k)
        u = U.getColumn(k)
        x = action(v, x)
        x.scale(1.0 / S[k, k])
        x.axpy(-1.0, u)
        error = x.norm()
        error_vec[k] = error.real
        if monitor:
            str = "Error for SVD triplet %d = %1.15e" % (k + 1, error)
            petscprint(PETSc.COMM_WORLD, str)
        U.restoreColumn(k, u)
        V.restoreColumn(k, v)
    x.destroy()
    if monitor:
        petscprint(
            PETSc.COMM_WORLD, "Executing SVD triplet convergence check..."
        )
        petscprint(PETSc.COMM_WORLD, " ")
