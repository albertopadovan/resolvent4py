__all__ = [
    "arnoldi_iteration",
    "eig",
    "match_right_and_left_eigenvectors",
    "check_eig_convergence",
]

import typing

import numpy as np
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from ..linear_operators import LinearOperator
from ..utils.miscellaneous import petscprint
from ..utils.random import generate_random_petsc_vector
from ..utils.vector import enforce_complex_conjugacy
from ..utils.matrix import create_dense_matrix
from ..utils.bv import bv_slice


def arnoldi_iteration(
    L: LinearOperator,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    krylov_dim: int,
    verbose: int = 0,
) -> typing.Tuple[SLEPc.BV, np.ndarray]:
    r"""
    Perform the Arnoldi iteration to compute an orthonormal basis
    and the corresponding Hessenberg matrix for the range of the
    linear operator specified by :code:`L` and :code:`action`.

    :param L: linear operator
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply`,
        :meth:`.LinearOperator.apply_hermitian_transpose`,
        :meth:`.LinearOperator.solve` or
        :meth:`.LinearOperator.solve_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param krylov_dim: dimension of the Krylov subspace
    :type krylov_dim: int
    :param verbose: 0 = no output, 1 = print progress
    :type verbose: int, default is 0

    :return: orthonormal basis and Hessenberg matrix
    :rtype: (SLEPc.BV, numpy.ndarray)
    """
    comm = L.get_comm()
    sizes = (
        L.get_dimensions()[0]
        if action == L.apply or action == L.solve
        else L.get_dimensions()[1]
    )
    nblocks = L.get_nblocks()
    block_cc = L.get_block_cc_flag()
    # Initialize the Hessenberg matrix and the BV for the Krylov subspace
    Q = SLEPc.BV().create(comm=comm)
    Q.setSizes(sizes, krylov_dim + 1)
    Q.setType("mat")
    H = np.zeros((krylov_dim + 1, krylov_dim), dtype=np.complex128)
    complex = False if L.get_real_flag() else True
    q = generate_random_petsc_vector(sizes, complex)
    enforce_complex_conjugacy(comm, q, nblocks) if block_cc == True else None
    q.scale(1.0 / q.norm())
    Q.insertVec(0, q)
    # Perform the Arnoldi iterations
    v = (
        L.create_left_vector()
        if action == L.apply or action == L.solve
        else L.create_right_vector()
    )
    for k in range(1, krylov_dim + 1):
        if verbose == 1:
            petscprint(comm, "Arnoldi iteration %d/%d" % (k, krylov_dim))
        v = action(q, v)
        Q.setActiveColumns(0, k)
        # Two-pass Classical Gram-Schmidt (CGS2): same stability as
        # Modified Gram-Schmidt but with only 2 Allreduces per step
        # instead of k (see Daniel, Gragg, Kaufman & Stewart, 1976)
        h = Q.dotVec(v)
        Q.multVec(-1.0, 1.0, v, h)
        h2 = Q.dotVec(v)
        Q.multVec(-1.0, 1.0, v, h2)
        H[:k, k - 1] = h + h2
        H[k, k - 1] = v.norm()
        v.scale(1.0 / H[k, k - 1])
        Q.insertVec(k, v)
        v.copy(q)
    q.destroy()
    v.destroy()
    Q.setActiveColumns(0, krylov_dim)
    return (Q, H[:-1,])


def eig(
    L: LinearOperator,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    krylov_dim: int,
    n_evals: int,
    process_evals: typing.Optional[
        typing.Callable[[np.ndarray], np.ndarray]
    ] = None,
    verbose: int = 0,
) -> typing.Tuple[np.ndarray, SLEPc.BV]:
    r"""
    Compute the eigendecomposition of the linear operator specified by
    :code:`L` and :code:`action`. For example, to compute the eigenvalues
    of :math:`L` closest to the origin, set :code:`action = L.solve` and
    :code:`process_evals = lambda x: 1./x`.

    :param L: linear operator
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply`,
        :meth:`.LinearOperator.apply_hermitian_transpose`,
        :meth:`.LinearOperator.solve` or
        :meth:`.LinearOperator.solve_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param krylov_dim: dimension of the Krylov subspace
    :type krylov_dim: int
    :param n_evals: number of eigenvalues to return
    :type n_evals: int
    :param process_evals: function to transform the eigenvalues
        of :code:`action` into eigenvalues of the desired operator
    :type process_evals: Optional[Callable[[np.ndarray], np.ndarray]]
    :param verbose: 0 = no output, 1 = print progress
    :type verbose: int, default is 0

    :return: eigenvalues as a diagonal matrix and corresponding eigenvectors
    :rtype: (numpy.ndarray of size :code:`n_evals x n_evals`,
        SLEPc.BV with :code:`n_evals` columns)
    """
    Q, H = arnoldi_iteration(L, action, krylov_dim, verbose)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:, idces]
    evecs_ = PETSc.Mat().createDense(
        evecs.shape, None, evecs, comm=PETSc.COMM_SELF
    )
    Q.multInPlace(evecs_, 0, n_evals)
    Q.setActiveColumns(0, n_evals)
    Q.resize(n_evals, copy=True)
    evals = evals if process_evals is None else process_evals(evals)
    evecs_.destroy()
    return (np.diag(evals), Q)


def match_right_and_left_eigenvectors(
    V: SLEPc.BV, W: SLEPc.BV, Dv: np.ndarray, Dw: np.ndarray
) -> typing.Tuple[SLEPc.BV, SLEPc.BV, np.ndarray, np.ndarray]:
    r"""
    Sort, match, and biorthogonalize the right and left eigenvectors
    of an operator :math:`L`, so that

    .. math::

        W^* L V = D_v = D_w,\quad W^* V = I \in\mathbb{R}^{m\times m}.

    The right eigenvalues are sorted by descending real part, and
    the left eigenvectors/eigenvalues are reordered to match.

    :param V: right eigenvectors of :math:`L`
    :type V: SLEPc.BV
    :param W: right eigenvectors of :math:`L^*`
        (conjugated internally to obtain left eigenvectors of :math:`L`)
    :type W: SLEPc.BV
    :param Dv: right eigenvalues of :math:`L` as a diagonal matrix
    :type Dv: numpy.ndarray of size :code:`m x m`
    :param Dw: right eigenvalues of :math:`L^*` as a diagonal matrix
        (conjugated internally to obtain left eigenvalues of :math:`L`)
    :type Dw: numpy.ndarray of size :code:`m x m`

    :return: biorthogonalized :math:`(V, W, D_v, D_w)`
    :rtype: (SLEPc.BV, SLEPc.BV, numpy.ndarray, numpy.ndarray)
    """
    # Sort right eigenvalues/vectors by descending real part
    Dv = np.diag(Dv)
    sort_idces = np.flipud(np.argsort(Dv.real))
    Dv = Dv[sort_idces]
    V = bv_slice(V, sort_idces)
    # Match the left eigenvalues/vectors to the right ones
    Dw = np.conj(np.diag(Dw))
    idces = [np.argmin(np.abs(Dw - val)) for val in Dv]
    Dw = np.diag(np.conj(Dw[idces]))
    Dv = np.diag(Dv)
    W = bv_slice(W, np.array(idces))
    # Biorthogonalize the eigenvectors
    M = V.dot(W)
    evals, evecs = sp.linalg.eig(M.getDenseArray())
    idces = np.argwhere(np.abs(evals) < 1e-10).reshape(-1)
    # evals[idces] += 1e-10
    Minv = evecs @ np.diag(1.0 / evals) @ sp.linalg.inv(evecs)
    MinvH = PETSc.Mat().createDense(Minv.shape, None, np.conj(Minv).T, PETSc.COMM_SELF)
    W.multInPlace(MinvH, 0, W.getSizes()[-1])
    M.destroy()
    return (V, W, Dv, Dw)


def check_eig_convergence(
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    D: np.ndarray,
    V: SLEPc.BV,
    monitor: bool = False,
) -> np.ndarray:
    r"""
    Check convergence of eigenpairs by computing
    :math:`\lVert L v - \lambda v\rVert` for each pair
    :math:`(\lambda, v)`.

    :param action: one of :meth:`.LinearOperator.apply` or
        :meth:`.LinearOperator.apply_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param D: eigenvalues as a diagonal matrix
    :type D: numpy.ndarray
    :param V: corresponding eigenvectors
    :type V: SLEPc.BV
    :param monitor: print per-eigenpair errors if True
    :type monitor: bool, default is False

    :return: error for each eigenpair
    :rtype: numpy.ndarray
    """
    if monitor:
        petscprint(PETSc.COMM_WORLD, " ")
        petscprint(
            PETSc.COMM_WORLD, "Executing eigenpair convergence check..."
        )
    error_vec = np.zeros(D.shape[0])
    w = V.createVec()
    for j in range(D.shape[-1]):
        v = V.getColumn(j)
        e = v.copy()
        e.scale(D[j, j])
        w = action(v, w)
        e.axpy(-1.0, w)
        error = e.norm()
        error_vec[j] = error.real
        V.restoreColumn(j, v)
        e.destroy()
        if monitor:
            str = "Error for eigenpair %d = %1.15e" % (j + 1, error)
            petscprint(PETSc.COMM_WORLD, str)
    w.destroy()
    if monitor:
        petscprint(
            PETSc.COMM_WORLD, "Executing eigenpair convergence check..."
        )
        petscprint(PETSc.COMM_WORLD, " ")
    return error_vec
