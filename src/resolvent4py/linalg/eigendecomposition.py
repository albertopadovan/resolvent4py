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
from ..utils.random_helpers import generate_random_petsc_vector
from ..utils.vec_helpers import enforce_complex_conjugacy


def arnoldi_iteration(
    L: LinearOperator,
    action: typing.Callable[
        [PETSc.Vec, typing.Optional[PETSc.Vec]], PETSc.Vec
    ],
    krylov_dim: int,
) -> typing.Tuple[SLEPc.BV, np.ndarray]:
    r"""
    Perform the Arnoldi iteration algorithm to compute an 
    orthonormal basis and the corresponding Hessenberg matrix 
    for the range of the linear operator specified by
    :code:`L` and :code:`action`.

    :param L: instance of the :class:`.LinearOperator` class
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply`, 
        :meth:`.LinearOperator.apply_hermitian_transpose`, 
        :meth:`.LinearOperator.solve` or 
        :meth:`.LinearOperator.solve_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param krylov_dim: dimension of the Arnoldi Krylov subspace
    :type krylov_dim: int

    :return: tuple with an orthonormal basis for the Krylov subspace
        and the Hessenberg matrix
    :rtype: (`BV`_ with :code:`krylov_dim` columns, \
        numpy.ndarray of size :code:`krylov_dim x krylov_dim`)
    """
    comm = L._comm
    sizes = (
        L._dimensions[0]
        if action == L.apply or action == L.solve
        else L._dimensions[1]
    )
    nblocks = L.get_nblocks()
    block_cc = L._block_cc
    # Initialize the Hessenberg matrix and the BV for the Krylov subspace
    Q = SLEPc.BV().create(comm=comm)
    Q.setSizes(sizes, krylov_dim)
    Q.setType("mat")
    H = np.zeros((krylov_dim, krylov_dim), dtype=np.complex128)
    complex = False if L._real else True
    q = generate_random_petsc_vector(comm, sizes, complex)
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
        v = action(q, v)
        for j in range(k):
            qj = Q.getColumn(j)
            H[j, k - 1] = v.dot(qj)
            v.axpy(-H[j, k - 1], qj)
            Q.restoreColumn(j, qj)
        if k < krylov_dim:
            H[k, k - 1] = v.norm()
            v.scale(1.0 / H[k, k - 1])
            Q.insertVec(k, v)
        q.destroy()
        q = v.copy()
    q.destroy()
    v.destroy()
    return (Q, H)


def eig(
    L: LinearOperator,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    krylov_dim: int,
    n_evals: int,
    process_evals: typing.Optional[
        typing.Callable[[np.ndarray], np.ndarray]
    ] = None,
) -> typing.Tuple[np.ndarray, SLEPc.BV]:
    r"""
    Compute the eigendecomposition of the linear operator specified by
    :code:`L` and :code:`action`. For example,
    to compute the eigenvalues of :math:`L` closest to the origin,
    set :code:`action = L.solve` and
    :code:`process_evals = lambda x: 1./x`.

    :param L: instance of the :class:`.LinearOperator` class
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply`,
        :meth:`.LinearOperator.apply_hermitian_transpose`,
        :meth:`.LinearOperator.solve` or
        :meth:`.LinearOperator.solve_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param krylov_dim: dimension of the Arnoldi Krylov subspace
    :type krylov_dim: int
    :param n_evals: number of eigenvalues to return
    :type n_evals: int
    :param process_evals: function to extract the desired eigenvalues
        (see description above for an example).
    :type process_evals: Optional[Callable[[np.ndarray], np.ndarray]], default
        is :code:`lambda x: x`

    :return: tuple with the desired eigenvalues and corresponding eigenvectors
    :rtype: (numpy.ndarray of size :code:`n_evals x n_evals`,
        SLEPc.BV with :code:`n_evals` columns)
    """
    Q, H = arnoldi_iteration(L, action, krylov_dim)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:, idces]
    evecs_ = PETSc.Mat().createDense(
        evecs.shape, None, evecs, comm=MPI.COMM_SELF
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
    Scale and sort the right and left eigenvectors and corresponding eigenvalues
    of an underlying operator :math:`L`, so that

    .. math::

        W^* L V = D_v = D_w,\quad W^* V = I \in\mathbb{R}^{m\times m}.


    :param V: :code:`m` right eigenvectors
    :type V: SLEPc.BV
    :param W: :code:`m` left eigenvectors
    :type W: SLEPc.BV
    :param Dv: right eigenvalues
    :type Dv: numpy.ndarray of size :code:`m x m`
    :param Dv: eigenvalues computed from the right eigendecomposition
        of :math:`L`
    :type Dv: numpy.ndarray of size :code:`m x m`
    :param Dw: eigenvalues computed from the left eigendecomposition
        of :math:`L`. (Attention: these have already been complex conjugated.)
    :type Dw: numpy.ndarray of size :code:`m x m`

    :return: tuple :math:`(V, W, D_v, D_w)` with the biorthogonalized
        eigenvectors and corresponding eigenvalues
    :rtype: (SLEPc.BV with :code:`m` columns, SLEPc.BV with :code:`m` columns,
        numpy.ndarray of size :code:`m x m`,
        numpy.ndarray of size :code:`m x m`)
    """
    # Match the right and left eigenvalues/vectors
    Dv = np.diag(Dv)
    Dw = np.diag(Dw)
    idces = [np.argmin(np.abs(Dv - val)) for val in Dw]
    Dw = np.diag(Dw[idces])
    Dv = np.diag(Dv)
    Qadj = W.copy()
    for j in range(len(idces)):
        q = Qadj.getColumn(idces[j])
        W.insertVec(j, q)
        Qadj.restoreColumn(idces[j], q)
    Qadj.destroy()
    # Biorthogonalize the eigenvectors
    M = V.dot(W)
    evals, evecs = sp.linalg.eig(M.getDenseArray())
    idces = np.argwhere(np.abs(evals) < 1e-10).reshape(-1)
    evals[idces] += 1e-10
    Minv = evecs @ np.diag(1.0 / evals) @ sp.linalg.inv(evecs)
    Minv = PETSc.Mat().createDense(Minv.shape, None, Minv, MPI.COMM_SELF)
    V.multInPlace(Minv, 0, V.getSizes()[-1])
    M.destroy()
    Minv.destroy()
    return (V, W, Dv, Dw)


def check_eig_convergence(
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    D: np.ndarray,
    V: SLEPc.BV,
) -> None:
    r"""
    Check convergence of the eigenpairs by measuring
    :math:`\lVert L v - \lambda v\rVert` for each pair :math:`(\lambda, v)`.

    :param action: one of :meth:`.LinearOperator.apply` or
        :meth:`.LinearOperator.apply_hermitian_transpose`
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param D: diagonal 2D numpy array with the eigenvalues
    :type D: numpy.ndarray
    :param V: corresponding eigenvectors
    :type V: SLEPc.BV

    :return: None
    """
    petscprint(MPI.COMM_WORLD, " ")
    petscprint(MPI.COMM_WORLD, "Executing eigenpair convergence check...")
    w = V.createVec()
    for j in range(D.shape[-1]):
        v = V.getColumn(j)
        e = v.copy()
        e.scale(D[j, j])
        w = action(v, w)
        e.axpy(-1.0, w)
        error = e.norm()
        V.restoreColumn(j, v)
        e.destroy()
        str = "Error for eigenpair %d = %1.15e" % (j + 1, error)
        petscprint(MPI.COMM_WORLD, str)
    w.destroy()
    petscprint(MPI.COMM_WORLD, "Executing eigenpair convergence check...")
    petscprint(MPI.COMM_WORLD, " ")
