__all__ = [
    "arnoldi_iteration",
    "eig",
    "match_right_and_left_eigenvectors",
    "check_eig_convergence",
    "eig_two_sided",
]

import typing

import numpy as np
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

from ..linear_operators import LinearOperator
from ..utils.miscellaneous import petscprint
from ..utils.random import generate_random_petsc_vector
from ..utils.vector import enforce_complex_conjugacy
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
    enforce_complex_conjugacy(q, nblocks) if block_cc == True else None
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

def eig_two_sided(
    L: LinearOperator,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    krylov_dim: int,
    n_evals: int,
    process_evals: typing.Optional[
        typing.Callable[[np.ndarray], np.ndarray]
    ] = None,
    verbose: int = 0,
) -> typing.Tuple[np.ndarray, SLEPc.BV, SLEPc.BV]:
    r"""
    Two-sided (Petrov-Galerkin) eigendecomposition of the linear operator
    specified by :code:`L` and :code:`action`, returning both **right**
    and **left** eigenvectors biorthogonalized to satisfy

    .. math::

        L\,V = V\,D, \quad L^{*}\,W = W\,D^{*}, \quad W^{*} V = I.

    Internally this runs two independent Arnoldi sweeps: one on
    :code:`action` (forward Krylov basis :math:`Q_f`, Hessenberg
    :math:`H_f`), and one on the Hermitian-transposed action -- i.e.
    :meth:`.LinearOperator.apply_hermitian_transpose` if
    :code:`action == L.apply`, otherwise
    :meth:`.LinearOperator.solve_hermitian_transpose` -- giving the
    adjoint Krylov basis :math:`Q_a`.  The cross-Gram
    :math:`G = Q_a^{*} Q_f` is SVD'd as :math:`G = U \Sigma V^{*}`, and
    the reduced operator

    .. math::

        \tilde{H} \;=\; (U\,\Sigma^{-1/2})^{*}\,G\,H_f\,(V\,\Sigma^{-1/2})

    is eigendecomposed in compact form to yield both right and left
    eigenvectors of the projected problem.  These are then lifted back
    via :math:`V = Q_f (V\,\Sigma^{-1/2})\,X_r` and
    :math:`W = Q_a (U\,\Sigma^{-1/2})\,X_\ell`, and the left set is
    rescaled so that :math:`W^{*} V = I` to machine precision *before*
    return -- no extra
    :func:`match_right_and_left_eigenvectors` pass is needed.

    Eigenvalues are sorted by descending magnitude and the top
    :code:`n_evals` retained.

    To target eigenvalues of :math:`L` closest to the origin, set
    :code:`action = L.solve` and :code:`process_evals = lambda x: 1./x`
    (the adjoint sweep is then automatically wired through
    :code:`L.solve_hermitian_transpose`).

    :param L: linear operator
    :type L: :class:`.LinearOperator`
    :param action: one of :meth:`.LinearOperator.apply` or
        :meth:`.LinearOperator.solve`.  The matching Hermitian-transposed
        action is picked up automatically for the adjoint sweep, so
        :code:`L` must implement
        :meth:`.LinearOperator.apply_hermitian_transpose` (resp.
        :meth:`.LinearOperator.solve_hermitian_transpose`).
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param krylov_dim: dimension of each Krylov subspace (same for
        forward and adjoint sweeps)
    :type krylov_dim: int
    :param n_evals: number of converged eigenpairs to return
    :type n_evals: int
    :param process_evals: function applied to the eigenvalues *after*
        sorting and truncation, to map eigenvalues of :code:`action`
        into eigenvalues of the operator of interest (e.g.
        :code:`lambda x: 1./x` when :code:`action = L.solve`).  Applied
        identically to both spectra.
    :type process_evals: Optional[Callable[[np.ndarray], np.ndarray]]
    :param verbose: 0 = no output, 1 = print Arnoldi progress
    :type verbose: int, default is 0

    :return: triplet :code:`(D, V, W)` where :code:`D` is the diagonal
        :code:`n_evals x n_evals` matrix of (post-:code:`process_evals`)
        eigenvalues, :code:`V` is the BV of right eigenvectors, and
        :code:`W` is the BV of left eigenvectors (already
        biorthogonalized: :code:`W.dot(V) == I` to machine precision).
    :rtype: (numpy.ndarray of size :code:`n_evals x n_evals`,
        SLEPc.BV with :code:`n_evals` columns,
        SLEPc.BV with :code:`n_evals` columns)
    """
    petscprint(L.get_comm(), "Two-sided eig: launching forward Arnoldi")
    Qfwd, Hfwd = arnoldi_iteration(L, action, krylov_dim, verbose)
    petscprint(L.get_comm(), "Two-sided eig: launching adjoint Arnoldi")
    action_adj = L.apply_hermitian_transpose if action == L.apply else L.solve_hermitian_transpose
    Qadj, _ = arnoldi_iteration(L, action_adj, krylov_dim, verbose)

    G = Qfwd.dot(Qadj)
    Ga = G.getDenseArray()
    u, s, v = sp.linalg.svd(Ga)
    v = v.conj().T
    idces = np.argwhere(s > 1e-12).reshape(-1)
    u = u[:, idces]
    s = s[idces]
    v = v[:, idces]

    Ssqrt_inv = np.diag(1 / np.sqrt(s))
    Phi = v @ Ssqrt_inv
    Psi = u @ Ssqrt_inv
    Htil = Psi.conj().T @ (Ga @ Hfwd) @ Phi
    evals, evecs_left, evecs_right = sp.linalg.eig(Htil, left=True, right=True)

    idces = np.flipud(np.argsort(np.abs(evals)))[:n_evals]
    evecs_right = evecs_right[:, idces]
    evecs_left = evecs_left[:, idces]
    evecs_left = evecs_left @ sp.linalg.inv(evecs_right.conj().T @ evecs_left)
    evecs_right = Phi @ evecs_right
    evecs_left = Psi @ evecs_left

    evecs_r = PETSc.Mat().createDense(
        evecs_right.shape, None, evecs_right, comm=PETSc.COMM_SELF
    )
    Qfwd.multInPlace(evecs_r, 0, n_evals)
    Qfwd.setActiveColumns(0, n_evals)
    Qfwd.resize(n_evals, copy=True)
    evecs_r.destroy()

    evecs_l = PETSc.Mat().createDense(
        evecs_left.shape, None, evecs_left, comm=PETSc.COMM_SELF
    )
    Qadj.multInPlace(evecs_l, 0, n_evals)
    Qadj.setActiveColumns(0, n_evals)
    Qadj.resize(n_evals, copy=True)
    evecs_l.destroy()

    evals = evals[idces]
    evals = evals if process_evals is None else process_evals(evals)
    
    return (np.diag(evals), Qfwd, Qadj)

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
    u, s, v = sp.linalg.svd(M.getDenseArray())
    v = v.conj().T
    idces = np.argwhere(np.abs(s) > 1e-12).reshape(-1)
    # Compute pseudo inverse (this will be the exact inverse if M is full rank)
    Minv = v[:, idces] @ np.diag(1.0 / s[idces]) @ (u[:, idces]).conj().T
    MinvH_data = Minv.conj().T
    MinvH = PETSc.Mat().createDense(
        MinvH_data.shape, None, MinvH_data, PETSc.COMM_SELF
    )
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
        lj = D[j, j]
        e.scale(lj)
        w = action(v, w)
        e.axpy(-1.0, w)
        error = e.norm()
        error_vec[j] = error.real
        V.restoreColumn(j, v)
        e.destroy()
        if monitor:
            str = (
                "Error for eigenpair %d (lam = %1.5e + i%1.5e) = %1.15e"
                % (j + 1, lj.real, lj.imag, error)
            ) if lj.imag >= 0 else (
                "Error for eigenpair %d (lam = %1.5e - i%1.5e) = %1.15e"
                % (j + 1, lj.real, -lj.imag, error)
            ) 
            petscprint(PETSc.COMM_WORLD, str)
    w.destroy()
    if monitor:
        petscprint(
            PETSc.COMM_WORLD, "Executing eigenpair convergence check..."
        )
        petscprint(PETSc.COMM_WORLD, " ")
    return error_vec
