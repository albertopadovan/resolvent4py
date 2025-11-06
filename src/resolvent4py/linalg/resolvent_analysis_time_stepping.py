__all__ = ["resolvent_analysis_rsvd_dt"]

import typing

import numpy as np
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

from ..linear_operators import LinearOperator
from ..linear_operators.matrix import MatrixLinearOperator
from ..utils.matrix import create_dense_matrix
from ..utils.matrix import create_AIJ_identity
from ..utils.miscellaneous import petscprint
from ..utils.vector import vec_real
from ..utils.time_stepping import compute_post_transient_solution
from ..utils.time_stepping import create_time_and_frequency_arrays


def _reorder_list(Qlist: list[SLEPc.BV], Qlist_reordered: list[SLEPc.BV]):
    for j in range(Qlist[0].getSizes()[-1]):
        for i in range(len(Qlist)):
            Qij = Qlist[i].getColumn(j)
            Qlist_reordered[j].insertVec(i, Qij)
            Qlist[i].restoreColumn(j, Qij)
    return Qlist_reordered


def resolvent_analysis_rsvd_dt(
    L: LinearOperator,
    dt: float,
    omega: float,
    n_omegas: int,
    n_periods: int,
    n_rand: int,
    n_loops: int,
    n_svals: int,
    B: typing.Optional[typing.Union[LinearOperator, None]] = None,
    C: typing.Optional[typing.Union[LinearOperator, None]] = None,
    tol: typing.Optional[float] = 1e-3,
    time_stepper: typing.Optional[str] = "RK2",
    verbose: typing.Optional[int] = 0,
) -> typing.Tuple[SLEPc.BV, np.ndarray, SLEPc.BV]:
    r"""
    Perform resolvent analysis using randomized linear algebra and time
    stepping.
    In particular, it can be shown that

    .. math::

        x_\omega e^{i\omega t} = \left(i\omega I - A\right)^{-1}f_\omega
            e^{i\omega t}
            \to \int_0^t e^{A(t-\tau)}f_\omega e^{i\omega\tau} d\tau

    for sufficiently long time :math:`t \gg 1` and for any complex-valued
    forcing :math:`f(t) = f_\omega e^{i\omega t}` (assuming that :math:`A` is
    stable).
    Computing the integral on the right-hand side can be done by integrating

    .. math::

        \frac{d}{dt}x(t) = Ax(t) + f(t)

    forward in time with initial condition :math:`x(0) = 0`.
    Thus, the action of the resolvent operator
    :math:`R(i\omega) = \left(i\omega I - A\right)^{-1}` on a vector
    :math:`f_\omega` can be computed by time-stepping the ODE above.
    For now, time integration is performed explicitly via the
    Adams-Bashforth scheme.
    (See [Martini2021]_ and [Farghadan2025]_ for more details.)

    .. note::

        This function is meant for systems whose dimension is so large
        that linear systems of the form
        :math:`(i\omega I - A)x_\omega = f_\omega`
        cannot be solved easily. Typically, this happens when
        :math:`\text{dim}(x_\omega) \sim O(10^7)` or larger.
        If you have a "small enough" system, then we highly recommend using
        :func:`.randomized_svd.randomized_svd` instead for the singular
        value decomposition of the resolvent.

    :param L: linear operator representing :math:`A`
    :type L: LinearOperator

    :param dt: target time step :math:`\Delta t` to integrate the ODE
    :type dt: float

    :param omega: fundamental frequency :math:`\omega` of the forcing
    :type omega: float

    :param n_omegas: number of integer harmonics of :math:`\omega` to resolve
    :type n_omegas: int

    :param n_periods: number of periods :math:`T = 2\pi/\omega` to integrate
        the ODE through. (This number should be large enough that we can
        expect transients to have decayed.)
    :type n_periods: int

    :param n_rand: number of random forcing vectors for randomized SVD
    :type n_rand: int

    :param n_loops: number of power iterations for randomized SVD
    :type n_loops: int

    :param n_svals: number of singular values/vectors to output
    :type n_svals: int

    :param tol: integrate the ODE forward for :code:`n_periods` or until
        :math:`\lVert x(kT) - x((k-1)T) \rVert < \mathrm{tol}`.
    :type tol: Optional[float], default is :math:`10^{-3}`

    :param time_stepper: integrator (one of 'RK2' or 'RK3')
    :type time_stepper: Optional[str], default is 'RK2'

    :param verbose: defines verbosity of output to terminal (useful to
        monitor progress during time stepping). = 0 no printout to terminal,
        = 1 monitor randomized SVD iterations, = 2 monitor randomized SVD
        iterations and time-stepping progress.
    :type verbose: Optional[int], default is 0

    :return: left singular vectors, singular values, and right singular vectors
        of the resolvent operators :math:`R(i\omega)` evaluated at frequencies
        :math:`\Omega = \omega\{0, 1, 2, \ldots, n_{\omega}\}` if
        the linear operator :math:`A` is real-valued; otherwise at frequencies
        :math:`\Omega = \omega\{0, 1, 2, \ldots, n_{\omega}, -n_{\omega},-(n_{\omega}-1) \ldots, -1\}`
    :rtype: Tuple[List[SLEPc.BV], List[np.ndarray], List[SLEPc.BV]]

    References
    ----------
    .. [Martini2021] Martini et al., *Efficient computation of global
        resolvent modes*, Journal of Fluid Mechanics, 2021
    .. [Farghadan2025] Farghadan et al., *Scalable resolvent analysis
        for three-dimensional flows*, Journal of Computational Physics, 2025
    """

    size = L.get_dimensions()[0]

    Id = create_AIJ_identity(L.get_comm(), (size, size))
    Idop = MatrixLinearOperator(Id)
    B = Idop if B == None else B
    C = Idop if C == None else C

    size_input = B.get_dimensions()[-1]
    size_output = C.get_dimensions()[-1]

    real = L.get_real_flag()
    tsim, nsave, omegas = create_time_and_frequency_arrays(
        dt, omega, n_omegas, real
    )
    n_omegas = len(omegas)

    Qadj_hat_lst, Qfwd_hat_lst = [], []
    for _ in range(n_rand):
        # Set seed
        rank = L.get_comm().getRank()
        rand = PETSc.Random().create(comm=L.get_comm())
        rand.setType(PETSc.Random.Type.RAND)
        rand.setSeed(round(np.random.randint(1000, 100000) + rank))

        # Initialize Qadj_hat and Qfwd_hat with random BVs of appropriate dim
        X = SLEPc.BV().create(comm=L.get_comm())
        X.setSizes(size_input, n_omegas)
        X.setType("mat")
        X.setRandomContext(rand)
        X.setRandomNormal()
        if L.get_real_flag():
            v = X.getColumn(0)
            v = vec_real(v, True)
            X.restoreColumn(0, v)
        Qadj_hat_lst.append(X.copy())
        X.destroy()

        X = SLEPc.BV().create(comm=L.get_comm())
        X.setSizes(size_output, n_omegas)
        X.setType("mat")
        X.setRandomContext(rand)
        X.setRandomNormal()
        if L.get_real_flag():
            v = X.getColumn(0)
            v = vec_real(v, True)
            X.restoreColumn(0, v)
        Qfwd_hat_lst.append(X.copy())
        X.destroy()

        rand.destroy()

    # Initialize Qadj_hat and Qfwd_hat with BVs of size N x n_omegas
    X = SLEPc.BV().create(comm=L.get_comm())
    X.setSizes(size_input, n_rand)
    X.setType("mat")
    Qadj_hat_lst2 = [X.copy() for _ in range(n_omegas)]
    X.destroy()

    X = SLEPc.BV().create(comm=L.get_comm())
    X.setSizes(size_output, n_rand)
    X.setType("mat")
    Qfwd_hat_lst2 = [X.copy() for _ in range(n_omegas)]
    X.destroy()

    Qadj_hat_lst2 = _reorder_list(Qadj_hat_lst, Qadj_hat_lst2)
    for i in range(len(Qadj_hat_lst2)):
        Qadj_hat_lst2[i].orthogonalize(None)
    Qadj_hat_lst = _reorder_list(Qadj_hat_lst2, Qadj_hat_lst)

    Qadj = SLEPc.BV().create(comm=L.get_comm())
    Qadj.setSizes(size, len(tsim[::nsave]))
    Qadj.setType("mat")
    Qfwd = Qadj.duplicate()

    x = L.create_left_vector()
    for j in range(n_loops):
        for k in range(n_rand):
            if verbose > 0:
                str = "Loop %d/%d, random vector %d/%d (forward action)" % (
                    j + 1,
                    n_loops,
                    k + 1,
                    n_rand,
                )
                petscprint(L.get_comm(), str)
            x.zeroEntries()
            Qfwd_hat_lst[k] = compute_post_transient_solution(
                L,
                B,
                C,
                L.apply,
                tsim,
                nsave,
                n_periods,
                omegas,
                x,
                Qadj_hat_lst[k],
                Qfwd_hat_lst[k],
                Qfwd,
                tol,
                time_stepper,
                verbose,
            )
        Qfwd_hat_lst2 = _reorder_list(Qfwd_hat_lst, Qfwd_hat_lst2)
        for i in range(len(Qfwd_hat_lst2)):
            Qfwd_hat_lst2[i].orthogonalize(None)
        Qfwd_hat_lst = _reorder_list(Qfwd_hat_lst2, Qfwd_hat_lst)

        for k in range(n_rand):
            if verbose > 0:
                str = "Loop %d/%d, random vector %d/%d (adjoint action)" % (
                    j + 1,
                    n_loops,
                    k + 1,
                    n_rand,
                )
                petscprint(L.get_comm(), str)
            x.zeroEntries()
            Qadj_hat_lst[k] = compute_post_transient_solution(
                L,
                C,
                B,
                L.apply_hermitian_transpose,
                tsim,
                nsave,
                n_periods,
                omegas,
                x,
                Qfwd_hat_lst[k],
                Qadj_hat_lst[k],
                Qadj,
                tol,
                time_stepper,
                verbose,
            )
        Qadj_hat_lst2 = _reorder_list(Qadj_hat_lst, Qadj_hat_lst2)
        Rlst = []
        R = create_dense_matrix(PETSc.COMM_SELF, (n_rand, n_rand))
        for i in range(len(Qadj_hat_lst2)):
            Qadj_hat_lst2[i].orthogonalize(R)
            Rlst.append(R.copy())
        R.destroy()
        Qadj_hat_lst = _reorder_list(Qadj_hat_lst2, Qadj_hat_lst)
        if j < n_loops - 1:
            for obj in Rlst:
                obj.destroy()

    x.destroy()
    # Compute low-rank SVD
    Slst = []
    for j, R in enumerate(Rlst):
        u, s, v = sp.linalg.svd(R.getDenseArray())
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
        Qfwd_hat_lst2[j].multInPlace(v, 0, n_svals)
        Qfwd_hat_lst2[j].setActiveColumns(0, n_svals)
        Qfwd_hat_lst2[j].resize(n_svals, copy=True)
        Qadj_hat_lst2[j].multInPlace(u, 0, n_svals)
        Qadj_hat_lst2[j].setActiveColumns(0, n_svals)
        Qadj_hat_lst2[j].resize(n_svals, copy=True)
        Slst.append(np.diag(s))
        u.destroy()
        v.destroy()

    lists = [Rlst, Qfwd_hat_lst, Qadj_hat_lst]
    for lst in lists:
        for obj in lst:
            obj.destroy()

    Idop.destroy()

    return Qfwd_hat_lst2, Slst, Qadj_hat_lst2
