import typing
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from ..utils.vector import vec_real
from ..utils.miscellaneous import petscprint

if typing.TYPE_CHECKING:
    from ..linear_operators import LinearOperator


def fft(
    X: SLEPc.BV,
    Xhat: SLEPc.BV,
    real: typing.Optional[bool] = True,
    harmonic_balancing_ordering: typing.Optional[bool] = False,
) -> SLEPc.BV:
    r"""
    Compute the discrete Fourier transform of time-domain samples
    stored column-wise in a SLEPc BV.

    Each row of ``X`` is one spatial DOF; column :math:`i` holds the
    value at time :math:`t_i`.  The Fourier coefficients

    .. math::

        \hat{X}_k = \frac{1}{n_{\text{tstore}}}
            \sum_{i=0}^{n_{\text{tstore}}-1}
            X(t_i)\, e^{-i k \omega t_i}

    are written into ``Xhat`` (one column per kept frequency) and
    normalised by ``1 / n_tstore``, where
    ``n_tstore = X.getActiveColumns()[-1]``.

    Two modes:

    - ``real=True`` — assume a real-valued signal and use
      :func:`numpy.fft.rfft`; keeps the first ``n_omegas`` non-negative
      Fourier coefficients
      :math:`\hat{X}_0, \hat{X}_1, \ldots, \hat{X}_{n_{\omega}-1}`.
      The negative half is recovered implicitly via conjugate
      symmetry.
    - ``real=False`` — full complex transform via
      :func:`numpy.fft.fft`.  Keeps a symmetric two-sided spectrum
      ``[0, 1, ..., m, -m, ..., -1]`` (numpy ordering), with
      :math:`m = (n_{\omega} - 1) / 2`.  If
      ``harmonic_balancing_ordering`` is ``True``, the columns are
      reordered into the harmonic-balanced convention
      ``[-m, ..., -1, 0, 1, ..., m]``.

    :param X: time-domain samples; column :math:`i` is :math:`X(t_i)`
    :type X: SLEPc.BV
    :param Xhat: output buffer; one column per kept frequency.  Its
        number of columns determines how many harmonics are kept.
    :type Xhat: SLEPc.BV
    :param real: whether to treat the signal as real-valued
    :type real: Optional[bool], default ``True``
    :param harmonic_balancing_ordering: only used when ``real=False``;
        reorder negative frequencies to the front of ``Xhat``
    :type harmonic_balancing_ordering: Optional[bool], default ``False``

    :return: the same ``Xhat`` BV, filled with the Fourier coefficients
    :rtype: SLEPc.BV
    """
    n_omegas = Xhat.getSizes()[-1]
    n_tstore = X.getActiveColumns()[-1]

    Xhat_mat = Xhat.getMat()
    Xhat_mat_a = Xhat_mat.getDenseArray()
    Xmat = X.getMat()
    Xmat_a = Xmat.getDenseArray().copy()
    if real:
        Xhat_mat_a[:, :] = (
            np.fft.rfft(Xmat_a.real, axis=-1)[:, :n_omegas] / n_tstore
        )
    else:
        n_omegas = int((n_omegas - 1) // 2 + 1)
        idces_pos = np.arange(n_omegas)
        idces_neg = np.arange(-n_omegas + 1, 0)
        idces = np.concatenate((idces_pos, idces_neg))
        Xhat_mat_a[:, :] = np.fft.fft(Xmat_a, axis=-1)[:, idces] / n_tstore
        if harmonic_balancing_ordering:
            idces = np.concatenate((idces_neg, idces_pos))
            Xhat_mat_a[:, :] = Xhat_mat_a[:, idces]

    X.restoreMat(Xmat)
    Xhat.restoreMat(Xhat_mat)
    return Xhat


def ifft(
    Xhat: SLEPc.BV, x: PETSc.Vec, omegas: np.array, t: float
) -> PETSc.Vec:
    r"""
    Evaluate the inverse discrete Fourier transform at a single
    physical time :math:`t`.

    Reconstructs

    .. math::

        x(t) = \sum_{k} \hat{X}_k\, e^{i \omega_k t}

    by computing the column combination
    :math:`\hat{X}\,(e^{i \omega_k t})_k` via
    :meth:`slepc4py.SLEPc.BV.multVec`, and writes the result into the
    pre-allocated PETSc vector ``x``.

    Two modes, selected automatically from ``omegas``:

    - **Real-valued signal** (``min(omegas) == 0``): ``Xhat`` is
      assumed to hold only the non-negative-frequency coefficients
      :math:`\hat{X}_0, \hat{X}_1, \ldots`  The negative half is
      reconstructed via conjugate symmetry by weighting the
      coefficients by ``[1, 2, 2, ..., 2]`` and taking the real part
      of the result (zero frequency keeps weight 1, all others are
      doubled).
    - **Complex signal**: ``Xhat`` holds the full two-sided spectrum
      and the reconstruction is the direct complex sum.

    :param Xhat: Fourier coefficients; column :math:`k` is
        :math:`\hat{X}_k` and ``omegas[k]`` is its frequency
    :type Xhat: SLEPc.BV
    :param x: pre-allocated output vector; reused and returned
    :type x: PETSc.Vec
    :param omegas: 1-D array of angular frequencies matching the
        columns of ``Xhat``.  Non-negative entries indicate a
        real-valued signal.
    :type omegas: np.array
    :param t: physical time at which to evaluate the series
    :type t: float

    :return: the same ``x`` PETSc Vec, holding :math:`x(t)`
    :rtype: PETSc.Vec
    """
    q = np.exp(1j * omegas * t)
    if np.min(omegas) == 0.0:
        c = 2 * np.ones(len(q))
        c[0] = 1.0
        q *= c
        Xhat.multVec(1.0, 0.0, x, q)
        x = vec_real(x, True)
    else:
        Xhat.multVec(1.0, 0.0, x, q)
    return x


def create_time_and_frequency_arrays(
    dt: float, omega: float, n_omegas: int, real: bool
) -> typing.Tuple[np.array, int, np.array]:
    r"""
    Build mutually-consistent simulation-time and frequency arrays for
    a signal with fundamental period :math:`T = 2\pi / \omega`.

    Starts from an aliasing-safe save grid of
    :math:`2 (n_{\omega} + 2)` equispaced samples on :math:`[0, T)`,
    yielding the save-step
    :math:`\Delta t_{\text{store}} = T / (2(n_{\omega}+2))`.  The
    requested integration step ``dt`` is then rounded down to the
    nearest fraction of :math:`\Delta t_{\text{store}}` (so that
    ``nsave = dt_store / dt`` is an exact integer), and the
    integration grid

    .. math::

        t_{\text{sim}} = \big[0,\, \Delta t,\, 2\Delta t,\, \ldots,\,
        n_{\text{steps}} \Delta t\big],
        \qquad n_{\text{steps}} = \mathrm{round}(T / \Delta t),

    is returned alongside the integer stride ``nsave`` such that
    ``tsim[::nsave]`` (excluding the final endpoint) is the save grid.

    The frequency vector is

    .. math::

        \Omega = \omega \cdot \{0, 1, \ldots, n_{\omega}\}

    for ``real=True``, or its two-sided extension
    :math:`\omega \cdot \{0, 1, \ldots, n_{\omega}, -n_{\omega},
    \ldots, -1\}` (numpy FFT ordering) for ``real=False``.

    :param dt: requested integration step; the returned grid may use a
        slightly smaller step so that ``dt_store / dt`` is an integer
    :type dt: float
    :param omega: fundamental angular frequency
        :math:`\omega = 2\pi / T`
    :type omega: float
    :param n_omegas: number of positive harmonics to retain
    :type n_omegas: int
    :param real: whether the signal is real-valued (one-sided
        frequency vector) or complex (two-sided)
    :type real: bool

    :return: ``(tsim, nsave, omegas)``

        - ``tsim``: simulation time grid of length
          ``nsteps + 1``
        - ``nsave``: integer stride such that ``tsim[::nsave]`` is
          the harmonic save grid (Nyquist-safe for the quadratic)
        - ``omegas``: angular-frequency vector, length
          ``n_omegas + 1`` if ``real`` else ``2*n_omegas + 1``
    :rtype: Tuple[np.array, int, np.array]

    :raises ValueError: if the constructed time grids fail the
        internal consistency check.
    """
    T = 2 * np.pi / omega
    tstore = np.linspace(0, T, num=2 * (n_omegas + 2), endpoint=False)
    dt_store = tstore[1] - tstore[0]
    dt = dt_store / round(dt_store / dt)
    nsteps = round(T / dt)
    tsim = dt * np.arange(0, nsteps + 1)
    nsave = round(dt_store / dt)
    if len(tsim[::nsave]) - 1 != len(tstore):
        raise ValueError(f"The time vectors were not constructed properly.")
    omegas = np.arange(n_omegas + 1) * omega
    omegas = (
        omegas if real else np.concatenate((omegas, -np.flipud(omegas[1:])))
    )
    return tsim, nsave, omegas


def solve_ivp(
    v: PETSc.Vec,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    t0: float,
    tf: float,
    nsteps: int,
    method: typing.Optional[str] = "RK2",
    m: typing.Optional[int] = -1,
    adjoint: typing.Optional[bool] = False,
    X: typing.Optional[SLEPc.BV] = None,
    periodic_forcing: typing.Optional[typing.Tuple[SLEPc.BV, np.array]] = None,
) -> typing.Union[PETSc.Vec, SLEPc.BV]:
    r"""
    Integrate a linear (time-invariant) system of equations of the form

    .. math::

        \frac{d}{dt}x(t) = A x(t) + f(t),\quad x(0) = v,

    from :math:`t = t_0` to :math:`t = t_f`.
    If the flag `adjoint` is :code:`True`, we solve

    .. math::

        -\frac{d}{dt}x(t) = A x(t) + f(t),\quad x(t_f) = v,\, t\in [t_0, t_f],

    backward in time from :math:`t = t_f` to :math:`t = t_0`.
    In both cases, the forcing function :math:`f(t)` is periodic and given by

    .. math::

        f(t) = f(t + T) = \sum_{k=-r}^r f_k e^{ik\omega t},\quad \omega = 2\pi/T.

    :param v: initial condition
    :type v: PETSc.Vec
    :param action: callable that defines the action of the linear operator
        A on a vector. One of `A.apply` or `A.apply_hermitian_transpose`.
    :type action: Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec]
    :param t0: initial time
    :type t0: float
    :param tf: final time
    :type tf: float
    :param nsteps: number of time steps to integrate the ODE
    :type nsteps: int
    :param method: integrator (e.g., Runge-Kutta 2)
    :type method: Optional[str], default is 'RK2'
    :param m: save the solution every :math:`m` steps. Should be either a
        number > 0 or -1 (with -1 indicating that we save only the solution
        at the final time :math:`t_f`).
    :type m: Optional[int], default is -1
    :param adjoint: flag to indicate whether we are integrating forward or
        backward in time.
    :type adjoint: Optional[bool], default is False
    :param X: structure to store the solution. Useful only if :math:`m > 0`.
    :type X: Optional[Union[SLEPc.BV, None]], default is None
    :param periodic_forcing: Fourier modes Fhat of the forcing function
        :math:`f(t)`, and array of frequencies corresponding to those modes.
        If :math:`f(t)` is real-valued, define

        .. math::

            \hat{F} = \begin{bmatrix}
            f_0 & f_1 & \ldots, f_r
            \end{bmatrix},\quad \Omega = \omega\{0,1,\ldots,r\}.

        Otherwise, include the negative frequencies as well.
    :type periodic_forcing: Optional[Union[Tuple[SLEPc.BV, np.array], None]],
        default is None
    """
    dt = (tf - t0) / (nsteps - 1)
    time = dt * np.arange(0, nsteps, 1) + t0
    time_f_eval = np.flipud(time) if adjoint else time
    dtf = time_f_eval[1] - time_f_eval[0]

    # Create array to store the solution (unless it is passed by the user)
    # When m = -1, we return only the solution at time tf (or t0, if
    # integrating backward in time).
    if X == None and m != -1:
        X = SLEPc.BV().create(comm=v.getComm())
        X.setSizes(v.getSizes(), len(time[::m]))
        X.setType("mat")

    if m != -1:
        if X.getSizes()[-1] != len(time[::m]):
            raise ValueError(
                f"The SLEPc BV X used to store the solution has the wrong "
                f"number of columns."
            )

    # Check if the user has provided an external forcing function
    if periodic_forcing == None:
        f = None

        def evaluate_dynamics(x, y, t, f=None):
            return action(x, y)
    else:
        FHat, omegas = periodic_forcing
        f = v.duplicate()

        def evaluate_dynamics(x, y, t, f):
            f = ifft(FHat, f, omegas, t)
            y = action(x, y)
            y.axpy(1.0, f)
            return y

    x = v.copy()
    if m != -1:
        x_ = X.getColumn(0)
        x.copy(x_)
        X.restoreColumn(0, x_)

    if method == "RK2":
        k1 = x.duplicate()
        k2 = x.duplicate()
        x_temp = x.duplicate()

        save_idx = 0
        for j in range(1, nsteps):
            t = time_f_eval[j - 1]

            k1 = evaluate_dynamics(x, k1, t, f)
            x.copy(x_temp)
            x_temp.axpy(dt, k1)
            k2 = evaluate_dynamics(x_temp, k2, t + dtf, f)
            x.axpy(dt / 2, k1)
            x.axpy(dt / 2, k2)

            if m != -1 and np.mod(j, m) == 0:
                save_idx += 1
                x_ = X.getColumn(save_idx)
                x.copy(x_)
                X.restoreColumn(save_idx, x_)

        vecs = [k1, k2, x_temp]
        for vec in vecs:
            vec.destroy()

    elif method == "RK3":
        k1 = x.duplicate()
        k2 = x.duplicate()
        k3 = x.duplicate()
        x_temp = x.duplicate()

        save_idx = 0
        for j in range(1, nsteps):
            t = time_f_eval[j - 1]

            k1 = evaluate_dynamics(x, k1, t, f)
            x.copy(x_temp)
            x_temp.axpy(dt / 2, k1)
            k2 = evaluate_dynamics(x_temp, k2, t + dtf / 2, f)
            x.copy(x_temp)
            x_temp.axpy(-dt, k1)
            x_temp.axpy(2 * dt, k2)
            k3 = evaluate_dynamics(x_temp, k3, t + dtf, f)
            x.axpy(dt / 6, k1)
            x.axpy(2 * dt / 3, k2)
            x.axpy(dt / 6, k3)

            # Save snapshots if requested
            if m != -1 and np.mod(j, m) == 0:
                save_idx += 1
                x_ = X.getColumn(save_idx)
                x.copy(x_)
                X.restoreColumn(save_idx, x_)

        vecs = [k1, k2, k3, x_temp]
        for vec in vecs:
            vec.destroy()

    else:
        raise ValueError(f"Integration method should be one of RK2 or RK3.")

    if adjoint and m != -1:
        Xmat = X.getMat()
        Xmat_a = Xmat.getDenseArray()
        Xmat_a[:, :] = np.fliplr(Xmat_a)
        X.restoreMat(Xmat)

    return x if m == -1 else X


def compute_post_transient_solution(
    L: "LinearOperator",
    B: "LinearOperator",
    C: "LinearOperator",
    Laction: typing.Callable,
    tsim: np.array,
    nsave: int,
    nperiods: int,
    omegas: np.array,
    x: PETSc.Vec,
    Fhat: SLEPc.BV,
    Yhat: SLEPc.BV,
    X: SLEPc.BV,
    tol: typing.Optional[float] = 1e-3,
    time_stpper: typing.Optional[str] = "RK2",
    verbose: typing.Optional[int] = 0,
):
    r"""
    Compute the post-transient (periodic steady-state) output of the
    forced linear time-invariant system

    .. math::

        \frac{d}{dt} x(t) = L\, x(t) + B\, f(t),
        \qquad y(t) = C\, x(t),

    where the forcing is periodic of period :math:`T = 2\pi / \omega`,

    .. math::

        f(t) = \sum_{k} \hat{f}_k\, e^{i \omega_k t}.

    The state-space forcing modes :math:`B \hat{f}_k` are precomputed
    once via :meth:`LinearOperator.apply_mat`, then the system is
    integrated over successive periods :math:`[0, T]` with
    :func:`solve_ivp`, using ``x`` as the initial condition at the
    start of each period and the final state of the previous period
    as the next initial condition.  After every period the
    *periodicity error*

    .. math::

        \varepsilon_k = \frac{\|C\,x(0) - C\,x(T)\|}{\|C\,x(T)\|}

    is monitored; the iteration stops once
    :math:`\varepsilon_k < \mathrm{tol}` or after ``nperiods``
    periods.  At convergence, the trajectory snapshots in ``X`` are
    projected through :math:`C` and Fourier-transformed into
    ``Yhat``.

    If ``Laction`` is :meth:`L.apply_hermitian_transpose <
    LinearOperator.apply_hermitian_transpose>` rather than
    :meth:`L.apply`, the adjoint system is integrated backward in time
    instead — :func:`solve_ivp` handles the flip internally and the
    snapshot ordering is corrected.

    :param L: state operator :math:`L`
    :type L: LinearOperator
    :param B: input operator :math:`B`
    :type B: LinearOperator
    :param C: output operator :math:`C`
    :type C: LinearOperator
    :param Laction: callable that applies :math:`L` (or its
        Hermitian transpose) to a vector.  One of ``L.apply`` or
        ``L.apply_hermitian_transpose``.
    :type Laction: Callable
    :param tsim: simulation time grid for one period (typically the
        ``tsim`` returned by
        :func:`create_time_and_frequency_arrays`)
    :type tsim: np.array
    :param nsave: save stride; ``tsim[::nsave]`` are the snapshot
        times stored in ``X``
    :type nsave: int
    :param nperiods: maximum number of periods to integrate
    :type nperiods: int
    :param omegas: angular frequencies matching the columns of
        ``Fhat``
    :type omegas: np.array
    :param x: initial state at the start of the first period; updated
        in-place at the end of each period and holds the
        periodic-steady-state initial condition on return
    :type x: PETSc.Vec
    :param Fhat: Fourier coefficients of the forcing :math:`f(t)`,
        one column per frequency in ``omegas``
    :type Fhat: SLEPc.BV
    :param Yhat: output buffer for the Fourier coefficients of
        :math:`y(t)`; filled and returned
    :type Yhat: SLEPc.BV
    :param X: snapshot buffer; column :math:`i` will hold
        :math:`x(t_i)` for :math:`t_i \in \text{tsim}[::n_{\text{save}}]`
    :type X: SLEPc.BV
    :param tol: convergence tolerance on the relative periodicity
        error :math:`\varepsilon_k`
    :type tol: Optional[float], default ``1e-3``
    :param time_stpper: time integrator passed to :func:`solve_ivp`
        (e.g. ``"RK2"`` or ``"RK3"``)
    :type time_stpper: Optional[str], default ``"RK2"``
    :param verbose: if greater than 1, prints the periodicity error
        at the end of every period
    :type verbose: Optional[int], default ``0``

    :return: the input ``Yhat`` BV, filled with the Fourier
        coefficients of the post-transient output :math:`y(t)`
    :rtype: SLEPc.BV
    """
    BFhat = B.apply_mat(Fhat)
    y0 = C.create_right_vector()
    yk = y0.duplicate()
    adjoint = False if Laction == L.apply else True
    idx = 0 if adjoint else X.getSizes()[-1] - 1
    for k in range(nperiods):
        X = solve_ivp(
            x,
            Laction,
            0.0,
            tsim[-1],
            len(tsim),
            time_stpper,
            nsave,
            adjoint,
            X,
            (BFhat, omegas),
        )
        y0 = C.apply_hermitian_transpose(x, y0)
        xk = X.getColumn(idx)
        yk = C.apply_hermitian_transpose(xk, yk)
        xk.copy(x)
        X.restoreColumn(idx, xk)
        y0.axpy(-1.0, yk)
        error = y0.norm() / yk.norm()
        if verbose > 1:
            str = (
                f"Deviation from periodicity at period {k + 1}/{nperiods} "
                f"= {error}"
            )
            petscprint(PETSc.COMM_WORLD, str)
        if error < tol:
            break

    Y = C.apply_hermitian_transpose_mat(X)
    Y.setActiveColumns(0, X.getSizes()[-1] - 1)
    Yhat = fft(Y, Yhat, L.get_real_flag())

    objects = [Y, BFhat, y0, yk, xk]
    for obj in objects:
        obj.destroy()
    return Yhat
