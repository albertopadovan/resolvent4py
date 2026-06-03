import typing
import warnings
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
    tstore = np.linspace(0, T, num=2 * (n_omegas + 4), endpoint=False)
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
    L: "LinearOperator",
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
    Integrate a (possibly time-dependent) linear system of the form

    .. math::

        \frac{d}{dt}x(t) = L(t)\, x(t) + f(t),\quad x(0) = v,

    from :math:`t = t_0` to :math:`t = t_f`.
    If the flag ``adjoint`` is :code:`True`, we solve

    .. math::

        -\frac{d}{dt}x(t) = L(t)\, x(t) + f(t),\quad x(t_f) = v,\, t\in [t_0, t_f],

    backward in time from :math:`t = t_f` to :math:`t = t_0`.
    In both cases, the forcing function :math:`f(t)` is periodic and given by

    .. math::

        f(t) = f(t + T) = \sum_{k=-r}^r f_k e^{ik\omega t},\quad \omega = 2\pi/T.

    At every RK stage the time ``t`` is forwarded to the operator via
    :meth:`LinearOperator.set_evaluation_time`, so any time-dependent
    operator (e.g. one built around a
    :class:`TimePeriodicMatrixLinearOperator`) is automatically kept in
    sync with the integrator.  For time-invariant operators, that call
    is a no-op walk over child attributes.

    :param v: initial condition
    :type v: PETSc.Vec
    :param L: linear operator :math:`L(t)`.  Its forward or
        Hermitian-transpose action is selected internally based on
        ``adjoint``.
    :type L: LinearOperator
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
    action = L.apply_hermitian_transpose if adjoint else L.apply

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
            L.set_evaluation_time(t)
            return action(x, y)
    else:
        FHat, omegas = periodic_forcing
        f = v.duplicate()

        def evaluate_dynamics(x, y, t, f):
            L.set_evaluation_time(t)
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
    adjoint: bool,
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
    harmonic_balancing_ordering: typing.Optional[bool] = False,
    method: typing.Optional[str] = "donothing",
    gmres_rtol: typing.Optional[float] = 1e-8,
    gmres_max_it: typing.Optional[int] = 200,
):
    r"""
    Compute the post-transient (periodic steady-state) output of the
    forced linear system

    .. math::

        \frac{d}{dt} x(t) = L(t)\, x(t) + B(t)\, f(t),
        \qquad y(t) = C(t)\, x(t),

    where the forcing is periodic of period :math:`T = 2\pi / \omega`,

    .. math::

        f(t) = \sum_{k} \hat{f}_k\, e^{i \omega_k t}.

    Each of :math:`L(t)`, :math:`B(t)`, :math:`C(t)` may be
    time-invariant or :math:`T`-periodic.  Time updates are forwarded
    to every operator via :meth:`LinearOperator.set_evaluation_time`:
    :func:`solve_ivp` advances ``L``'s time at every RK stage; this
    routine advances ``B``'s time while pre-computing the
    state-space forcing and advances ``C``'s time when projecting
    snapshots and the periodicity-check endpoints.  For an operator
    with no internal time dependence the call is a no-op walk over
    child attributes — see
    :class:`~resolvent4py.linear_operators.TimePeriodicMatrixLinearOperator`
    for the canonical implementation that does use the time.

    The Fourier coefficients of :math:`g(t) = B(t) f(t)` are
    pre-computed once by sampling :math:`g` on the FFT save grid and
    forward-transforming.  This reproduces :math:`B \hat{f}_k`
    column-by-column when :math:`B` is time-invariant (the
    IFFT/FFT pair is exact on the matching grid) and gives the proper
    convolution coefficients otherwise.  The system is then integrated
    over successive periods :math:`[0, T]` with :func:`solve_ivp`,
    using ``x`` as the initial condition at the start of each period
    and the final state of the previous period as the next initial
    condition.  After every period the *periodicity error*

    .. math::

        \varepsilon_k = \frac{\|C(0)\,x(0) - C(0)\,x(T)\|}
                              {\|C(0)\,x(T)\|}

    is monitored (with :math:`C` clocked to :math:`t = 0`, which equals
    :math:`C(T)` by periodicity); the iteration stops once
    :math:`\varepsilon_k < \mathrm{tol}` or after ``nperiods`` periods.
    At convergence, the trajectory snapshots in ``X`` are projected
    column-wise through :math:`C(t_i)` and Fourier-transformed into
    ``Yhat``.

    If ``adjoint`` is ``True`` the adjoint system
    :math:`-\dot{x} = L^*(t)\, x + B(t)\, f(t)` is integrated backward
    in time instead — :func:`solve_ivp` handles the flip internally
    and the snapshot ordering is corrected.

    :param L: state operator :math:`L(t)`.  Time-invariant or
        :math:`T`-periodic; if time-periodic it must override
        :meth:`LinearOperator.set_evaluation_time` to update its
        internal time (composite operators inherit the propagation
        from the base class).
    :type L: LinearOperator
    :param B: input operator :math:`B(t)`.  Time-invariant or
        :math:`T`-periodic (same conventions as ``L``).
    :type B: LinearOperator
    :param C: output operator :math:`C(t)`.  Time-invariant or
        :math:`T`-periodic (same conventions as ``L``).
    :type C: LinearOperator
    :param adjoint: if ``False``, integrate the forward system using
        ``L.apply``; if ``True``, integrate the adjoint backward in
        time using ``L.apply_hermitian_transpose``.
    :type adjoint: bool
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
        error :math:`\varepsilon_k` for the ``'donothing'`` method;
        ignored when ``method='gmres'``
    :type tol: Optional[float], default ``1e-3``
    :param time_stpper: time integrator passed to :func:`solve_ivp`
        (e.g. ``"RK2"`` or ``"RK3"``)
    :type time_stpper: Optional[str], default ``"RK2"``
    :param verbose: ``0`` is silent; ``1`` prints a one-line GMRES
        convergence summary (only for ``'gmres'``); ``> 1`` also
        prints per-iteration residual (``'gmres'``) or per-period
        periodicity error (``'donothing'``)
    :type verbose: Optional[int], default ``0``
    :param harmonic_balancing_ordering: passed to the output FFT.
        ``False`` (default) gives numpy-FFT column order
        ``[0, 1, …, m, -m, …, -1]``; ``True`` gives the
        :class:`~resolvent4py.spectral_submanifold.PeriodicDifferentialEquation`
        convention ``[-m, …, -1, 0, 1, …, m]``.  Only relevant when
        ``omegas`` is two-sided.
    :type harmonic_balancing_ordering: Optional[bool], default ``False``
    :param method: how to enforce :math:`T`-periodicity of the steady
        state.

        - ``'donothing'``: integrate one period at a time, reseed
          with :math:`x(T)`, stop once
          :math:`\|C(0)(x(0) - x(T))\| / \|C(0)\, x(T)\| < \mathrm{tol}`
          or after ``nperiods`` periods.  Cheap per call but requires
          the *shifted* Floquet spectrum to lie strictly in the left
          half-plane (i.e.\ the time-stepping is stable).  Diverges
          if any direction has :math:`\mathrm{Re}(\lambda) \geq 0`.

        - ``'gmres'``: solve the linear BVP
          :math:`(I - \Phi(T, 0))\, x(0) = \int_0^T \Phi(T, \tau)\,
          (B f)(\tau)\, d\tau` via GMRES on a PETSc shell whose
          ``mult`` is one homogeneous shot of :func:`solve_ivp` per
          iteration.  ``Φ`` is the monodromy of the homogeneous
          shifted ODE.  Works for **any** shift where
          :math:`I - \Phi(T, 0)` is non-singular — including shifts
          in the unstable half-plane where ``'donothing'`` would
          diverge.  Costs one extra time integration (the snapshot
          fill after the GMRES solve), plus one integration per GMRES
          iteration.  Fails only at resonance:
          :math:`s \in \mathrm{spec}(L_{\rm HB})`.
    :type method: Optional[str], default ``'donothing'``
    :param gmres_rtol: relative residual tolerance for the GMRES
        solve; only used when ``method='gmres'``
    :type gmres_rtol: Optional[float], default ``1e-8``
    :param gmres_max_it: GMRES iteration cap; only used when
        ``method='gmres'``
    :type gmres_max_it: Optional[int], default ``200``

    :return: the input ``Yhat`` BV, filled with the Fourier
        coefficients of the post-transient output :math:`y(t)`
    :rtype: SLEPc.BV
    """
    comm = L.get_comm()
    # The FFT format is dictated by ``omegas``: a non-negative spectrum
    # implies a real time-domain signal (use rfft), a two-sided spectrum
    # implies a complex signal (use full fft).  ``harmonic_balancing_ordering``
    # additionally selects whether the two-sided output columns are
    # numpy-ordered ``[0, …, m, -m, …, -1]`` or HB-ordered
    # ``[-m, …, -1, 0, …, m]`` — the latter matches the
    # :class:`PeriodicDifferentialEquation` convention.
    real_signal = np.min(omegas) == 0.0

    # ── Pre-compute Fourier coefficients of g(t) = B(t) f(t) ───────────
    # by sampling g on the FFT save grid and forward-transforming.  This
    # path works for both time-invariant and time-periodic B: when B is
    # time-invariant it reproduces ``B.apply_mat(Fhat)`` (the IFFT/FFT
    # round-trip is exact on the matching grid); when B is time-periodic
    # it produces the correct convolution coefficients.  Time updates
    # are forwarded to B via :meth:`set_evaluation_time` — a cheap
    # attribute-walk no-op when B has no internal time dependence.
    save_times = tsim[::nsave][:-1]
    state_sizes = X.getSizes()[0]
    f_buf = Fhat.createVec()
    g_buf = PETSc.Vec().create(comm=comm)
    g_buf.setSizes(state_sizes)
    g_buf.setUp()
    g_time = SLEPc.BV().create(comm=comm)
    g_time.setSizes(state_sizes, len(save_times))
    g_time.setType("mat")
    for i, t_i in enumerate(save_times):
        f_buf = ifft(Fhat, f_buf, omegas, t_i)
        B.set_evaluation_time(t_i)
        g_buf = B.apply(f_buf, g_buf)
        g_col = g_time.getColumn(i)
        g_buf.copy(g_col)
        g_time.restoreColumn(i, g_col)
    g_time.setActiveColumns(0, len(save_times))
    BFhat = SLEPc.BV().create(comm=comm)
    BFhat.setSizes(state_sizes, len(omegas))
    BFhat.setType("mat")
    BFhat = fft(g_time, BFhat, real_signal, harmonic_balancing_ordering)
    g_time.destroy()
    f_buf.destroy()
    g_buf.destroy()

    y0 = C.create_right_vector()
    yk = y0.duplicate()

    if method == "donothing":
        # ── Post-transient iteration ────────────────────────────────────
        # Integrate one period at a time, reseed with the final state,
        # and stop once the periodicity error
        # ‖C(0)(x(0) − x(T))‖ / ‖C(0) x(T)‖ falls below ``tol``.
        # Requires the shifted system to be stable; for shifts whose
        # real part lands inside the Floquet spectrum, use
        # ``method='gmres'`` instead.
        idx = 0 if adjoint else X.getSizes()[-1] - 1
        for k in range(nperiods):
            X = solve_ivp(
                x,
                L,
                0.0,
                tsim[-1],
                len(tsim),
                time_stpper,
                nsave,
                adjoint,
                X,
                (BFhat, omegas),
            )
            # By periodicity C(t=0) == C(t=T); evaluate both endpoints at 0.
            C.set_evaluation_time(0.0)
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
        else:
            if comm.getRank() == 0:
                warnings.warn(
                    f"compute_post_transient_solution: did not converge after "
                    f"{nperiods} periods.  Final periodicity error = "
                    f"{error:.3e}, tol = {tol:.3e}.  Increase ``nperiods`` "
                    f"or relax ``tol``.",
                    UserWarning,
                    stacklevel=2,
                )

    elif method == "gmres":
        # ── Shoot-and-solve via GMRES ──────────────────────────────────
        # The T-periodic steady state satisfies
        #     (I − Φ(T, 0)) x(0) = ∫_0^T Φ(T, τ) (B f)(τ) dτ
        # where Φ is the monodromy of the homogeneous shifted ODE.
        # The RHS is one solve_ivp shot from zero IC; (I − Φ) is wrapped
        # as a PETSc shell whose mult is one homogeneous shot per
        # GMRES iteration.  Works for any shift where I − Φ is
        # invertible — including shifts in the unstable half-plane
        # where the iteration path diverges.
        x.zeroEntries()
        rhs = solve_ivp(
            x, L, 0.0, tsim[-1], len(tsim), time_stpper,
            m=-1, adjoint=adjoint, X=None,
            periodic_forcing=(BFhat, omegas),
        )

        class _IminusPhiShell:
            r"""``mult: y = x − Φ_eff(T, 0) x`` where Φ_eff is the
            forward or adjoint propagator depending on
            ``adjoint``."""
            def mult(self_, _M, x_in, y_out):
                sol = solve_ivp(
                    x_in, L, 0.0, tsim[-1], len(tsim), time_stpper,
                    m=-1, adjoint=adjoint, X=None,
                    periodic_forcing=None,
                )
                sol.aypx(-1.0, x_in)   # sol = x_in − Φ_eff x_in
                sol.copy(y_out)
                sol.destroy()

        shell = PETSc.Mat().create(comm)
        shell.setSizes(L.get_dimensions())
        shell.setType("python")
        shell.setPythonContext(_IminusPhiShell())
        shell.setUp()

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(shell)
        ksp.setType("gmres")
        ksp.getPC().setType("none")
        ksp.setTolerances(rtol=gmres_rtol, max_it=gmres_max_it)
        if verbose > 1:
            def _mon(_ksp, it, rn):
                petscprint(
                    comm,
                    f"GMRES iter {it}: |r| = {rn:.3e}",
                )
            ksp.setMonitor(_mon)
        ksp.solve(rhs, x)
        if verbose > 0:
            reason = ksp.getConvergedReason()
            iters = ksp.getIterationNumber()
            petscprint(
                comm,
                f"compute_post_transient_solution[gmres]: "
                f"{iters} iters, reason={reason}",
            )
        if comm.getRank() == 0 and ksp.getConvergedReason() <= 0:
            warnings.warn(
                f"compute_post_transient_solution[gmres]: GMRES failed "
                f"to converge (reason = {ksp.getConvergedReason()}, "
                f"{ksp.getIterationNumber()} iters).  Either I − Φ(T,0) "
                f"is singular (resonance: s on the spectrum of L_HB) "
                f"or ``gmres_max_it`` is too small.",
                UserWarning,
                stacklevel=2,
            )
        rhs.destroy()
        ksp.destroy()
        shell.destroy()

        # One more integration from the periodic IC ``x`` to fill ``X``
        # with snapshots for the FFT below.
        X = solve_ivp(
            x, L, 0.0, tsim[-1], len(tsim), time_stpper,
            nsave, adjoint, X, (BFhat, omegas),
        )

    else:
        raise ValueError(
            f"compute_post_transient_solution: unknown method "
            f"{method!r}.  Expected 'donothing' or 'gmres'."
        )

    # ── Apply C^H column-by-column to support time-varying C ───────────
    Y = SLEPc.BV().create(comm=comm)
    Y.setSizes(y0.getSizes(), X.getSizes()[-1])
    Y.setType("mat")
    snapshot_times = tsim[::nsave]
    for i, t_i in enumerate(snapshot_times):
        C.set_evaluation_time(t_i)
        x_col = X.getColumn(i)
        y_col = Y.getColumn(i)
        y_col = C.apply_hermitian_transpose(x_col, y_col)
        Y.restoreColumn(i, y_col)
        X.restoreColumn(i, x_col)
    Y.setActiveColumns(0, X.getSizes()[-1] - 1)
    Yhat = fft(Y, Yhat, real_signal, harmonic_balancing_ordering)

    objects = [Y, BFhat, y0, yk]
    for obj in objects:
        obj.destroy()
    return Yhat
