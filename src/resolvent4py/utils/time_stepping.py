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
) -> SLEPc.BV:
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

    X.restoreMat(Xmat)
    Xhat.restoreMat(Xhat_mat)
    return Xhat


def ifft(
    Xhat: SLEPc.BV, x: PETSc.Vec, omegas: np.array, t: float
) -> PETSc.Vec:
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
