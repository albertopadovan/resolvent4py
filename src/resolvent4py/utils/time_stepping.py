import typing
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from ..utils.vector import vec_real


def ifft(
    Xhat: SLEPc.BV,
    x: PETSc.Vec,
    omegas: np.array,
    t: float,
    adjoint: typing.Optional[bool] = False,
) -> PETSc.Vec:
    sign = -1 if adjoint else 1
    q = np.exp(1j * omegas * t * sign)
    if np.min(omegas) == 0.0:
        c = 2 * np.ones(len(q))
        c[0] = 1.0
        q *= c
        Xhat.multVec(1.0, 0.0, x, q)
        x = vec_real(x, True)
    else:
        Xhat.multVec(1.0, 0.0, x, q)
    return x


def solve_ivp(
    x0: PETSc.Vec,
    action: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    t0: float,
    tf: float,
    nsteps: int,
    method: typing.Optional[str] = "RK2",
    m: typing.Optional[int] = -1,
    adjoint: typing.Optional[bool] = False,
    X: typing.Optional[SLEPc.BV] = None,
    periodic_forcing: typing.Optional[typing.Tuple[SLEPc.BV, np.array]] = None,
):
    r"""
    Integrate a linear (time-invariant) system of equations of the form

    .. math::

        \frac{d}{dt}x(t) = A x(t) (+ f(t)),\quad x(0) = x_0.

    The external forcing :math:`f(t)` is optional.

    :param x0: initial condition
    :type x0: PETSc.Vec
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
    :param adjoint: flag to indicate whether we are integrating an adjoint system.
        This has no effect if the dynamics are unforced. 
    :type adjoint: Optional[bool], default is False
    :param X: structure to store the solution. Useful only if :math:`m > 0`.
    :type X: Optional[Union[SLEPc.BV, None]], default is None
    :param periodic_forcing: Fourier modes Fhat of the forcing function 
        :math:`f(t)`, and array of frequencies corresponding to those modes.
    :type periodic_forcing: Optional[Union[Tuple[SLEPc.BV, np.array], None]],
        default is None
    """
    dt = (tf - t0) / (nsteps - 1)
    time = dt * np.arange(0, nsteps, 1) + t0

    # Create array to store the solution (unless it is passed by the user)
    # and if m != -1. When m = -1, we return only the solution at time tf
    if X == None and m != -1:
        X = SLEPc.BV().create(comm=x0.getComm())
        X.setSizes(x0.getSizes(), len(time[::m]))
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
        f = x0.duplicate()

        def evaluate_dynamics(x, y, t, f):
            f = ifft(FHat, f, omegas, t, adjoint)
            y = action(x, y)
            y.axpy(1.0, f)
            return y

    x = x0.copy()
    if m != -1:
        # Store the first snapshot
        x_ = X.getColumn(0)
        x.copy(x_)
        X.restoreColumn(0, x_)

    if method == "RK2":
        k1 = x.duplicate()
        k2 = x.duplicate()
        x_temp = x.duplicate()

        save_idx = 0
        for j in range(1, nsteps):
            k1 = evaluate_dynamics(x, k1, time[j - 1], f)
            x.copy(x_temp)
            x_temp.axpy(dt, k1)
            k2 = evaluate_dynamics(x_temp, k2, time[j - 1] + dt, f)
            x.axpy(dt / 2, k1)
            x.axpy(dt / 2, k2)
        
            if m != -1 and np.mod(j, m) == 0:
                save_idx += 1
                # Store the first snapshot
                x_ = X.getColumn(save_idx)
                x.copy(x_)
                X.restoreColumn(save_idx, x_)

        vecs = [k1, k2, x_temp]
        for vec in vecs:
            vec.destroy()

    return x if m == -1 else X
