"""
IMEX-CN/RK2 time integrator for systems of the form

    dc/dt = L*c + N(c)

where L is a diagonal linear operator (stiff) and N is a nonlinear term.

Scheme (second-order, one step):

  Predictor  — Crank-Nicolson linear + explicit Euler nonlinear:

    (I - dt/2 * L) * c_pred = (I + dt/2 * L) * c^n  +  dt * N(c^n)

  Corrector  — Crank-Nicolson linear + Heun (RK2) average nonlinear:

    (I - dt/2 * L) * c^{n+1} = (I + dt/2 * L) * c^n
                                + dt/2 * (N(c^n) + N(c_pred))

Because L is diagonal, both linear solves are elementwise divisions.
"""

import numpy as np
from typing import Callable


def imex_step(
    c: np.ndarray,
    lam: np.ndarray,
    nonlinear_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    Advance the state c by one IMEX-CN/RK2 time step.

    Parameters
    ----------
    c            : (n,) current spectral coefficients
    lam          : (n,) diagonal of the linear operator L
    nonlinear_fn : callable  c -> N(c),  returns (n,) array
    dt           : time step size

    Returns
    -------
    c_new : (n,) updated spectral coefficients
    """
    lhs = 1.0 - 0.5 * dt * lam  # diagonal of (I - dt/2 * L)
    rhs_lin = (1.0 + 0.5 * dt * lam) * c  # (I + dt/2 * L) * c^n

    N0 = nonlinear_fn(c)

    # Predictor: CN linear + explicit Euler nonlinear
    c_pred = (rhs_lin + dt * N0) / lhs

    # Corrector: CN linear + Heun average of nonlinear evaluations
    N1 = nonlinear_fn(c_pred)
    return (rhs_lin + 0.5 * dt * (N0 + N1)) / lhs


def imex_step_3(
    c: np.ndarray,
    lam: np.ndarray,
    nonlinear_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    r"""
    Third-order IMEX-SSP3(3,3,2) step (Pareschi–Russo).

    For ``dc/dt = L c + N(c)`` with diagonal ``L = diag(lam)``,
    advance one step using the three-stage scheme

    .. math::

        (I - \gamma\,\Delta t\,L)\,Y_1 &= c^n \\
        (I - \gamma\,\Delta t\,L)\,Y_2 &= c^n + \Delta t\,N(Y_1)
                                          + \Delta t (1-2\gamma)\,L Y_1 \\
        (I - \gamma\,\Delta t\,L)\,Y_3 &= c^n
            + \tfrac{\Delta t}{4}\bigl[N(Y_1) + N(Y_2)\bigr]
            + \Delta t (\tfrac{1}{2} - \gamma)\,L Y_1 \\
        c^{n+1} &= c^n + \tfrac{\Delta t}{6}\bigl[N(Y_1) + N(Y_2)\bigr]
                       + \tfrac{2\Delta t}{3}\,N(Y_3) \\
                &\quad + \tfrac{\Delta t}{6}\,L\,(Y_1 + Y_2)
                       + \tfrac{2\Delta t}{3}\,L\,Y_3

    with :math:`\gamma = (3 + \sqrt{3})/6 \approx 0.7887`.  The DIRK on
    the linear side is L-stable, so stiff modes are damped
    unconditionally; the explicit RK on the nonlinear side is
    third-order accurate (asymptotic order 2 only in the deep stiff
    limit, but that's the regime where the linear part dominates and
    L-stability already handles it).

    Cost: 3 ``nonlinear_fn`` evaluations and 3 element-wise solves
    per step (vs. 2 and 2 for :func:`imex_step`).

    Parameters
    ----------
    c            : (n,) current spectral coefficients
    lam          : (n,) diagonal of the linear operator L
    nonlinear_fn : callable ``c -> N(c)``
    dt           : time step size

    Returns
    -------
    c_new : (n,) updated spectral coefficients
    """
    gamma = (3.0 + np.sqrt(3.0)) / 6.0
    one_minus_2g = 1.0 - 2.0 * gamma
    half_minus_g = 0.5 - gamma

    lhs = 1.0 - gamma * dt * lam   # (I - γ dt L), elementwise

    # Stage 1
    Y1 = c / lhs
    N1 = nonlinear_fn(Y1)

    # Stage 2
    Y2 = (c + dt * N1 + dt * one_minus_2g * lam * Y1) / lhs
    N2 = nonlinear_fn(Y2)

    # Stage 3
    Y3 = (
        c
        + 0.25 * dt * (N1 + N2)
        + dt * half_minus_g * lam * Y1
    ) / lhs
    N3 = nonlinear_fn(Y3)

    # Final update — b = b̃ = (1/6, 1/6, 2/3) so we can fold the
    # linear and nonlinear combinations into one expression.
    Y_comb = (Y1 + Y2) / 6.0 + (2.0 / 3.0) * Y3
    N_comb = (N1 + N2) / 6.0 + (2.0 / 3.0) * N3
    return c + dt * (lam * Y_comb + N_comb)


def integrate(
    c0: np.ndarray,
    lam: np.ndarray,
    nonlinear_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
    save_every: int = 1,
    stepper: Callable = imex_step,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate dc/dt = L*c + N(c) from c0 for n_steps time steps.

    Parameters
    ----------
    c0           : (n,) initial spectral coefficients
    lam          : (n,) diagonal of L
    nonlinear_fn : callable  c -> N(c)
    dt           : time step size
    n_steps      : total number of time steps to take
    save_every   : store a snapshot every this many steps (default: every step)
    stepper      : single-step IMEX routine with signature
                   ``stepper(c, lam, nonlinear_fn, dt) -> c_new``.
                   Defaults to :func:`imex_step` (2nd order); pass
                   :func:`imex_step_3` for the 3rd-order SSP3 scheme.

    Returns
    -------
    t : (n_saves,) saved times, starting at 0
    C : (n, n_saves) coefficient snapshots, C[:, 0] == c0
    """
    c = c0.copy()
    n_saves = n_steps // save_every + 1
    C = np.zeros((len(c0), n_saves))
    t_arr = np.zeros(n_saves)

    C[:, 0] = c
    save_idx = 1

    for step in range(1, n_steps + 1):
        c = stepper(c, lam, nonlinear_fn, dt)
        if step % save_every == 0 and save_idx < n_saves:
            C[:, save_idx] = c
            t_arr[save_idx] = step * dt
            save_idx += 1

    return t_arr[:save_idx], C[:, :save_idx]
