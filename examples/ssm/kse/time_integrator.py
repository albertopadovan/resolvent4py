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


def integrate(
    c0: np.ndarray,
    lam: np.ndarray,
    nonlinear_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
    save_every: int = 1,
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
        c = imex_step(c, lam, nonlinear_fn, dt)
        if step % save_every == 0 and save_idx < n_saves:
            C[:, save_idx] = c
            t_arr[save_idx] = step * dt
            save_idx += 1

    return t_arr[:save_idx], C[:, :save_idx]
