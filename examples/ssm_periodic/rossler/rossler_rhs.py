"""
Pure-numpy RHS helpers for the Rössler system from
Padovan & Rowley, PRF (2022), Section IV:

    ẋ = -y - z
    ẏ =  x + 0.1 y
    ż =  0.1 + z (x - c)

Split into linear matrix M, symmetric bilinear form B, and constant
drive b0:

    q̇ = M q + B(q, q) + b0,
    M  = [[0,  -1,  -1 ],
          [1,   0.1, 0 ],
          [0,   0,  -c ]],
    B(q1, q2) = (0, 0, ½(x1 z2 + x2 z1)),
    b0 = (0, 0, 0.1).

Linearising about a T-periodic orbit q*(t) gives the perturbation
dynamics

    v̇ = A(t) v + B(v, v),    A(t) = M + 2 B(q*(t), ·),

which is what the SSM machinery consumes.
"""

import numpy as np


def linear_matrix(c: float) -> np.ndarray:
    return np.array([
        [0.0, -1.0, -1.0],
        [1.0,  0.1,  0.0],
        [0.0,  0.0,  -c ],
    ])


def constant_drive() -> np.ndarray:
    return np.array([0.0, 0.0, 0.1])


def quadratic_bilinear(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Symmetric bilinear form: B(q, q) = (0, 0, x z)."""
    x1, _, z1 = q1
    x2, _, z2 = q2
    return np.array([0.0, 0.0, 0.5 * (x1 * z2 + x2 * z1)])


def rossler_rhs(t: float, q: np.ndarray, c: float) -> np.ndarray:
    """Full Rössler vector field, signature compatible with solve_ivp."""
    return linear_matrix(c) @ q + quadratic_bilinear(q, q) + constant_drive()


def perturbation_linear_action(
    c_star_t: np.ndarray, v: np.ndarray, c: float,
) -> np.ndarray:
    """A(t) v = M v + 2 B(q*(t), v) — single time instant."""
    return linear_matrix(c) @ v + 2.0 * quadratic_bilinear(c_star_t, v)
