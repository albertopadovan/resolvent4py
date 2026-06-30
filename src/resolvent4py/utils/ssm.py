from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only imported for the type annotations below — avoids dragging in
    # matplotlib at module-import time.
    import matplotlib.pyplot as plt


def plot_convergence_radius(
    orders: np.ndarray,
    coeff_sums: np.ndarray,
    slope: float,
    intercept: float,
    R_estimate: float,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Axes":
    r"""
    Plot SSM coefficient decay vs polynomial order with fitted line.

    :param orders: polynomial orders
    :type orders: np.ndarray
    :param coeff_sums: :math:`C_k` values for each order
    :type coeff_sums: np.ndarray
    :param slope: slope of the log-linear fit
    :type slope: float
    :param intercept: intercept of the log-linear fit
    :type intercept: float
    :param R_estimate: estimated convergence radius
    :type R_estimate: float
    :param ax: existing axes (creates new figure if None)
    :type ax: Optional[matplotlib.axes.Axes]

    :return: axes with the plot
    :rtype: matplotlib.axes.Axes
    """
    # Imported lazily so that `import resolvent4py` does not drag in
    # matplotlib (and its import-time warnings) unless plotting is used.
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    valid = coeff_sums > 0
    log_sums = np.full_like(coeff_sums, np.nan)
    log_sums[valid] = np.log10(coeff_sums[valid])

    ax.plot(orders, log_sums, "o", label="Coefficient sums")
    ax.plot(
        orders,
        slope * orders + intercept,
        "r--",
        label=rf"Fit: slope={slope:.3f}, $R\approx${R_estimate:.3f}",
    )
    ax.set_xlabel("Polynomial order $k$")
    ax.set_ylabel(r"$\log_{10}(\sum_{|j|=k} \|p_j\|_1)$")
    ax.set_title("SSM coefficient decay")
    ax.legend()
    plt.tight_layout()
    return ax


def proper_radius(
    manifold_tol: float,
    intercept: float,
    m: int,
) -> Tuple[float, float]:
    r"""
    Compute the fraction :math:`p` of the convergence radius at which
    the truncation error of the SSM polynomial is below a tolerance.

    The truncation error is modelled as

    .. math::

        \varepsilon \approx 10^{\text{intercept}}\,
        \frac{p^{m+1}}{1 - p},

    and we solve for :math:`p` via Newton's method.

    :param manifold_tol: desired truncation error bound
    :type manifold_tol: float
    :param intercept: intercept from the log-linear fit
    :type intercept: float
    :param m: polynomial expansion order
    :type m: int

    :return: ``(p, est_error)`` — fraction of the convergence
        radius and estimated truncation error at that fraction
    :rtype: Tuple[float, float]
    """
    eps = manifold_tol / 10**intercept
    p = min(eps ** (1.0 / (m + 1)), 1.0)

    for _ in range(10):
        fp = p ** (m + 1) - eps * (1 - p)
        fpp = (m + 1) * p**m + eps
        step = fp / fpp
        p -= step
        if abs(step) < 1e-12 * (1 + abs(p)):
            break

    assert 0 < p <= 1.0
    est_error = 10**intercept * p ** (m + 1) / (1 - p)
    return p, est_error


def evaluate_dP(
    s: np.ndarray,
    multiindices: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the Jacobian :math:`\partial P / \partial s` of a polynomial
    SSM map at the latent point :math:`s`.

    The manifold map has the polynomial form

    .. math::

        P(s) \;=\; \sum_{j} p_j \, s^{j},
        \qquad s^j \;=\; \prod_{k=1}^{r} s_k^{j_k},

    so its Jacobian column for direction :math:`k = 0, \dots, r-1` is

    .. math::

        \frac{\partial P}{\partial s_k}(s)
        \;=\; \sum_{j: j_k > 0}\, j_k\, p_j\, s^{j - e_k}.

    The function is fully numpy and works with arbitrary trailing shape on
    ``ps`` — it sums coefficients along the first axis, so ``ps`` can be
    ``(n_terms, n)``, ``(n_terms, n_harmonics, n)``, etc.

    :param s: latent coordinates, shape ``(r,)``
    :type s: np.ndarray
    :param multiindices: polynomial multi-indices, shape ``(n_terms, r)``
    :type multiindices: np.ndarray
    :param ps: polynomial coefficients, shape ``(n_terms, *trailing)``
    :type ps: np.ndarray

    :return: Jacobian columns, shape ``(r, *trailing)`` — column ``k`` is
        :math:`\partial P / \partial s_k(s)`.
    :rtype: np.ndarray
    """
    s = np.asarray(s)
    mi = np.asarray(multiindices, dtype=int)
    ps_arr = np.asarray(ps)
    r = mi.shape[1]
    n_terms = mi.shape[0]

    if s.shape != (r,):
        raise ValueError(f"s must have shape ({r},); got {s.shape}.")
    if ps_arr.shape[0] != n_terms:
        raise ValueError(
            f"ps.shape[0] = {ps_arr.shape[0]} must match "
            f"multiindices.shape[0] = {n_terms}."
        )

    out_shape = (r,) + ps_arr.shape[1:]
    dP = np.zeros(out_shape, dtype=complex)

    # Broadcasting shape for the per-term scalar coefficient
    bcast = (n_terms,) + (1,) * (ps_arr.ndim - 1)

    for k in range(r):
        j_k = mi[:, k]                       # (n_terms,)  exponent on s_k
        mask = j_k > 0
        if not np.any(mask):
            continue
        mi_shifted = mi.copy()
        mi_shifted[:, k] -= 1                # multiindex for s^{j - e_k}
        # Monomial s^{j - e_k}; for masked-out terms set to 0 contribution.
        # We compute on the full set since np.power handles 0**(-1) only
        # when mask filters it out.
        mono = np.zeros(n_terms, dtype=complex)
        mono[mask] = np.prod(s[None, :] ** mi_shifted[mask], axis=1)
        coef = (j_k * mono).reshape(bcast)   # (n_terms, 1, 1, ...)
        dP[k] = (coef * ps_arr).sum(axis=0)

    return dP
