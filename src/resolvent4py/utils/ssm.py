from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence_radius(
    orders: np.ndarray,
    coeff_sums: np.ndarray,
    slope: float,
    intercept: float,
    R_estimate: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
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
