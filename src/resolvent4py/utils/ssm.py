import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_radius(
    orders: np.ndarray,
    coeff_sums: np.ndarray,
    slope: float,
    intercept: float,
    R_estimate: float,
    ax=None,
):
    r"""
    Plot coefficient decay vs polynomial order with fitted line.

    :param orders: 1-D array of polynomial orders
    :param coeff_sums: :math:`C_k` values for each order
    :param slope: slope of the log-linear fit
    :param intercept: intercept of the log-linear fit
    :param R_estimate: estimated convergence radius
    :param ax: existing axes; a new figure is created when ``None``

    :returns: the axes
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    valid = coeff_sums > 0
    log_sums = np.full_like(coeff_sums, np.nan)
    log_sums[valid] = np.log10(coeff_sums[valid])

    ax.plot(orders, log_sums, "o", label="Coefficient sums")
    fit_line = slope * orders + intercept
    ax.plot(
        orders,
        fit_line,
        "r--",
        label=rf"Fit: slope={slope:.3f}, $R\approx${R_estimate:.3f}",
    )
    ax.set_xlabel("Polynomial order $k$")
    ax.set_ylabel(r"$\log_{10}(\sum_{|j|=k} \|p_j\|_1)$")
    ax.set_title("SSM coefficient decay")
    ax.legend()
    plt.tight_layout()
    return ax