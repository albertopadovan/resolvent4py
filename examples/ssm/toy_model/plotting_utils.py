"""Plotting utilities for spectral submanifold examples."""

import numpy as np
from typing import Optional, List, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import cm

import resolvent4py as res4py

if TYPE_CHECKING:
    from resolvent4py.spectral_submanifold import SpectralSubmanifold


def _decode_to_numpy(ssm: "SpectralSubmanifold", s) -> np.ndarray:
    """Decode latent-space coordinates to a real numpy array."""
    vec = ssm.decode(np.asarray(s))
    vec_seq = res4py.distributed_to_sequential_vector(vec)
    result = vec_seq.getArray().real.copy()
    vec_seq.destroy()
    vec.destroy()
    return result


# ---- Manifold surface -------------------------------------------------------


def plot_manifold_3d(
    ssm: "SpectralSubmanifold",
    rho_max: float = 0.4,
    n_rho: int = 50,
    n_theta: int = 50,
    surface_alpha: float = 0.3,
    surface_cmap=None,
    ax=None,
):
    r"""
    Plot the 2D SSM as a surface in 3D (first 3 state components).
    Only supported for :math:`r = 2`.

    :param ssm: solved SSM instance
    :param rho_max: max amplitude in the polar grid
    :param n_rho: radial grid points
    :param n_theta: angular grid points
    :param surface_alpha: surface transparency
    :param surface_cmap: colormap (default: ``cm.plasma``)
    :param ax: existing 3D axes

    :return: 3D axes
    """
    if ssm.ps is None:
        raise RuntimeError("Call solve() first.")
    if ssm.r != 2:
        raise ValueError(f"Requires r=2, got r={ssm.r}.")

    rho = np.linspace(0, rho_max, n_rho)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    s1 = np.outer(np.exp(1j * theta), rho)

    X = np.zeros((n_theta, n_rho))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i in range(n_theta):
        for j in range(n_rho):
            v = _decode_to_numpy(ssm, (s1[i, j], s1[i, j].conj()))
            X[i, j], Y[i, j], Z[i, j] = v[0], v[1], v[2]

    ax = None
    if ssm.diff_eq.get_comm().getRank() == 0:
        if ax is None:
            ax = plt.figure().add_subplot(111, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=surface_cmap or cm.plasma,
            linewidth=0,
            antialiased=False,
            alpha=surface_alpha,
        )
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$x_3$")
    return ax


def plot_rom_vs_truth_3d(
    Q_rom: np.ndarray,
    Q_truth: np.ndarray,
    ssm: Optional["SpectralSubmanifold"] = None,
    rho_max: float = 0.4,
    plot_manifold: bool = True,
    ax=None,
    **manifold_kwargs,
):
    r"""
    Overlay ROM and truth trajectories in 3D (first 3 components),
    optionally on top of the SSM surface.

    :param Q_rom: ROM trajectory (n, n_t)
    :param Q_truth: truth trajectory (n, n_t)
    :param ssm: solved SSM (required if plot_manifold is True)
    :param rho_max: passed to :func:`plot_manifold_3d`
    :param plot_manifold: render SSM surface
    :param ax: existing 3D axes
    :param manifold_kwargs: forwarded to :func:`plot_manifold_3d`

    :return: 3D axes
    """
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    if plot_manifold:
        if ssm is None:
            raise ValueError("ssm required when plot_manifold=True.")
        plot_manifold_3d(ssm, rho_max=rho_max, ax=ax, **manifold_kwargs)

    ax.plot3D(*Q_rom[:3], color="k", ls="--", lw=2, label="ROM")
    ax.plot3D(*Q_truth[:3], color="r", lw=2, label="Truth")
    ax.legend()
    return ax
