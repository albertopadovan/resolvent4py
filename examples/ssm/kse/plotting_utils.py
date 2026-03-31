import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Tuple, TYPE_CHECKING

import resolvent4py as res4py
from slepc4py import SLEPc

if TYPE_CHECKING:
    from resolvent4py.spectral_submanifold import SpectralSubmanifold


def _decode_to_numpy(ssm: "SpectralSubmanifold", s) -> np.ndarray:
    """Decode latent-space coordinates to a numpy array (complex)."""
    vec = ssm.decode(np.asarray(s))
    vec_seq = res4py.distributed_to_sequential_vector(vec)
    result = vec_seq.getArray().copy()
    vec_seq.destroy()
    vec.destroy()
    return result


def _eigenspace_basis_3d(
    W: SLEPc.BV, Lams: np.ndarray, tol: float = 1e-8,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Build a 3-column real projection basis from the most unstable
    left eigenvectors stored in the BV ``W``.

    For a complex conjugate pair, :math:`\mathrm{Re}(w)` and
    :math:`\mathrm{Im}(w)` span the 2D real eigenspace. A real
    eigenvalue contributes its (real) eigenvector.

    :param W: left eigenvectors (SLEPc BV), sorted by descending
        real part of eigenvalue
    :type W: SLEPc.BV
    :param Lams: corresponding eigenvalues
    :type Lams: np.ndarray
    :param tol: threshold for treating an eigenvalue as real
    :type tol: float

    :return: ``(B, labels)`` where ``B`` has shape ``(n, 3)``
        with unit-norm columns
    :rtype: Tuple[np.ndarray, List[str]]
    """
    directions: List[np.ndarray] = []
    labels: List[str] = []
    consumed: set = set()
    n_eigs = len(Lams)

    for i in range(n_eigs):
        if len(directions) >= 3:
            break
        if i in consumed:
            continue

        consumed.add(i)
        wi = W.getColumn(i)
        wi_seq = res4py.distributed_to_sequential_vector(wi)
        w_np = wi_seq.getArray().copy()
        wi_seq.destroy()
        W.restoreColumn(i, wi)

        if abs(Lams[i].imag) < tol:
            directions.append(w_np.real)
            labels.append(rf"$\mathrm{{Re}}(w_{{{i+1}}})$")
        else:
            # Mark conjugate partner
            for j in range(i + 1, n_eigs):
                if j not in consumed and abs(Lams[j] - np.conj(Lams[i])) < tol:
                    consumed.add(j)
                    break
            if len(directions) + 2 <= 3:
                directions.append(w_np.real)
                directions.append(w_np.imag)
                labels.append(rf"$\mathrm{{Re}}(w_{{{i+1}}})$")
                labels.append(rf"$\mathrm{{Im}}(w_{{{i+1}}})$")
            else:
                directions.append(w_np.real)
                labels.append(rf"$\mathrm{{Re}}(w_{{{i+1}}})$")

    assert len(directions) >= 3, (
        "Could not collect 3 independent projection directions."
    )
    B = np.column_stack([d / np.linalg.norm(d) for d in directions[:3]])
    return B, labels[:3]


def project_to_3d(
    B: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    r"""
    Project a complex vector onto the real 3-column basis ``B``.

    :param B: real basis of shape ``(n, 3)``
    :type B: np.ndarray
    :param v: vector of shape ``(n,)``
    :type v: np.ndarray

    :return: 3D coordinates (real)
    :rtype: np.ndarray
    """
    return (B.T @ v).real


def plot_manifold_3d(
    W: SLEPc.BV,
    Lams: np.ndarray,
    ssm: "SpectralSubmanifold",
    rho_max: float = 0.4,
    n_rho: int = 50,
    n_theta: int = 50,
    surface_alpha: float = 0.3,
    surface_cmap=None,
    ax=None,
) -> Tuple[Optional[plt.Axes], np.ndarray, List[str]]:
    r"""
    Plot the 2D SSM as a surface in 3D, projected onto a real
    eigenspace basis built from the left eigenvectors.
    Only supported for :math:`r = 2`.

    :param W: left eigenvectors (full set, not just r)
    :type W: SLEPc.BV
    :param Lams: eigenvalues corresponding to ``W``
    :type Lams: np.ndarray
    :param ssm: solved SSM instance
    :param rho_max: max amplitude in the polar grid
    :param n_rho: radial grid points
    :param n_theta: angular grid points
    :param surface_alpha: surface transparency
    :param surface_cmap: colormap (default: ``cm.plasma``)
    :param ax: existing 3D axes

    :return: ``(ax, B, labels)`` — 3D axes (None on non-root ranks),
        projection basis, and axis labels
    :rtype: Tuple[Optional[matplotlib.axes.Axes], np.ndarray, List[str]]
    """
    if ssm.ps is None:
        raise RuntimeError("Call solve() first.")
    if ssm.r != 2:
        raise ValueError(f"Requires r=2, got r={ssm.r}.")

    B, labels = _eigenspace_basis_3d(W, Lams)

    rho = np.linspace(0, rho_max, n_rho)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    s1 = np.outer(np.exp(1j * theta), rho)

    X = np.zeros((n_theta, n_rho))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i in range(n_theta):
        for j in range(n_rho):
            v = _decode_to_numpy(ssm, (s1[i, j], s1[i, j].conj()))
            coords = project_to_3d(B, v)
            X[i, j], Y[i, j], Z[i, j] = coords

    ax = None
    if ssm.diff_eq.get_comm().getRank() == 0:
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.plot_surface(
            X, Y, Z,
            rstride=1, cstride=1,
            cmap=surface_cmap or cm.plasma,
            linewidth=0, antialiased=False, alpha=surface_alpha,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    return ax, B, labels
