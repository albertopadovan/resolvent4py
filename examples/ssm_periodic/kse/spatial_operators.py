"""
Spatial operators for the Kuramoto-Sivashinsky equation on x in [0, 2*pi]:

    u_t = -u*u_x - u_xx - nu*u_xxxx

Ansatz:
    u(x, t) = sum_{j=1}^{n} c_j(t) * sin(j*x)

In spectral space the linear operator L is diagonal with eigenvalues

    lambda_j = j^2 - nu*j^4

The nonlinear term N(u) = -u*u_x is evaluated pseudospectrally via the rfft,
using the correspondence between the sine/cosine series and the real DFT:

    u   = sum_j  c_j * sin(j*x)   <->  rfft[j] = -i * (N/2) * c_j
    u_x = sum_j  j*c_j * cos(j*x) <->  rfft[j] =      (N/2) * j * c_j   (real)
"""

import numpy as np
from scipy.fft import rfft, irfft


def linear_eigenvalues(n: int, nu: float) -> np.ndarray:
    """
    Diagonal eigenvalues of the linear operator L in spectral space.

    L applied to sin(j*x) gives lambda_j * sin(j*x), where

        lambda_j = j^2 - nu*j^4

    arising from -u_xx - nu*u_xxxx.

    Parameters
    ----------
    n  : number of sine modes
    nu : viscosity coefficient (nu > 0)

    Returns
    -------
    lam : (n,) array,  lam[j-1] = lambda_j
    """
    j = np.arange(1, n + 1, dtype=float)
    return j**2 - nu * j**4


def nonlinear_rhs(c: np.ndarray, n_pts: int = None) -> np.ndarray:
    """
    Evaluate N(u) = -u * u_x pseudospectrally, returning sine coefficients.

    Steps:
      1. Build rfft spectra for u and u_x from the sine coefficients c.
      2. Transform both to physical space (irfft).
      3. Multiply pointwise: N_phys = -u * u_x.
      4. Transform back (rfft) and extract sine coefficients.

    The physical grid has n_pts >= 4*n points to eliminate aliasing errors
    from the quadratic product (which generates modes up to 2*n).

    rfft convention used (scipy.fft, N = n_pts):
        rfft(sin(j*x_m))[j]   = -i * (N/2)           (purely imaginary)
        rfft(cos(j*x_m))[j]   =      (N/2)            (real)
    where x_m = 2*pi*m/N,  m = 0, ..., N-1.

    Parameters
    ----------
    c     : (n,) real array of current sine coefficients
    n_pts : physical-space grid size (default: 4*n)

    Returns
    -------
    N_c : (n,) real array of sine coefficients of -u*u_x
    """
    n = len(c)
    if n_pts is None:
        n_pts = 4 * n  # quadratic product needs >= 4n pts

    j = np.arange(1, n + 1, dtype=float)
    half_N = n_pts / 2.0

    # rfft spectrum of u = sum c_j sin(j*x):   rfft[j] = -i*(N/2)*c_j
    u_rfft = np.zeros(n_pts // 2 + 1, dtype=complex)
    u_rfft[1 : n + 1] = -1j * half_N * c

    # rfft spectrum of u_x = sum j*c_j cos(j*x):  rfft[j] = (N/2)*j*c_j  (real)
    ux_rfft = np.zeros(n_pts // 2 + 1, dtype=complex)
    ux_rfft[1 : n + 1] = half_N * j * c

    # Physical-space values
    u_phys = irfft(u_rfft, n=n_pts)  # real: sum c_j sin(j * 2pi*m/N)
    ux_phys = irfft(ux_rfft, n=n_pts)  # real: sum j*c_j cos(j * 2pi*m/N)

    # Pointwise nonlinear product
    N_phys = -u_phys * ux_phys  # odd function -> sine series

    # Project back: N_rfft[j] = -i*(N/2)*d_j  =>  d_j = Re(2i * N_rfft[j] / N)
    N_rfft = rfft(N_phys)
    return np.real(2j * N_rfft[1 : n + 1] / n_pts)
