r"""
Numpy-only ROM for :class:`SpectralSubmanifoldPeriodicG` — the
time-periodic-:math:`g_j(t)` variant of the SSM reduction.

Distinction from :class:`SpectralSubmanifoldROM`
------------------------------------------------

The intrinsic dynamics coefficients ``gs`` are stored as
``(n_terms, n_harm, r)`` complex arrays: for each multi-index
:math:`j`, we now carry the full Fourier expansion of :math:`g_j(t)`,
not just a length-:math:`r` scalar.  Everything else — the
polynomial map ``PS``, the encoder ``W``, and the encode/decode
API — is identical.

Only :meth:`latent_space_dynamics` differs materially:

.. math::

    \dot s \;=\; \sum_j g_j(t)\, s^j,
    \qquad
    g_j(t) \;=\; \sum_k g_{j,\,k}\, e^{\, i\, k\, \omega_{HB}\, t}.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline


class SpectralSubmanifoldROMPeriodicG:
    r"""
    Numpy-only ROM for the periodic-:math:`g_j(t)` SSM.

    :param multiindices: ``(n_terms, r)`` multi-indices of the SSM
        polynomial (:math:`|j| \le m`).
    :param Lams: master eigenvalues, ``(r,)`` complex.
    :param gs: **periodic** intrinsic-dynamics coefficients,
        ``(n_terms, n_harm, r)`` complex.  For each multi-index
        row ``j``, ``gs[j, k, m]`` is the :math:`k`-th HB harmonic
        of the :math:`m`-th component of :math:`g_j(t)`.
    :param PS: manifold polynomial coefficients, ``(n_terms, n_harm,
        n)`` complex.
    :param W: master left eigenvectors in HB form, ``(n_harm, n, r)``
        complex.
    :param omega: fundamental HB angular frequency :math:`\omega_{HB}
        = 2\pi / T_{HB}`.
    """

    def __init__(
        self,
        multiindices: np.ndarray,
        Lams: np.ndarray,
        gs: np.ndarray,
        PS: np.ndarray,
        W: np.ndarray,
        omega: float,
    ):
        self.multiindices = np.asarray(multiindices, dtype=np.int64)
        self.Lams = np.asarray(Lams, dtype=np.complex128)
        self.gs = np.asarray(gs, dtype=np.complex128)                # (n_terms, n_harm, r)
        self.PS = np.asarray(PS, dtype=np.complex128)                # (n_terms, n_harm, n)
        self.W = np.asarray(W, dtype=np.complex128)                  # (n_harm, n, r)
        self.omega = float(omega)

        self.n_terms = self.multiindices.shape[0]
        self.r = self.multiindices.shape[1]
        self.n_harm = self.PS.shape[1]                          # for P_j(t)
        self.n_harm_g = self.gs.shape[1]                         # for g_j(t)
        self.n = self.PS.shape[2]
        self.nf = (self.n_harm - 1) // 2
        self.nf_g = (self.n_harm_g - 1) // 2
        assert self.n_harm_g <= self.n_harm, (
            f"gs harmonics ({self.n_harm_g}) exceed PS harmonics "
            f"({self.n_harm})"
        )

        # Consistency checks
        assert self.gs.shape == (self.n_terms, self.n_harm_g, self.r), (
            f"gs shape {self.gs.shape} != "
            f"{(self.n_terms, self.n_harm_g, self.r)}"
        )
        assert self.W.shape == (self.n_harm, self.n, self.r)

        # HB indices for the two representations
        self._k_idx = np.arange(-self.nf, self.nf + 1)           # decode/encode
        self._k_idx_g = np.arange(-self.nf_g, self.nf_g + 1)     # g_j(t)

        # Period of g(t)
        self._T = 2.0 * np.pi / self.omega

        # Pre-built g_j(t) interpolator (built lazily on first call to
        # latent_space_dynamics — solve_ivp calls the RHS thousands of
        # times, and doing an einsum over (n_harm, n_terms) each time is
        # measurably slow.  A cubic spline over a fine grid is ~40x
        # faster per call).
        self._g_spline_real: Optional[CubicSpline] = None
        self._g_spline_imag: Optional[CubicSpline] = None
        # Set to False to bypass the spline and evaluate g_j(t) via
        # exact HB IFFT at every call.  Bit-perfect but much slower.
        # Useful for pinning down whether spline undersampling is
        # producing artefacts in reconstructions.
        self.use_g_spline: bool = False

    def _build_g_spline(self) -> None:
        r"""
        Precompute :math:`g_j(t)` for every multi-index on a fine
        time grid, then wrap in a periodic :class:`CubicSpline` so
        subsequent evaluations at arbitrary :math:`t` are O(log n).
        Called lazily on the first RHS evaluation.
        """
        # 4x oversample per HB harmonic — plenty for a cubic spline
        # to reach round-off accuracy on band-limited data.
        # Reconstruction uses +1j weights consistent with the shift-DOWN
        # convention in the solve (w_k^H L v_k = Λ + iωk).  With
        # _k_idx_g = (-nf_g, ..., 0, ..., +nf_g), this pairs each stored
        # coefficient at strip index nf_g + k with the mode exp(+ikωt).
        n_samples = max(4 * self.n_harm_g, 512)
        t_grid = np.linspace(0.0, self._T, n_samples, endpoint=False)
        weights = np.exp(
            1j * self._k_idx_g[:, None] * self.omega * t_grid[None, :]
        )                                                          # (n_harm_g, n_samples)
        g_grid = np.einsum(
            "ht,jhr->jrt", weights, self.gs,
        )                                                          # (n_terms, r, n_samples)
        # Periodic extension: append t=T and value at t=0
        t_grid_ext = np.append(t_grid, self._T)
        g_grid_ext = np.concatenate([g_grid, g_grid[..., :1]], axis=-1)
        # CubicSpline handles real y only; split real/imag.  For a real
        # KSE with conjugate-symmetric HB gs the imaginary part is
        # ~machine noise, but split anyway for safety.
        self._g_spline_real = CubicSpline(
            t_grid_ext, g_grid_ext.real, axis=-1, bc_type="periodic",
        )
        self._g_spline_imag = CubicSpline(
            t_grid_ext, g_grid_ext.imag, axis=-1, bc_type="periodic",
        )

    # ─── IFFT weights + monomials ─────────────────────────────────────
    def _ifft_weights(self, t: float) -> np.ndarray:
        return np.exp(1j * self._k_idx * self.omega * t)

    def _monomials(self, s: np.ndarray) -> np.ndarray:
        r"""``prod_l s_l**j_l`` for every multi-index row."""
        return np.prod(s[None, :] ** self.multiindices, axis=1)   # (n_terms,)

    # ─── Encode / decode — identical to the constant-g ROM ────────────
    def decode(self, t: float, s: np.ndarray) -> np.ndarray:
        r""":math:`P(t, s) = \sum_j P_j(t)\, s^j`."""
        weights = self._ifft_weights(t)                           # (n_harm,)
        P_t = np.einsum("h,jhn->jn", weights, self.PS)            # (n_terms, n)
        monos = self._monomials(s)                                # (n_terms,)
        return monos @ P_t                                         # (n,)

    def encode(self, t: float, x: np.ndarray) -> np.ndarray:
        r""":math:`s = w(t)^\ast x`."""
        weights = self._ifft_weights(t)                           # (n_harm,)
        W_phys = np.einsum("h,hjr->jr", weights, self.W)          # (n, r)
        return W_phys.conj().T @ x

    # ─── Periodic latent dynamics — the only material change ─────────
    def latent_space_dynamics(
        self, t: float, s: np.ndarray,
    ) -> np.ndarray:
        r"""
        :math:`\dot s = \sum_j g_j(t)\, s^j` with
        :math:`g_j(t) = \sum_k g_{j,k}\, e^{i k \omega_{HB} t}`.

        Fast path: cubic spline over a pre-built fine grid.
        Slow path (``self.use_g_spline = False``): exact IFFT of the
        HB coefficients at every call.  Use the slow path to test
        whether spline undersampling is producing artefacts.
        """
        if self.use_g_spline:
            if self._g_spline_real is None:
                self._build_g_spline()
            t_mod = t % self._T
            g_t_real = self._g_spline_real(t_mod)                 # (n_terms, r)
            g_t_imag = self._g_spline_imag(t_mod)
            g_t = g_t_real + 1j * g_t_imag                        # (n_terms, r) complex
        else:
            # Exact IFFT: no interpolation artefact possible.
            # +1j to match the shift-DOWN convention in the SSM solve
            # (see _build_g_spline for the same choice and explanation).
            weights = np.exp(
                1j * self._k_idx_g * self.omega * t
            )                                                      # (n_harm_g,)
            g_t = np.einsum(
                "h,jhr->jr", weights, self.gs,
            )                                                      # (n_terms, r) complex
        monos = self._monomials(s)                                 # (n_terms,)
        return monos @ g_t                        # (r,)
