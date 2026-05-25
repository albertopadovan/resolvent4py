"""
Serial, pure-numpy reduced-order model built from the SSM polynomial
expansion computed by :class:`SpectralSubmanifold` (which itself runs
parallel under PETSc/SLEPc).

After ``SpectralSubmanifold.solve(...)`` finishes, the relevant pieces
— polynomial coefficients ``ps``, left eigenvectors ``W``, eigenvalues
``Lams``, latent-space nonlinear coefficients ``gs``, multi-indices,
and (for a time-periodic SSM) the orbit-tangent neutral Floquet mode
— can be gathered into numpy arrays and handed to
:class:`SpectralSubmanifoldROM`.  The ROM is then self-contained
(numpy only, no PETSc / no MPI), suitable for
``scipy.integrate.solve_ivp`` and post-processing.

The class supports both **autonomous** (time-invariant) and
**time-periodic** SSMs.  Periodicity is auto-detected from the shape of
``PS`` and triggers a harmonic-balanced (HB) physical-space encoder /
decoder via IFFT, matching the recipe used in
``examples/ssm_periodic/kse/demonstrate_ssm.py``.
"""

from typing import Optional

import numpy as np


__all__ = ["SpectralSubmanifoldROM"]


class SpectralSubmanifoldROM:
    r"""
    Serial, numpy-only reduced-order model from a
    :class:`SpectralSubmanifold` solve.

    Implements

    - :meth:`decode` — latent coordinate :math:`s \to` physical state
      :math:`x` (at time :math:`t` for periodic SSMs).
    - :meth:`encode` — physical state :math:`x \to` latent coordinate
      :math:`s` (at time :math:`t` for periodic SSMs).
    - :meth:`latent_space_dynamics` — :math:`\dot{s} = \Lambda s + g(s)`,
      compatible with :func:`scipy.integrate.solve_ivp`.
    - :meth:`neutral_project` — physical-space oblique projector that
      removes the orbit-tangent neutral Floquet direction (periodic
      only).

    **Autonomous mode** (``PS.ndim == 2``):

    - ``PS`` has shape ``(n_terms, n)``.
    - ``W``  has shape ``(n, r)``.
    - Decoder: :math:`x = \sum_\alpha s^{\alpha}\, P_\alpha`.
    - Encoder: :math:`s = W^H x`.

    **Time-periodic mode** (``PS.ndim == 3``, auto-detected):

    - ``PS`` has shape ``(n_terms, n_harmonics, n)``  (HB blocks).
    - ``W``  has shape ``(n_harmonics, n, r)``.
    - ``omega`` is required (fundamental angular frequency
      :math:`\omega = 2\pi/T`).
    - Let
      :math:`E(t) = (e^{i k \omega t})_{k=-n_f}^{n_f}` be the IFFT
      weights.  Then
        decoder:  :math:`x(t) = \mathrm{Re}\,[ E(t) \cdot
                  (\sum_\alpha s^\alpha P_\alpha) ]`,
        encoder:  :math:`s = W_\text{phys}(t)^H x` with
                  :math:`W_\text{phys}(t) = E(t) \cdot W`.
    """

    def __init__(
        self,
        multiindices: np.ndarray,
        Lams: np.ndarray,
        gs: np.ndarray,
        PS: np.ndarray,
        W: np.ndarray,
        conj_to_linear_dynamics: bool = False,
        omega: Optional[float] = None,
        v_neutral: Optional[np.ndarray] = None,
        w_neutral: Optional[np.ndarray] = None,
    ) -> None:
        r"""
        :param multiindices: ``(n_terms, r)`` integer array; the
            multi-index of each polynomial term (so ``PS[i]`` carries
            :math:`s^{\text{multiindices}[i]}`).
        :param Lams: ``(r,)`` complex array of leading eigenvalues of
            the latent-space linear part.
        :param gs: ``(n_terms, r)`` complex array of latent-space
            nonlinear coefficients (linear-order entries are ignored —
            those are handled by ``Lams``).
        :param PS: SSM polynomial coefficients in physical (autonomous)
            or HB (periodic) layout.  See class docstring.
        :param W: left eigenvectors used by the encoder.  See class
            docstring.
        :param conj_to_linear_dynamics: if ``True``, ``g(s)`` is forced
            to zero (latent space conjugate to a linear flow).
        :param omega: fundamental angular frequency; **required** for
            periodic SSMs, ignored otherwise.
        :param v_neutral: optional ``(n_harmonics, n)`` complex array —
            the :math:`\sigma = 0` right neutral Floquet mode in HB
            form.  Periodic only; needed by :meth:`neutral_project`.
        :param w_neutral: optional ``(n_harmonics, n)`` complex array —
            the :math:`\sigma = 0` left neutral Floquet mode in HB
            form.  Biorthogonal to ``v_neutral`` in HB.

        :raises ValueError: if ``PS.ndim`` is not 2 or 3, if ``omega``
            is missing when ``PS.ndim == 3``, or if the shapes of
            ``W`` / ``v_neutral`` / ``w_neutral`` are inconsistent
            with ``PS``.
        """
        self.multiindices = np.asarray(multiindices, dtype=np.int64)
        self.Lams = np.asarray(Lams, dtype=np.complex128)
        self.gs = np.asarray(gs, dtype=np.complex128)
        self.PS = np.asarray(PS, dtype=np.complex128)
        self.W = np.asarray(W, dtype=np.complex128)
        self.conj_to_linear_dynamics = bool(conj_to_linear_dynamics)

        self.r = self.Lams.shape[0]

        # Periodicity is detected from the polynomial-coefficient layout:
        # (n_terms, n) → autonomous;  (n_terms, n_harmonics, n) → periodic.
        self.periodic = self.PS.ndim == 3

        if self.periodic:
            if omega is None:
                raise ValueError(
                    "periodic SSM (PS.ndim == 3) requires `omega` "
                    "(the fundamental angular frequency = 2*pi / T)."
                )
            self.omega = float(omega)
            self.n_harmonics = int(self.PS.shape[1])
            self.n = int(self.PS.shape[2])
            self.nf = (self.n_harmonics - 1) // 2
            self._k_idx = np.arange(-self.nf, self.nf + 1)

            if self.W.shape != (self.n_harmonics, self.n, self.r):
                raise ValueError(
                    f"Expected W of shape "
                    f"({self.n_harmonics}, {self.n}, {self.r}); "
                    f"got {self.W.shape}."
                )

            self.v_neutral = (
                np.asarray(v_neutral, dtype=np.complex128)
                if v_neutral is not None else None
            )
            self.w_neutral = (
                np.asarray(w_neutral, dtype=np.complex128)
                if w_neutral is not None else None
            )
            for nm, arr in (("v_neutral", self.v_neutral),
                            ("w_neutral", self.w_neutral)):
                if arr is not None and arr.shape != (self.n_harmonics, self.n):
                    raise ValueError(
                        f"Expected `{nm}` of shape "
                        f"({self.n_harmonics}, {self.n}); got {arr.shape}."
                    )
        else:
            if self.PS.ndim != 2:
                raise ValueError(
                    f"`PS` must be 2-D (autonomous) or 3-D (periodic); "
                    f"got ndim={self.PS.ndim}."
                )
            self.omega = None
            self.n = int(self.PS.shape[1])
            if self.W.shape != (self.n, self.r):
                raise ValueError(
                    f"Expected W of shape ({self.n}, {self.r}); "
                    f"got {self.W.shape}."
                )
            self.v_neutral = None
            self.w_neutral = None

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _ifft_weights(self, t: float) -> np.ndarray:
        r"""``exp(i k omega t)`` for ``k`` in ``[-nf, ..., nf]``."""
        return np.exp(1j * self._k_idx * self.omega * t)

    def _monomials(self, s: np.ndarray) -> np.ndarray:
        r"""``s**multiindices`` reduced along the latent axis."""
        s = np.asarray(s)
        return np.prod(s[None, :] ** self.multiindices, axis=1)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def decode(self, t: float, s: np.ndarray) -> np.ndarray:
        r"""
        Decode the latent coordinate ``s`` into the physical state.

        :param t: time at which to evaluate the decoder (used only for
            periodic SSMs; ignored for autonomous SSMs).
        :param s: ``(r,)`` complex latent coordinate.
        :return: real ``(n,)`` physical state.
        """
        coefs = self._monomials(s).astype(np.complex128, copy=False)
        if self.periodic:
            hb = np.einsum("i,ihj->hj", coefs, self.PS)  # (n_harmonics, n)
            return (self._ifft_weights(t) @ hb).real
        return (coefs @ self.PS).real

    def encode(self, t: float, x: np.ndarray) -> np.ndarray:
        r"""
        Encode the physical state ``x`` into the latent coordinate.

        :param t: time at which to evaluate the encoder (used only for
            periodic SSMs; ignored for autonomous SSMs).
        :param x: ``(n,)`` real (or complex) physical state.
        :return: ``(r,)`` complex latent coordinate.
        """
        if self.periodic:
            W_phys = np.einsum(
                "h,hjr->jr", self._ifft_weights(t), self.W,
            )  # (n, r)
            return W_phys.conj().T @ x
        return self.W.conj().T @ x

    def latent_space_dynamics(
        self, t: float, s: np.ndarray,
    ) -> np.ndarray:
        r"""
        Right-hand side of the latent flow:
        :math:`\dot{s} = \Lambda s + g(s)`.

        Signature ``(t, s) -> ds/dt`` matches what
        :func:`scipy.integrate.solve_ivp` expects.  ``t`` is unused.

        :param t: time (ignored — present for solver compatibility).
        :param s: ``(r,)`` complex latent coordinate.
        :return: ``(r,)`` complex ``ds/dt``.
        """
        ds = self.Lams * s
        if not self.conj_to_linear_dynamics:
            # Skip the first ``r + 1`` rows: the DC term and the ``r``
            # linear terms, whose contribution is already captured by
            # ``Lams * s`` above.
            _start = self.r + 1
            J = self.multiindices[_start:]
            G = self.gs[_start:]
            monomials = np.prod(s[None, :] ** J, axis=1)
            ds += monomials @ G
        return ds

    def neutral_project(
        self, t: float, x: np.ndarray,
    ) -> np.ndarray:
        r"""
        Remove the orbit-tangent neutral Floquet direction from ``x``
        at time ``t`` (periodic SSMs only).

        Uses the :math:`\sigma = 0` neutral pair stored in
        ``v_neutral`` / ``w_neutral`` (HB form), IFFT-ed at time ``t``:

        .. math::

            x_{\text{proj}} = x - v(t)\,\frac{w(t)^H x}{w(t)^H v(t)}.

        :param t: time at which to evaluate the neutral mode.
        :param x: ``(n,)`` physical state to project.
        :return: ``(n,)`` projected state.
        :raises NotImplementedError: if the SSM is autonomous.
        :raises ValueError: if ``v_neutral`` / ``w_neutral`` were not
            supplied to the constructor.
        """
        if not self.periodic:
            raise NotImplementedError(
                "neutral_project is only defined for periodic SSMs."
            )
        if self.v_neutral is None or self.w_neutral is None:
            raise ValueError(
                "neutral_project requires `v_neutral` and `w_neutral` "
                "to have been provided to the constructor."
            )
        w_ifft = self._ifft_weights(t)
        v_t = w_ifft @ self.v_neutral
        w_t = w_ifft @ self.w_neutral
        return x - v_t * (np.vdot(w_t, x) / np.vdot(w_t, v_t))
