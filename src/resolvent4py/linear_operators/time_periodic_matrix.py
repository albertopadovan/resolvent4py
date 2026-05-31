import typing

import numpy as np

from .linear_operator import LinearOperator
from ..utils.bv import bv_add


class TimePeriodicMatrixLinearOperator:
    r"""
    A :math:`T`-periodic family of linear operators built from the
    Fourier coefficients :math:`\{A_k\}` of
    :math:`A(t) = A(t + T)`, optionally shifted by :math:`s I`:

    .. math::

        L(t)\, v
        \;=\;
        \left(A(t) - s I\right) v
        \;=\;
        \sum_{k=-r_b}^{r_b} A_k\, v\, e^{i \omega_k t}
        \;-\; s\, v,
        \qquad s \in \mathbb{C}.

    This class is **not** itself a :class:`LinearOperator` — it has no
    fixed time, so ``apply`` would be meaningless on it.  To obtain a
    concrete :class:`LinearOperator` evaluated at a particular time
    :math:`t`, call :meth:`at`:

    .. code-block:: python

        family = TimePeriodicMatrixLinearOperator(Alst, freqs)
        snap   = family.at(1.5)         # snap is a LinearOperator
        y      = snap.apply(x)          # evaluates A(1.5) x - s x

    Snapshots are cheap — they hold only a parent reference and a
    scalar :math:`t`.  All Fourier coefficients :math:`A_k` and the
    reusable work vectors / BVs live on the parent and are shared by
    every snapshot, so multiple snapshots at different times can
    coexist without re-allocation.

    Snapshots share the parent's work buffers, so two snapshots cannot
    be applied concurrently from different Python threads.  MPI ranks
    are fine; thread-level concurrency is not.

    :param Alst: list of Fourier coefficient operators
        :math:`\{A_k\}`.  All entries must share the same domain and
        range dimensions.
    :type Alst: List[LinearOperator]
    :param freqs: angular frequencies :math:`\{\omega_k\}`, one per
        entry of ``Alst``.
    :type freqs: numpy.ndarray
    :param nblocks: number of blocks if the operator has a known block
        structure (forwarded to each :class:`LinearOperator` snapshot).
    :type nblocks: Optional[int], default None
    :param s: optional complex shift applied as
        :math:`L \mapsto L - s I`.  Requires :math:`A_k` to be square.
    :type s: Optional[complex], default None
    """

    def __init__(
        self,
        Alst: typing.List[LinearOperator],
        freqs: np.ndarray,
        nblocks: typing.Optional[int] = None,
        s: typing.Optional[complex] = None,
    ) -> None:
        if len(Alst) != len(freqs):
            raise ValueError(
                f"len(Alst)={len(Alst)} must equal len(freqs)={len(freqs)}."
            )
        comm = Alst[0].get_comm()
        dimensions = Alst[0].get_dimensions()
        if s is not None and dimensions[0][-1] != dimensions[-1][-1]:
            raise ValueError(
                "The operator must be square if you wish to pass a 's' value."
            )

        self.Alst = Alst
        self.freqs = freqs
        self.s = s

        # Metadata used by every snapshot to populate the LinearOperator
        # base.  Accessed via get_comm() / get_dimensions() / get_nblocks().
        self._comm = comm
        self._dimensions = dimensions
        self._nblocks = nblocks

        # Caches for the LinearOperator base-class flags.  The first
        # snapshot to need each flag pays one apply-probe and seeds the
        # cache here; every subsequent at(t) just reads it back.  This
        # relies on real_flag / block_cc_flag being independent of t,
        # which holds for this family (the time dependence enters only
        # through scalar complex exponentials that preserve both
        # properties).
        self._real_flag_cache = None
        self._block_cc_flag_cache = None

        # Reusable work buffers — allocated once, shared by all
        # snapshots through the _apply_*_at helpers below.
        self.vleft = Alst[0].create_left_vector()
        self.vright = Alst[0].create_right_vector()
        self.bvleft = Alst[0].create_left_bv()
        self.bvright = Alst[0].create_right_bv()

    # ── Metadata accessors (mirror the LinearOperator interface so the
    # ── family can be inspected the same way as a snapshot) ────────────

    def get_comm(self):
        return self._comm

    def get_dimensions(self):
        return self._dimensions

    def get_nblocks(self):
        return self._nblocks

    # ── Snapshot factory ───────────────────────────────────────────────

    def at(self, t: float) -> "_TimePeriodicMatrixSnapshot":
        r"""Return a :class:`LinearOperator` snapshot of this family at
        time ``t``.  The snapshot shares the parent's work buffers and
        Fourier coefficients; only the scalar :math:`t` is held on the
        snapshot itself."""
        return _TimePeriodicMatrixSnapshot(self, t)

    # ── Per-time math (called by snapshots) ────────────────────────────

    def _apply_at(self, x, t, y=None):
        y = x.duplicate() if y is None else y
        y.zeroEntries()
        for k, Ak in enumerate(self.Alst):
            self.vleft = Ak.apply(x, self.vleft)
            y.axpy(np.exp(1j * self.freqs[k] * t), self.vleft)
        if self.s is not None:
            y.axpy(-self.s, x)
        return y

    def _apply_hermitian_transpose_at(self, x, t, y=None):
        y = x.duplicate() if y is None else y
        y.zeroEntries()
        for k, Ak in enumerate(self.Alst):
            self.vright = Ak.apply_hermitian_transpose(x, self.vright)
            y.axpy(np.exp(-1j * self.freqs[k] * t), self.vright)
        if self.s is not None:
            y.axpy(-np.conj(self.s), x)
        return y

    def _apply_mat_at(self, X, t, Y=None):
        Y = X.copy() if Y is None else Y
        Y.scale(0.0)
        for k, Ak in enumerate(self.Alst):
            self.bvleft = Ak.apply_mat(X, self.bvleft)
            Y = bv_add(np.exp(1j * self.freqs[k] * t), Y, self.bvleft)
        if self.s is not None:
            Y = bv_add(-self.s, Y, X)
        return Y

    def _apply_hermitian_transpose_mat_at(self, X, t, Y=None):
        Y = X.copy() if Y is None else Y
        Y.scale(0.0)
        for k, Ak in enumerate(self.Alst):
            self.bvright = Ak.apply_hermitian_transpose_mat(X, self.bvright)
            Y = bv_add(np.exp(-1j * self.freqs[k] * t), Y, self.bvright)
        if self.s is not None:
            Y = bv_add(-np.conj(self.s), Y, X)
        return Y

    def destroy(self) -> None:
        r"""Destroy the work vectors / BVs owned by this family.  Each
        snapshot's :meth:`destroy` is a no-op — call this once when
        the family is no longer needed."""
        for obj in (self.bvleft, self.bvright, self.vleft, self.vright):
            if obj is not None:
                obj.destroy()


class _TimePeriodicMatrixSnapshot(LinearOperator):
    r"""
    :class:`LinearOperator` view of a
    :class:`TimePeriodicMatrixLinearOperator` family evaluated at a
    fixed time ``t``.  Holds only the parent reference and the scalar
    ``t``; every ``apply*`` call delegates to the parent with ``t``
    baked in.  Created exclusively through
    :meth:`TimePeriodicMatrixLinearOperator.at`.
    """

    def __init__(
        self,
        parent: TimePeriodicMatrixLinearOperator,
        t: float,
    ) -> None:
        self._parent = parent
        self._t = float(t)
        super().__init__(
            parent.get_comm(),
            "TimePeriodicMatrixSnapshot",
            parent.get_dimensions(),
            parent.get_nblocks(),
        )

    def apply(self, x, y=None):
        return self._parent._apply_at(x, self._t, y)

    def apply_hermitian_transpose(self, x, y=None):
        return self._parent._apply_hermitian_transpose_at(x, self._t, y)

    def apply_mat(self, X, Y=None):
        return self._parent._apply_mat_at(X, self._t, Y)

    def apply_hermitian_transpose_mat(self, X, Y=None):
        return self._parent._apply_hermitian_transpose_mat_at(X, self._t, Y)

    # Override the LinearOperator base probes so only the first snapshot
    # pays the apply-with-random-vector cost; every subsequent at(t)
    # reuses the cached flag stored on the parent.
    def check_if_real_valued(self):
        if self._parent._real_flag_cache is None:
            self._parent._real_flag_cache = super().check_if_real_valued()
        return self._parent._real_flag_cache

    def check_if_complex_conjugate_structure(self):
        if self._parent._block_cc_flag_cache is None:
            self._parent._block_cc_flag_cache = (
                super().check_if_complex_conjugate_structure()
            )
        return self._parent._block_cc_flag_cache

    def destroy(self):
        # The snapshot owns nothing — work buffers belong to the parent.
        pass
