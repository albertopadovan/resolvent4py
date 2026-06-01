import typing
import numpy as np
from .linear_operator import LinearOperator
from ..utils.bv import bv_add, bv_real, bv_conj
from ..utils.vector import vec_real

from mpi4py import MPI


class TimePeriodicMatrixLinearOperator(LinearOperator):
    r"""
    Time-periodic linear operator built from the Fourier coefficients of
    a :math:`T`-periodic operator-valued function
    :math:`A(t) = A(t + T)`:

    .. math::

        L\,v
        \;=\;
        A(t)\, v
        \;=\;
        \sum_{k=-r_b}^{r_b} A_k\, v\, e^{i \omega_k t}.

    The Fourier coefficients :math:`A_k` are supplied as a list of
    :class:`LinearOperator` objects, paired with their angular
    frequencies :math:`\omega_k`.  The operator is evaluated at a single
    time instant ``time`` (changing ``self.time`` between calls re-uses
    the same :math:`A_k` and just updates the exponential weights).

    If ``freqs`` is *one-sided* (i.e. ``np.min(freqs) == 0``), the
    operator assumes :math:`A(t)` is real and uses the conjugate
    symmetry :math:`A_{-k} = \overline{A_k}` to avoid storing the
    negative-frequency coefficients.  The ``apply*`` methods then take
    a fast path when the input is real and a slightly more expensive
    path when it is complex (handled internally via the identity
    :math:`A_{-k} v = \overline{A_k \overline{v}}`).

    A shifted operator :math:`A(t) - s I` can be obtained by composing
    this class with
    :class:`~resolvent4py.linear_operators.ShiftAndScaleLinearOperator`.

    :param Alst: list of Fourier coefficient operators
        :math:`\{A_k\}`.  All entries must share the same domain and
        range dimensions.
    :type Alst: List[LinearOperator]
    :param freqs: angular frequencies :math:`\{\omega_k\}`, one per
        entry of ``Alst``.
    :type freqs: numpy.ndarray
    :param time: time instant :math:`t` at which to evaluate
        :math:`A(t)`.
    :type time: float
    :param nblocks: number of blocks if the operator has a known block
        structure (forwarded to :class:`LinearOperator`).
    :type nblocks: Optional[int], default None
    """

    def __init__(
        self: "TimePeriodicMatrixLinearOperator",
        Alst: typing.List[LinearOperator],
        freqs: np.ndarray,
        time: float,
        nblocks: typing.Optional[int] = None,
    ) -> None:
        comm = Alst[0].get_comm()
        dimensions = Alst[0].get_dimensions()
        self.Alst = Alst
        self.freqs = freqs
        self.time = time
        self._real_A = True if np.min(freqs) == 0 else False
        self.create_intermediate_vectors()
        # Work BVs are lazily (re)allocated inside apply_mat /
        # apply_hermitian_transpose_mat so they always match the
        # number of columns of the BV passed in.  The *cc variants
        # are used only on the complex-input path when self._real_A.
        self.bvleft = None
        self.bvright = None
        self.bvleftcc = None
        self.bvrightcc = None
        super().__init__(
            comm, "TimePeriodicMatrixLinearOperator", dimensions, nblocks
        )

    def create_intermediate_vectors(self) -> None:
        r"""Allocate the work vectors used by :meth:`apply` and
        :meth:`apply_hermitian_transpose`."""
        self.vright = self.Alst[0].create_right_vector()
        self.vleft = self.Alst[0].create_left_vector()
        if self._real_A:
            self.vrightcc = self.vright.duplicate()
            self.vleftcc = self.vleft.duplicate()
        else:
            self.vrightcc = None
            self.vleftcc = None

    def _ensure_work_bv(self, bv, create_fn, ncols):
        r"""Return a BV produced by ``create_fn(ncols)``, reusing the
        passed-in ``bv`` if it already has the right column count and
        destroying it otherwise.  Used to keep the work BVs in sync
        with the column count of the BV currently being acted on."""
        if bv is None or bv.getSizes()[-1] != ncols:
            if bv is not None:
                bv.destroy()
            bv = create_fn(ncols)
        return bv

    def set_evaluation_time(self, time: float) -> None:
        r"""Overwrite :code:`self.time`, used by subsequent ``apply*``
        calls to compute the exponential weights :math:`e^{i\omega_k t}`."""
        self.time = time

    def apply(self, x, y=None):
        y = x.duplicate() if y is None else y
        y.zeroEntries()
        # Check if x is real (matters only if self._real_A is True)
        if self._real_A:
            sq_norm = np.linalg.norm(x.getArray().imag) ** 2
            norm = np.sqrt(self.get_comm().tompi4py().allreduce(sq_norm, op=MPI.SUM))
            if norm <= 1e-14:
                is_x_real = True
            else:
                is_x_real = False
                xcc = x.copy()
                xcc.conjugate()

        for k, Ak in enumerate(self.Alst):
            self.vleft = Ak.apply(x, self.vleft)
            self.vleft.scale(np.exp(1j * self.freqs[k] * self.time))
            # If self._real_A is true, then leverage the fact that
            # A_{k} = conj(A_{-k})
            if self._real_A:
                # If the input vector is real, then use the usual symmetry
                # If the input vector is complex, then use the fact that
                # A_{-k}v = conj(A_{k})v = conj(A_k conj(v))
                if is_x_real:
                    self.vleft = vec_real(self.vleft, inplace=True)
                    coef = 2.0 if self.freqs[k] > 0.0 else 1.0
                    self.vleft.scale(coef)
                else:
                    if self.freqs[k] > 0.0:
                        self.vleftcc = Ak.apply(xcc, self.vleftcc)
                        self.vleftcc.conjugate()
                        self.vleftcc.scale(np.exp(-1j * self.freqs[k] * self.time))
                        self.vleft.axpy(1.0, self.vleftcc)
            y.axpy(1.0, self.vleft)

        if self._real_A and not is_x_real:
            xcc.destroy()
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = x.duplicate() if y is None else y
        y.zeroEntries()
        # Check if x is real (matters only if self._real_A is True)
        if self._real_A:
            sq_norm = np.linalg.norm(x.getArray().imag) ** 2
            norm = np.sqrt(
                self.get_comm().tompi4py().allreduce(sq_norm, op=MPI.SUM)
            )
            if norm <= 1e-14:
                is_x_real = True
            else:
                is_x_real = False
                xcc = x.copy()
                xcc.conjugate()

        for k, Ak in enumerate(self.Alst):
            self.vright = Ak.apply_hermitian_transpose(x, self.vright)
            self.vright.scale(np.exp(-1j * self.freqs[k] * self.time))
            # For real A(t): A_{-k}^H = A_k^T, and the identity
            # A_k^T x = conj(A_k^H conj(x)) holds for any x.  For real x
            # this collapses to 2 Re[A_k^H x exp(-iω_k t)]; for complex
            # x we form the (k, -k) pair explicitly.
            if self._real_A:
                if is_x_real:
                    self.vright = vec_real(self.vright, inplace=True)
                    coef = 2.0 if self.freqs[k] > 0.0 else 1.0
                    self.vright.scale(coef)
                else:
                    if self.freqs[k] > 0.0:
                        self.vrightcc = Ak.apply_hermitian_transpose(
                            xcc, self.vrightcc
                        )
                        self.vrightcc.conjugate()
                        self.vrightcc.scale(
                            np.exp(1j * self.freqs[k] * self.time)
                        )
                        self.vright.axpy(1.0, self.vrightcc)
            y.axpy(1.0, self.vright)

        if self._real_A and not is_x_real:
            xcc.destroy()
        return y

    def apply_mat(self, X, Y=None):
        Y = X.copy() if Y is None else Y
        # X.copy() seeds Y with X's values; zero so we don't pick up an
        # extra X term in the accumulation below.
        Y.scale(0.0)
        ncols = X.getSizes()[-1]
        self.bvleft = self._ensure_work_bv(
            self.bvleft, self.Alst[0].create_left_bv, ncols
        )
        # Check if X is real (matters only if self._real_A is True)
        if self._real_A:
            Xm = X.getMat()
            sq_norm = np.linalg.norm(Xm.getDenseArray().imag) ** 2
            X.restoreMat(Xm)
            norm = np.sqrt(
                self.get_comm().tompi4py().allreduce(sq_norm, op=MPI.SUM)
            )
            if norm <= 1e-14:
                is_X_real = True
            else:
                is_X_real = False
                Xcc = X.copy()
                Xcc = bv_conj(Xcc, inplace=True)
                self.bvleftcc = self._ensure_work_bv(
                    self.bvleftcc, self.Alst[0].create_left_bv, ncols
                )

        for k, Ak in enumerate(self.Alst):
            self.bvleft = Ak.apply_mat(X, self.bvleft)
            self.bvleft.scale(np.exp(1j * self.freqs[k] * self.time))
            if self._real_A:
                if is_X_real:
                    self.bvleft = bv_real(self.bvleft, inplace=True)
                    coef = 2.0 if self.freqs[k] > 0.0 else 1.0
                    self.bvleft.scale(coef)
                else:
                    if self.freqs[k] > 0.0:
                        self.bvleftcc = Ak.apply_mat(Xcc, self.bvleftcc)
                        self.bvleftcc = bv_conj(self.bvleftcc, inplace=True)
                        self.bvleftcc.scale(
                            np.exp(-1j * self.freqs[k] * self.time)
                        )
                        self.bvleft = bv_add(
                            1.0, self.bvleft, self.bvleftcc
                        )
            Y = bv_add(1.0, Y, self.bvleft)

        if self._real_A and not is_X_real:
            Xcc.destroy()
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        Y = X.copy() if Y is None else Y
        Y.scale(0.0)
        ncols = X.getSizes()[-1]
        self.bvright = self._ensure_work_bv(
            self.bvright, self.Alst[0].create_right_bv, ncols
        )
        if self._real_A:
            Xm = X.getMat()
            sq_norm = np.linalg.norm(Xm.getDenseArray().imag) ** 2
            X.restoreMat(Xm)
            norm = np.sqrt(
                self.get_comm().tompi4py().allreduce(sq_norm, op=MPI.SUM)
            )
            if norm <= 1e-14:
                is_X_real = True
            else:
                is_X_real = False
                Xcc = X.copy()
                Xcc = bv_conj(Xcc, inplace=True)
                self.bvrightcc = self._ensure_work_bv(
                    self.bvrightcc, self.Alst[0].create_right_bv, ncols
                )

        for k, Ak in enumerate(self.Alst):
            self.bvright = Ak.apply_hermitian_transpose_mat(X, self.bvright)
            self.bvright.scale(np.exp(-1j * self.freqs[k] * self.time))
            if self._real_A:
                if is_X_real:
                    self.bvright = bv_real(self.bvright, inplace=True)
                    coef = 2.0 if self.freqs[k] > 0.0 else 1.0
                    self.bvright.scale(coef)
                else:
                    if self.freqs[k] > 0.0:
                        self.bvrightcc = Ak.apply_hermitian_transpose_mat(
                            Xcc, self.bvrightcc
                        )
                        self.bvrightcc = bv_conj(
                            self.bvrightcc, inplace=True
                        )
                        self.bvrightcc.scale(
                            np.exp(1j * self.freqs[k] * self.time)
                        )
                        self.bvright = bv_add(
                            1.0, self.bvright, self.bvrightcc
                        )
            Y = bv_add(1.0, Y, self.bvright)

        if self._real_A and not is_X_real:
            Xcc.destroy()
        return Y

    def destroy(self):
        for obj in (
            self.bvleft,
            self.bvright,
            self.bvleftcc,
            self.bvrightcc,
            self.vleft,
            self.vright,
            self.vleftcc,
            self.vrightcc,
        ):
            if obj is not None:
                obj.destroy()
