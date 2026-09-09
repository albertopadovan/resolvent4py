import typing

import numpy as np

from ..utils.bv import bv_add
from .linear_operator import LinearOperator


class ShiftAndScaleLinearOperator(LinearOperator):
    r"""
    Class for a linear operator of the form

    .. math::

        L = \alpha I + \beta A

    where :math:`A` is a (square) resolvent4py Linear operator
    (see :class:`.LinearOperator`) and :math:`I` is the identity.

    :param A: square linear operator
    :type A: LinearOperator
    :param alpha: complex-valued scalar
    :type alpha: Optional[complex], default is 1.0
    :param beta: complex-valued scalar
    :type beta: Optional[complex], default is 1.0
    """

    def __init__(
        self: "ShiftAndScaleLinearOperator",
        A: LinearOperator,
        alpha: typing.Optional[np.complex128] = 1.0,
        beta: typing.Optional[np.complex128] = 1.0,
    ) -> None:
        comm = A.get_comm()
        dimensions = A.get_dimensions()
        nrows, ncols = dimensions[0][-1], dimensions[-1][-1]
        if dimensions[0][-1] != dimensions[1][-1]:
            raise ValueError(
                f"A should be a square linear operator. "
                f"Currently nrows = {nrows} and ncols = {ncols}."
            )
        self.A = A
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            comm, "ShiftAndScaleLinearOperator", dimensions, A.get_nblocks()
        )

    def check_if_real_valued(self) -> bool:
        r"""Real iff :math:`A` is real **and** both shift and scale are
        real-valued scalars; otherwise the linear combination
        :math:`\alpha I + \beta A` maps real inputs to a complex
        output."""
        scalars_real = np.imag(self.alpha) == 0.0 and np.imag(self.beta) == 0.0
        return bool(self.A.get_real_flag() and scalars_real)

    def check_if_complex_conjugate_structure(self) -> bool:
        r"""Inherit the block-conjugate-symmetry flag from :math:`A` —
        :math:`\alpha I + \beta A` preserves whatever cc structure
        :math:`A` has (both :math:`I` and :math:`A` are diagonal /
        block-Toeplitz in the same basis)."""
        return self.A.get_block_cc_flag()

    def apply(self, x, y=None):
        y = x.duplicate() if y is None else y
        y = self.A.apply(x, y)
        y.scale(self.beta)
        y.axpy(self.alpha, x)
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = x.duplicate() if y is None else y
        y = self.A.apply_hermitian_transpose(x, y)
        y.scale(np.conj(self.beta))
        y.axpy(np.conj(self.alpha), x)
        return y

    def apply_mat(self, X, Y=None):
        Y = X.duplicate() if Y is None else Y
        Y = self.A.apply_mat(X, Y)
        Y.scale(self.beta)
        bv_add(self.alpha, Y, X)
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        Y = X.duplicate() if Y is None else Y
        Y = self.A.apply_hermitian_transpose_mat(X, Y)
        Y.scale(np.conj(self.beta))
        bv_add(np.conj(self.alpha), Y, X)
        return Y

    def destroy(self):
        pass
