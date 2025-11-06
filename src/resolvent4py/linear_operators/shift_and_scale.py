import typing
import numpy as np
from .linear_operator import LinearOperator
from ..utils.bv import bv_add


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

    def apply(self, x, y=None):
        y = x.duplicate() if y == None else y
        y = self.A.apply(x, y)
        y.scale(self.beta)
        y.axpy(self.alpha, x)
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = x.duplicate() if y == None else y
        y = self.A.apply_hermitian_transpose(x, y)
        y.scale(np.conj(self.beta))
        y.axpy(np.conj(self.alpha), x)
        return y

    def apply_mat(self, X, Y=None):
        Y = X.duplicate() if Y == None else Y
        Y = self.A.apply_mat(X, Y)
        Y.scale(self.beta)
        bv_add(self.alpha, Y, X)
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        Y = X.duplicate() if Y == None else Y
        Y = self.A.apply_hermitian_transpose_mat(X, Y)
        Y.scale(np.conj(self.beta))
        bv_add(np.conj(self.alpha), Y, X)
        return Y

    def destroy(self):
        pass
