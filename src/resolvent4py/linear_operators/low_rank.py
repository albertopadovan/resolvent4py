import typing

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from .linear_operator import LinearOperator


class LowRankLinearOperator(LinearOperator):
    r"""
    Class for a linear operator of the form

    .. math::

        L = U \Sigma V^*,

    where :math:`U`, :math:`\Sigma` and :math:`V` are matrices of
    conformal sizes (and :math:`\Sigma` is not necessarily diagonal).

    :param U: a tall and skinny matrix
    :type U: SLEPc.BV
    :param Sigma: 2D numpy array
    :type Sigma: numpy.ndarray
    :param V: a tall and skinny matrix
    :type V: SLEPc.BV
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "LowRankLinearOperator",
        U: SLEPc.BV,
        Sigma: np.ndarray,
        V: SLEPc.BV,
        nblocks: typing.Optional[typing.Union[int, None]] = None,
    ) -> None:
        self.U = U
        self.Sigma = Sigma
        self.V = V
        dimensions = (U.getSizes()[0], V.getSizes()[0])
        super().__init__(
            U.getComm(), "LowRankLinearOperator", dimensions, nblocks
        )

    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        q = self.Sigma @ self.V.dotVec(x)
        self.U.multVec(1.0, 0.0, y, q)
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        q = self.Sigma.conj().T @ self.U.dotVec(x)
        self.V.multVec(1.0, 0.0, y, q)
        return y

    def apply_mat(self, X, Y=None):
        M = X.dot(self.V)
        L = self.Sigma @ M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, PETSc.COMM_SELF)
        Y = self.create_left_bv(X.getSizes()[-1]) if Y == None else Y
        Y.mult(1.0, 0.0, self.U, Lm)
        Lm.destroy()
        M.destroy()
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        M = X.dot(self.U)
        L = self.Sigma.conj().T @ M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, PETSc.COMM_SELF)
        Y = self.create_right_bv(X.getSizes()[-1]) if Y == None else Y
        Y.mult(1.0, 0.0, self.V, Lm)
        Lm.destroy()
        M.destroy()
        return Y

    def destroy(self):
        # U, Sigma and V are all user-supplied (passed to __init__); nothing
        # is created internally, so there is nothing to destroy here.
        pass
