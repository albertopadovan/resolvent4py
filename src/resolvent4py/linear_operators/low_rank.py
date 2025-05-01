import typing

import numpy as np
from mpi4py import MPI
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

    :param comm: MPI communicator :code:`MPI.COMM_WORLD`
    :type comm: MPI.Comm
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
        comm: MPI.Comm,
        U: SLEPc.BV,
        Sigma: np.ndarray,
        V: SLEPc.BV,
        nblocks: typing.Optional[int] = None,
    ) -> None:
        self.U = U
        self.Sigma = Sigma
        self.V = V
        dimensions = (U.getSizes()[0], V.getSizes()[0])
        super().__init__(comm, "LowRankLinearOperator", dimensions, nblocks)

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
        Lm = PETSc.Mat().createDense(L.shape, None, L, MPI.COMM_SELF)
        Y = X.duplicate() if Y == None else Y
        Y.mult(1.0, 0.0, self.U, Lm)
        Lm.destroy()
        M.destroy()
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        M = X.dot(self.U)
        L = self.Sigma.conj().T @ M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, MPI.COMM_SELF)
        Y = X.duplicate() if Y == None else Y
        Y.mult(1.0, 0.0, self.V, Lm)
        Lm.destroy()
        M.destroy()
        return Y

    def destroy_U(self: "LowRankLinearOperator") -> None:
        self.U.destroy()

    def destroy_V(self: "LowRankLinearOperator") -> None:
        self.V.destroy()

    def destroy_Sigma(self: "LowRankLinearOperator") -> None:
        del self.Sigma

    def destroy(self):
        self.destroy_U()
        self.destroy_V()
        self.destroy_Sigma()
