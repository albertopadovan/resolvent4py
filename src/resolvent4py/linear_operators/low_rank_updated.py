from ..utils.bv_helpers import bv_add
from .low_rank import LowRankLinearOperator
from .linear_operator import LinearOperator

from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from .. import typing


class LowRankUpdatedLinearOperator(LinearOperator):
    r"""
    Class for a linear operator of the form

    .. math::

        L = A + B K C^*

    where :math:`A` is an instance of the :class:`.LinearOperator` class,
    and :math:`B`, :math:`K` and :math:`C` are low-rank (dense) matrices of
    conformal sizes. If :code:`A.solve()` is enabled, then the :code:`solve()`
    method in this class is implemented using the Woodbury matrix identity

    .. math::

        L^{-1} = A^{-1} - X D Y^*,

    where :math:`X`, :math:`D` and :math:`Y` the Woodbury factors defined as

    .. math::

        \textcolor{black}{X}
        = A^{-1}B,\quad \textcolor{black}{D} =
        K\left(I + C^* A^{-1}B K\right)^{-1},\quad
        \textcolor{black}{Y} = A^{-*}C,


    :param comm: MPI communicator :code:`MPI.COMM_WORLD`
    :type comm: MPI.Comm
    :param A: instance of the :class:`.LinearOperator` class
    :param B: tall and skinny matrix
    :type B: SLEPc.BV
    :param K: dense matrix
    :type K: numpy.ndarray
    :param C: tall and skinny matrix
    :type C: SLEPc.BV
    :param woodbury_factors: tuple :math:`(X, D, Y)` of Woodbury factors. If
        :code:`A.solve()` is enabled and the argument :code:`woodbury_factors`
        is :code:`None`, the factors :math:`X`, :math:`D` and :math:`Y` are
        computed at initialization
    :type woodbury_factors: Optional[Tuple[SLEPc.BV, numpy.ndarray,
        SLEPc.BV], None], default is None
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "LowRankUpdatedLinearOperator",
        comm: MPI.Comm,
        A: LinearOperator,
        B: SLEPc.BV,
        K: np.ndarray,
        C: SLEPc.BV,
        woodbury_factors: typing.Optional[
            typing.Tuple[SLEPc.BV, np.ndarray, SLEPc.BV]
        ] = None,
        nblocks: typing.Optional[int] = None,
    ) -> None:
        self.A = A
        self.L = LowRankLinearOperator(comm, B, K, C, nblocks)
        self.W = (
            self.compute_woodbury_operator(comm, nblocks)
            if woodbury_factors == None
            else LowRankLinearOperator(comm, *woodbury_factors, nblocks)
        )
        self.create_intermediate_vectors()
        super().__init__(
            comm, "LowRankUpdatedLinearOperator", A.get_dimensions(), nblocks
        )

    def compute_woodbury_operator(
        self: "LowRankUpdatedLinearOperator",
        comm: MPI.Comm,
        nblocks: typing.Union[int, None],
    ) -> LowRankLinearOperator:
        r"""
        :param comm: MPI communicator :code:`MPI.COMM_WORLD`
        :type comm: MPI.Comm
        :param nblocks: number of blocks (if the operator has block structure)
        :type nblocks: Unions[int, None]

        :return: a :class:`.LowRankLinearOperator` constructed from the
            Woodbury factors :code:`X`, :code:`D` and :code:`Y`
        :rtype: LowRankUpdatedLinearOperator
        """
        try:
            X = self.A.solve_mat(self.L.U)
            Y = self.A.solve_hermitian_transpose_mat(self.L.V)
            S = PETSc.Mat().createDense(
                self.L.Sigma.shape, None, self.L.Sigma, MPI.COMM_SELF
            )
            XS = SLEPc.BV().create(comm)
            XS.setSizes(X.getSizes()[0], self.L.Sigma.shape[-1])
            XS.setType("mat")
            XS.mult(1.0, 0.0, X, S)
            M = XS.dot(self.L.V)
            Ma = M.getDenseArray()
            D = self.L.Sigma @ sp.linalg.inv(np.eye(Ma.shape[0]) + Ma)
            W = LowRankLinearOperator(comm, X, D, Y, nblocks)
            XS.destroy()
            M.destroy()
            S.destroy()
        except:
            W = None
        return W

    def create_intermediate_vectors(
        self: "LowRankUpdatedLinearOperator",
    ) -> None:
        self.Ax = self.A.create_left_vector()
        self.ATx = self.A.create_right_vector()

    def create_intermediate_bv(
        self: "LowRankUpdatedLinearOperator", m: int
    ) -> SLEPc.BV:
        r"""
        Create matrix to store :math:`X D Y^* Z`, for any matrix
        Z with :math:`m` columns.

        .. attention::

            It is the user's responsibility to destroy this object
            when no longer needed.

        :param m: number of columns of :math:`Z`
        :type m: int

        :rtype: SLEPc.BV
        """
        X = SLEPc.BV().create(self._comm)
        X.setSizes(self._dimensions[0], m)
        X.setType("mat")
        return X

    def create_intermediate_bv_hermitian_transpose(
        self: "LowRankUpdatedLinearOperator", m: int
    ) -> SLEPc.BV:
        r"""
        Create matrix to store :math:`Y D X^* Z`, for any matrix
        Z with :math:`m` columns.

        .. attention::

            It is the user's responsibility to destroy this object
            when no longer needed.

        :param m: number of columns of :math:`Z`
        :type m: int

        :rtype: SLEPc.BV
        """
        X = SLEPc.BV().create(self._comm)
        X.setSizes(self._dimensions[-1], m)
        X.setType("mat")
        return X

    def apply(self, x, y=None):
        self.Ax = self.A.apply(x, self.Ax)
        y = self.L.apply(x, y)
        y.axpy(1.0, self.Ax)
        return y

    def apply_hermitian_transpose(self, x, y=None):
        self.ATx = self.A.apply_hermitian_transpose(x, self.ATx)
        y = self.L.apply_hermitian_transpose(x, y)
        y.axpy(1.0, self.ATx)
        return y

    def apply_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv(X.getSizes()[-1])
        Z = self.A.apply_mat(X, Z)
        Y = self.L.apply_mat(X, Y)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv_hermitian_transpose(
                X.getSizes()[-1]
            )
        Z = self.A.apply_hermitian_transpose_mat(X, Z)
        Y = self.L.apply_hermitian_transpose_mat(X, Y)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y

    def solve(self, x, y=None):
        self.Ax = self.A.solve(x, self.Ax)
        y = self.W.apply(x, y)
        y.scale(-1.0)
        y.axpy(1.0, self.Ax)
        return y

    def solve_hermitian_transpose(self, x, y=None):
        self.ATx = self.A.solve_hermitian_transpose(x, self.ATx)
        y = self.W.apply_hermitian_transpose(x, y)
        y.scale(-1.0)
        y.axpy(1.0, self.ATx)
        return y

    def solve_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv(X.getSizes()[-1])
        Z = self.A.solve_mat(X, Z)
        Y = self.W.apply_mat(X, Y)
        Y.scale(-1.0)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y

    def solve_hermitian_transpose_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv_hermitian_transpose(
                X.getSizes()[-1]
            )
        Z = self.A.solve_hermitian_transpose_mat(X, Z)
        Y = self.W.apply_hermitian_transpose_mat(X, Y)
        Y.scale(-1.0)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y

    def destroy_woodbury_operator(
        self: "LowRankUpdatedLinearOperator",
    ) -> None:
        self.W.destroy() if self.W is not None else None

    def destroy_low_rank_update(self: "LowRankUpdatedLinearOperator") -> None:
        self.L.destroy()

    def destroy_intermediate_vectors(
        self: "LowRankUpdatedLinearOperator",
    ) -> None:
        self.Ax.destroy()
        self.ATx.destroy()

    def destroy(self):
        self.destroy_intermediate_vectors()
        self.destroy_woodbury_operator()
        self.destroy_low_rank_update()
