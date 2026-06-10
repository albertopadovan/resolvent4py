import typing

from petsc4py import PETSc
from slepc4py import SLEPc

from .linear_operator import LinearOperator
from .matrix import MatrixLinearOperator


class LerayProjectorLinearOperator(LinearOperator):
    r"""
    Leray projector onto the divergence-free velocity subspace,

    .. math::

        P = I - G \, (D G)^{-1} \, D,

    where :math:`D` is the (discrete) divergence, :math:`G` is the gradient,
    and :math:`(DG)^{-1}` is the pressure-Poisson inverse. Both the range and
    the domain of :math:`P` are the velocity-only space (of size
    :code:`n_vel * nblocks` in the harmonic-balanced case). Since
    :math:`P^2 = P`, this operator is a projection.

    The Hermitian transpose

    .. math::

        P^{*} = I - D^{*} \, (D G)^{-*} \, G^{*}

    is also provided and is implemented via the
    :code:`apply_hermitian_transpose` / :code:`solve_hermitian_transpose`
    methods of the constituent operators.

    :param D_op: :class:`.MatrixLinearOperator` wrapping the divergence
        :math:`D` (per block: :code:`n_p_pinned x n_vel`)
    :type D_op: MatrixLinearOperator
    :param DG_op: :class:`.MatrixLinearOperator` wrapping :math:`DG` with a
        KSP attached (per block: :code:`n_p_pinned x n_p_pinned`, typically
        MUMPS-factorized so that its :code:`solve` realizes :math:`(DG)^{-1}`)
    :type DG_op: MatrixLinearOperator
    :param G_op: :class:`.MatrixLinearOperator` wrapping the gradient
        :math:`G` (per block: :code:`n_vel x n_p_pinned`)
    :type G_op: MatrixLinearOperator
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "LerayProjectorLinearOperator",
        D_op: MatrixLinearOperator,
        DG_op: MatrixLinearOperator,
        G_op: MatrixLinearOperator,
        nblocks: typing.Optional[int] = None,
    ) -> None:
        self._D = D_op
        self._DG = DG_op
        self._G = G_op

        # Persistent pressure-space scratch vectors so that apply() does not
        # allocate on every call. D's row space is the pinned-pressure space.
        self._tmp_p1 = D_op.create_left_vector()
        self._tmp_p2 = D_op.create_left_vector()

        # P maps velocity -> velocity; use G's row space (= velocity).
        dimensions = (G_op.get_dimensions()[0], G_op.get_dimensions()[0])
        super().__init__(
            G_op.get_comm(),
            "LerayProjectorLinearOperator",
            dimensions,
            nblocks,
        )

    def apply(
        self: "LerayProjectorLinearOperator",
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        # y = x - G (DG)^{-1} D x
        if y is None:
            y = self.create_left_vector()
        self._D.apply(x, self._tmp_p1)  # vel -> pres
        self._DG.solve(self._tmp_p1, self._tmp_p2)  # pres -> pres
        self._G.apply(self._tmp_p2, y)  # pres -> vel
        y.aypx(-1.0, x)  # y = x - y
        return y

    def apply_hermitian_transpose(
        self: "LerayProjectorLinearOperator",
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        # y = x - D^* (DG)^{-*} G^* x
        if y is None:
            y = self.create_left_vector()
        self._G.apply_hermitian_transpose(x, self._tmp_p1)
        self._DG.solve_hermitian_transpose(self._tmp_p1, self._tmp_p2)
        self._D.apply_hermitian_transpose(self._tmp_p2, y)
        y.aypx(-1.0, x)
        return y

    def apply_mat(
        self: "LerayProjectorLinearOperator",
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        # Column-by-column (the (DG)^{-1} solve sits in the middle).
        ncols = X.getSizes()[-1]
        if Y is None:
            Y = self.create_left_bv(ncols)
        for j in range(ncols):
            xj = X.getColumn(j)
            yj = Y.getColumn(j)
            self.apply(xj, yj)
            X.restoreColumn(j, xj)
            Y.restoreColumn(j, yj)
        return Y

    def apply_hermitian_transpose_mat(
        self: "LerayProjectorLinearOperator",
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        ncols = X.getSizes()[-1]
        if Y is None:
            Y = self.create_left_bv(ncols)
        for j in range(ncols):
            xj = X.getColumn(j)
            yj = Y.getColumn(j)
            self.apply_hermitian_transpose(xj, yj)
            X.restoreColumn(j, xj)
            Y.restoreColumn(j, yj)
        return Y

    def destroy(self: "LerayProjectorLinearOperator") -> None:
        # Only destroy the scratch vectors we created; the constituent
        # operators are user-owned.
        self._tmp_p1.destroy()
        self._tmp_p2.destroy()
