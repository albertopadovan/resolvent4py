import typing

import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.matrix import create_AIJ_identity
from .linear_operator import LinearOperator
from .low_rank import LowRankLinearOperator
from .low_rank_updated import LowRankUpdatedLinearOperator
from .matrix import MatrixLinearOperator


class ProjectionLinearOperator(LinearOperator):
    r"""
    Class for a linear operator of the form

    .. math::

        L = \begin{cases}
        \Phi \left(\Psi^*\Phi\right)^{-1} \Psi^{*} & \text{if } \texttt{complement=False} \\ 
        I - \Phi \left(\Psi^*\Phi\right)^{-1} \Psi^{*} & \text{otherwise}
        \end{cases}

    where :math:`\Phi` and :math:`\Psi` are tall and skinny matrices of size
    :math:`N\times r` stored as SLEPc BVs. (It is easy to check that :math:`L` 
    is a projection since :math:`L^2 = L`.)
    
    :param Phi: tall and skinny matrix
    :type Phi: SLEPc.BV
    :param Psi: tall and skinny matrix
    :type Psi: SLEPc.BV
    :param complement: see definition of :math:`L`
    :type complement: Optional[bool], default if False
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "ProjectionLinearOperator",
        Phi: SLEPc.BV,
        Psi: SLEPc.BV,
        complement: typing.Optional[bool] = False,
        nblocks: typing.Optional[int] = None,
    ) -> None:
        comm = Phi.getComm()
        ncolsPhi = Phi.getSizes()[-1]
        ncolsPsi = Psi.getSizes()[-1]
        if ncolsPhi != ncolsPsi:
            raise ValueError(
                "Phi and Psi should have the same number of columns."
            )
        dimensions = (Phi.getSizes()[0], Psi.getSizes()[0])
        Smat = Phi.dot(Psi)
        Sig = sp.linalg.inv(Smat.getDenseArray())
        Smat.destroy()

        self.complement = complement
        self.Id = None
        self.Idop = None
        if self.complement:
            self.Id = create_AIJ_identity(comm, dimensions)
            self.Idop = MatrixLinearOperator(self.Id, None, nblocks)
            self.L = LowRankUpdatedLinearOperator(
                self.Idop, Phi, -Sig, Psi, None, nblocks
            )
        else:
            self.L = LowRankLinearOperator(Phi, Sig, Psi, nblocks)

        super().__init__(comm, "ProjectionLinearOperator", dimensions, nblocks)

    def apply(self, x, y=None):
        return self.L.apply(x, y)

    def apply_hermitian_transpose(self, x, y=None):
        return self.L.apply_hermitian_transpose(x, y)

    def apply_mat(self, X, Y=None):
        return self.L.apply_mat(X, Y)

    def apply_hermitian_transpose_mat(self, X, Y=None):
        return self.L.apply_hermitian_transpose_mat(X, Y)

    def destroy(self):
        # Phi and Psi are user-supplied. self.L, and (for the complement) the
        # identity matrix self.Id and its wrapper self.Idop, are created
        # internally here, so they are ours to destroy. self.Idop.destroy()
        # frees only its internal Hermitian transpose, so self.Id must be
        # destroyed explicitly.
        self.L.destroy()
        if self.complement:
            self.Idop.destroy()
            self.Id.destroy()
