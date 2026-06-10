import typing

import os
import numpy as np
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.bv import bv_add
from ..utils.io import read_coo_matrix, read_harmonic_balanced_matrix
from ..utils.comms import compute_local_size_block_aligned, compute_local_size
from ..utils.ksp import create_mumps_solver, create_gmres_bjacobi_solver
from .linear_operator import LinearOperator
from .matrix import MatrixLinearOperator
from .product import ProductLinearOperator
from .leray_projector import LerayProjectorLinearOperator


class IncompressibleNavierStokesLinearOperator(LinearOperator):
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
    :type woodbury_factors: Optional[Union[Tuple[SLEPc.BV, numpy.ndarray,
        SLEPc.BV], None]], default is None
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "IncompressibleNavierStokesLinearOperator",
        comm: PETSc.Comm,
        fname_A: typing.List[typing.Tuple[str, str, str]] | typing.Tuple[str, str, str],
        fname_M: typing.Tuple[str, str, str],
        sizes: typing.Tuple[int, int, int],
        solver_type: str = 'MUMPS_DIRECT',
        fname_Dm: typing.Optional[typing.Tuple[str, str, str]] = None,
        fname_Gm: typing.Optional[typing.Tuple[str, str, str]] = None,
        freqs: typing.Optional[typing.List[float]] = None,
        icntl: typing.Optional[typing.Dict[int, int]] = None,
        cntl: typing.Optional[typing.Dict[int, float]] = None,
    ) -> None:
        self.solver_type = solver_type
        self.icntl = icntl
        self.cntl = cntl

        # Instantiate every owned attribute up front so that they always
        # exist (as None) regardless of which construction branch runs below.
        # This makes destroy() safe to call unconditionally.
        self.Am = None
        self.Mm = None
        self.Lm = None
        self.L = None
        self.Dm = None
        self.Gm = None
        self.DmGm = None
        self.kspDmGm = None
        self.D = None
        self.G = None
        self.DG = None
        self.P = None
        self.R = None
        self.Rinv = None
        self.sMr = None
        self.sMg = None
        self.ksp = None
        self.ResolventGenOp = None
        self.ResolventOp = None

        nblocks = None

        if freqs is None:
            N_v, N_c = sizes
            N = N_v + N_c
            n = compute_local_size(N)
            n_v = compute_local_size(N_v)
            n_c = compute_local_size(N_c)

            self.Am = read_coo_matrix(fname_A, ((n, N), (n, N)))
            self.Mm = read_coo_matrix(fname_M, ((n, N), (n, N)))
            self.Lm = read_coo_matrix(fname_M, ((n, N), (n_v, N_v)))
            self.L = MatrixLinearOperator(self.Lm)

            if fname_Dm is not None and fname_Gm is not None:
                self.Dm = read_coo_matrix(fname_Dm, ((n_c, N_c), (n_v, N_v)))
                self.Gm = read_coo_matrix(fname_Gm, ((n_v, N_v), (n_c, N_c)))
                self.DmGm = self.Dm.matMult(self.Gm)
                self.kspDmGm = create_mumps_solver(self.DmGm)

                self.D = MatrixLinearOperator(self.Dm)
                self.G = MatrixLinearOperator(self.Gm)
                self.DG = MatrixLinearOperator(self.DmGm, self.kspDmGm)
                self.P = LerayProjectorLinearOperator(
                    self.D, self.DG, self.G
                )

        else:

            nblocks = 2 * (len(freqs) - 1) + 1

        super().__init__(
            comm, "LowRankUpdatedLinearOperator", self.A.get_dimensions(), nblocks
        )

    def update_resolvent_operator(
        self: "IncompressibleNavierStokesLinearOperator",
        s: np.complex128,
    ) -> None:

        # Free the resources created by the previous update. ProductLinearOperator
        # and MatrixLinearOperator now only destroy their own internal scratch
        # (not the shared, persistent self.L), so sMr and ksp are freed here.
        self.ResolventOp.destroy() if self.ResolventOp is not None else None
        self.R.destroy() if self.R is not None else None
        self.ksp.destroy() if self.ksp is not None else None
        self.sMr.destroy() if self.sMr is not None else None

        self.sMr = self.Mm.copy()
        self.sMr.scale(s)
        self.sMr.axpy(-1.0, self.Am)

        if self.solver_type == 'MUMPS_DIRECT':
            self.ksp = create_mumps_solver(self.sMr, self.icntl, self.cntl)
        elif self.solver_type == 'GMRES_BJACOBI':
            self.ksp = create_gmres_bjacobi_solver(
                self.sMr,
                self.get_nblocks(),
                sub_icntl=self.icntl,
                sub_cntl=self.cntl,
            )
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")

        self.R = MatrixLinearOperator(self.sMr, self.ksp, self.get_nblocks())
        Lops = [self.L, self.R, self.L]
        acts = [self.L.apply_hermitian_transpose, self.R.solve, self.L.apply]
        self.ResolventOp = ProductLinearOperator(Lops, acts, self.get_nblocks())

    def update_resolvent_generator(
        self: "IncompressibleNavierStokesLinearOperator",
        s: np.complex128,
    ) -> None:
        
        # Free the resources created by the previous update; see the note in
        # update_resolvent_operator. self.Rinv wraps the previous sMg.
        self.ResolventGenOp.destroy() if self.ResolventGenOp is not None else None
        self.Rinv.destroy() if self.Rinv is not None else None
        self.sMg.destroy() if self.sMg is not None else None

        self.sMg = self.Mm.copy()
        self.sMg.scale(s)
        self.sMg.axpy(-1.0, self.Am)
        # The generator applies sM - A (no inverse), so no KSP is attached.
        self.Rinv = MatrixLinearOperator(self.sMg, None, self.get_nblocks())

        Lops = [self.P, self.L, self.Rinv, self.L, self.P]
        acts = [
            self.P.apply,
            self.L.apply_hermitian_transpose,
            self.Rinv.apply,
            self.L.apply,
            self.P.apply,
        ]
        self.ResolventGenOp = ProductLinearOperator(Lops, acts, self.get_nblocks())
        

    def apply(
        self,
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        if self.ResolventGenOp is None:
            raise ValueError("Resolvent generator not initialized. Call update_resolvent_generator() first.")
        return self.ResolventGenOp.apply(x, y)
    
    def apply_mat(
        self,
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        if self.ResolventGenOp is None:
            raise ValueError("Resolvent generator not initialized. Call update_resolvent_generator() first.")
        return self.ResolventGenOp.apply_mat(X, Y)

    def apply_hermitian_transpose(
        self,
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        if self.ResolventGenOp is None:
            raise ValueError("Resolvent generator not initialized. Call update_resolvent_generator() first.")
        return self.ResolventGenOp.apply_hermitian_transpose(x, y)

    def apply_hermitian_transpose_mat(
        self,
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        if self.ResolventGenOp is None:
            raise ValueError("Resolvent generator not initialized. Call update_resolvent_generator() first.")
        return self.ResolventGenOp.apply_hermitian_transpose_mat(X, Y)
    
    def solve(
        self,
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        if self.ResolventOp is None:
            raise ValueError("Resolvent operator not initialized. Call update_resolvent_operator() first.")
        return self.ResolventOp.apply(x, y)
    
    def solve_mat(
        self,
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        if self.ResolventOp is None:
            raise ValueError("Resolvent operator not initialized. Call update_resolvent_operator() first.")
        return self.ResolventOp.apply_mat(X, Y)
    
    def solve_hermitian_transpose(
        self,
        x: PETSc.Vec,
        y: typing.Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        if self.ResolventOp is None:
            raise ValueError("Resolvent operator not initialized. Call update_resolvent_operator() first.")
        return self.ResolventOp.apply_hermitian_transpose(x, y)
    
    def solve_hermitian_transpose_mat(
        self,
        X: SLEPc.BV,
        Y: typing.Optional[SLEPc.BV] = None,
    ) -> SLEPc.BV:
        if self.ResolventOp is None:
            raise ValueError("Resolvent operator not initialized. Call update_resolvent_operator() first.")
        return self.ResolventOp.apply_hermitian_transpose_mat(X, Y)
    
    def destroy(self) -> None:
        # Everything below is created internally by this operator (the user
        # only passes file names), so we own all of it. Each wrapper operator
        # now destroys only its own internal scratch, so the underlying
        # matrices and KSPs are destroyed explicitly here. petsc4py .destroy()
        # is idempotent, so any overlap is a safe no-op.
        destroy_lst = [
            self.ResolventGenOp,
            self.ResolventOp,
            self.P,
            self.R,
            self.Rinv,
            self.D,
            self.G,
            self.DG,
            self.L,
            self.ksp,
            self.kspDmGm,
            self.sMr,
            self.sMg,
            self.Am,
            self.Mm,
            self.Lm,
            self.Dm,
            self.Gm,
            self.DmGm,
        ]
        for op in destroy_lst:
            op.destroy() if op is not None else None


            