import typing

import numpy as np
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.bv import bv_add, reshape_bv_into_harmonic_balanced_vector
from ..utils.matrix import extract_matrix_block
from ..utils.ksp import create_mumps_solver, check_lu_factorization
from ..utils.vector import reshape_harmonic_balanced_vector_into_bv
from .linear_operator import LinearOperator


class SuperoptimalBlockCirculantLinearOperator(LinearOperator):
    r"""
    Superoptimal block-circulant preconditioner for the quasi-block-Toeplitz
    harmonic-balance system

    .. math::

        T_{k,j} = (s + ik\omega) M \delta_{k,j} - A_{k-j}.

    The preconditioner :math:`C^{-1}` minimises
    :math:`\|I - C^{-1}T\|_F^2` over all block-circulant matrices
    :math:`C`.  Its application (Algorithm 1) consists of a block-FFT,
    :math:`n` independent :math:`N \times N` solves and sparse mat-vecs,
    and a block-IFFT.

    Call :meth:`setup` after construction to compute the DFT-domain
    blocks :math:`G_{kk}`, the row block-norm matrices :math:`R_k`,
    and their factorisations.

    :param T: linear operator wrapping the :math:`nN \times nN`
        harmonic-balance system
    :type T: LinearOperator
    :param A: assembled :math:`nN \times nN` block-Toeplitz PETSc sparse
        matrix with blocks :math:`\hat{A}_{k,j} = A_{k-j}`
    :type A: PETSc.Mat
    :param M: assembled :math:`nN \times nN` block-Toeplitz PETSc sparse
        mass matrix (only the zeroth block :math:`M_0` is used)
    :type M: PETSc.Mat
    :param omega: fundamental frequency
    :type omega: float
    :param s: Laplace-domain shift
    :type s: complex
    """

    def __init__(
        self: "SuperoptimalBlockCirculantLinearOperator",
        T: LinearOperator,
        A: PETSc.Mat,
        M: PETSc.Mat,
        omega: float,
        s: complex,
    ) -> None:
        comm = T.get_comm()
        size = T.get_dimensions()
        nblk = T.get_nblocks()
        self.T = T
        self.A = A
        self.M = M
        self.s = s
        self.omega = omega
        super().__init__(
            comm, "SuperoptimalBlockCirculantLinearOperator", size, nblk
        )

    def setup(self) -> None:
        r"""
        One-time setup (Algorithm 2) for the superoptimal block-circulant
        preconditioner.

        Notation (consistent with the codebase):

        .. math::

            T_{k,j} = (s + ik\omega) M \delta_{k,j} - A_{k-j}

        The DFT-domain diagonal blocks and row block-norm matrices are:

        .. math::

            G_{kk} = (s + ik\omega) M - \sum_{l=-m}^{m} A_l e^{2\pi ikl/(2m + 1)},\quad

        .. math::

            R_k = \sum_{j=-2m}^{2m} B_j e^{2\pi i jk / (2m + 1)},\quad
            B_j = \frac{1}{2m + 1} \sum_{p=-m}^m (TT^*)_{p,p+j}
        
        Uses the attributes ``self.T``, ``self.A``, ``self.M``,
        ``self.omega``, and ``self.s`` set in :meth:`__init__`.
        """
        nblocks = self.get_nblocks()
        nN = self.get_dimensions()[0][-1]
        N = nN // nblocks

        # ------------------------------------------------------------------
        # Extract A_j and M_0 blocks from the block-Toeplitz matrices.
        # _extract_toeplitz_blocks returns nblocks PETSc.Mat objects
        # ordered by block-row index (0 .. nblocks-1).
        # ------------------------------------------------------------------
        Ak_list = _extract_toeplitz_blocks(self.A, nblocks)
        Mlst_petsc = _extract_toeplitz_blocks(self.M, nblocks)
        m = (nblocks - 1) // 2
        M0 = Mlst_petsc[m].copy()
        for obj in Mlst_petsc:
            obj.destroy()

        # ------------------------------------------------------------------
        # Assemble G_{kk} as described in the docstrings
        # ------------------------------------------------------------------
        Gkk_list = []
        for k in range(-m, m + 1):
            Gkk = M0.copy()
            Gkk.scale(self.s + 1j * k * self.omega)
            for l in range(-m, m + 1):
                Gkk.axpy(-np.exp(2j * np.pi * l * k / nblocks), Ak_list[l + m])
            Gkk_list.append(Gkk.copy())
            Gkk.destroy()
        self.Gkk_list = Gkk_list

        Gkk_ksp_list = []
        for k in range(nblocks):
            ksp_gk = create_mumps_solver(Gkk_list[k])
            check_lu_factorization(Gkk_list[k], ksp_gk)
            Gkk_ksp_list.append(ksp_gk)
        self.Gkk_ksp_list = Gkk_ksp_list

        # ------------------------------------------------------------------
        # Assemble R_{k} as described in the docstrings. To do that, we 
        # first need to assemble B_{k}
        # ------------------------------------------------------------------
        TH = self.T.A.copy()
        TH.hermitianTranspose()
        TTH = self.T.A.matMult(TH)
        TH.destroy()
        Bk_list = []
        for k in range (-2 * m, 2 * m + 1):
            Bk_initialized = False
            for p in range (-m, m + 1):
                if -m <= p + k <= m:
                    Mpk = extract_matrix_block(TTH, nblocks, rowblock=p + m, colblock=p + k + m)
                    if not Bk_initialized:
                        Bk = Mpk.copy()
                        Bk_initialized = True
                    else:
                        Bk.axpy(1.0, Mpk)
                    Mpk.destroy()
            if not Bk_initialized:
                Bk = extract_matrix_block(TTH, nblocks, rowblock=0, colblock=0)
                Bk.scale(0.0)
            Bk.scale(1.0 / nblocks)
            Bk_list.append(Bk.copy())
            Bk.destroy()
        TTH.destroy()

        Rk_list = []
        for k in range (-m, m + 1):
            for j in range(-2 * m, 2 * m + 1):
                scaling = np.exp(2j * np.pi * k * j / nblocks)
                if j == - 2 * m:
                    Rk = Bk_list[j + 2 * m].copy()
                    Rk.scale(scaling)
                else:
                    Rk.axpy(scaling, Bk_list[j + 2 * m])
            Rk_list.append(Rk.copy())
            Rk.destroy()
        
        for obj in Bk_list:
            obj.destroy()
        self.Rk_list = Rk_list
        # ------------------------------------------------------------------
        # Factor R_k via PETSc KSP (MUMPS LU).
        # Each entry is a PETSc.KSP ready to solve R_k x = b.
        # ------------------------------------------------------------------
        Rk_ksp_list = []
        for k in range(nblocks):
            ksp_rk = create_mumps_solver(Rk_list[k])
            check_lu_factorization(Rk_list[k], ksp_rk)
            Rk_ksp_list.append(ksp_rk)
        self.Rk_ksp_list = Rk_ksp_list
        
    def apply(self, x: PETSc.Vec, y: PETSc.Vec=None) -> PETSc.Vec:
        y = self.create_left_vector() if y is None else y
        nblocks = self.get_nblocks()
        xBV = reshape_harmonic_balanced_vector_into_bv(x, nblocks)
        xBV = _fft(xBV, "bwd")
        yBV = xBV.copy()
        wi = y.duplicate()
        for i in range(nblocks):
            xi = xBV.getColumn(i)
            self.Gkk_ksp_list[i].solve(xi, wi)
            xBV.restoreColumn(i, xi)
            yi = yBV.getColumn(i)
            self.Rk_list[i].mult(wi, yi)
            yBV.restoreColumn(i, yi)
        yBV = _fft(yBV, "fwd")
        y = reshape_bv_into_harmonic_balanced_vector(yBV, y)
        objs = [wi, xBV, yBV]
        for obj in objs:
            obj.destroy()
        return y

    def apply_hermitian_transpose(
        self, x: PETSc.Vec, y: PETSc.Vec = None
    ) -> PETSc.Vec:
        y = self.create_right_vector() if y is None else y
        nblocks = self.get_nblocks()
        xBV = reshape_harmonic_balanced_vector_into_bv(x, nblocks)
        xBV = _fft(xBV, "bwd")
        yBV = xBV.copy()
        wi = y.duplicate()
        for i in range(nblocks):
            # R_k^* xi -> wi (R_k = R_k^*)
            xi = xBV.getColumn(i)
            self.Rk_list[i].mult(xi, wi)
            xBV.restoreColumn(i, xi)
            # G_{kk}^{-*} wi -> yi  (solve G_{kk}^H yi = wi)
            yi = yBV.getColumn(i)
            wi.conjugate()
            self.Gkk_ksp_list[i].solveTranspose(wi, yi)
            yi.conjugate()
            wi.conjugate()
            yBV.restoreColumn(i, yi)
        yBV = _fft(yBV, "fwd")
        y = reshape_bv_into_harmonic_balanced_vector(yBV, y)
        objs = [wi, xBV, yBV]
        for obj in objs:
            obj.destroy()
        return y

    def apply_mat(
        self, X: SLEPc.BV, Y: SLEPc.BV = None
    ) -> SLEPc.BV:
        ncols = X.getSizes()[-1]
        if Y is None:
            Y = self.create_left_bv(ncols)
        y = self.create_left_vector()
        for j in range(ncols):
            x = X.getColumn(j)
            y = self.apply(x, y)
            X.restoreColumn(j, x)
            Y.insertVec(j, y)
        y.destroy()
        return Y

    def apply_hermitian_transpose_mat(
        self, X: SLEPc.BV, Y: SLEPc.BV = None
    ) -> SLEPc.BV:
        ncols = X.getSizes()[-1]
        if Y is None:
            Y = self.create_right_bv(ncols)
        y = self.create_right_vector()
        for j in range(ncols):
            x = X.getColumn(j)
            y = self.apply_hermitian_transpose(x, y)
            X.restoreColumn(j, x)
            Y.insertVec(j, y)
        y.destroy()
        return Y


def _fft(xBV: SLEPc.BV, direction: str = "fwd") -> SLEPc.BV:
    r"""
    Apply the block-FFT or block-IFFT in-place along the harmonic index
    of a SLEPc BV.

    The BV has shape :math:`n \times (2m+1)` with columns ordered as
    :math:`[v_{-m}, \ldots, v_0, \ldots, v_m]`.  Since
    :func:`scipy.fft.fft` expects standard DFT ordering
    :math:`[0, \ldots, 2m]`, :func:`scipy.fft.ifftshift` is applied
    before the transform and :func:`scipy.fft.fftshift` afterwards.

    :param xBV: SLEPc BV of shape :math:`n \times (2m+1)`, modified
        in-place
    :type xBV: SLEPc.BV
    :param direction: ``"fwd"`` for forward FFT, ``"bwd"`` for inverse FFT
    :type direction: str
    """
    fft_func = sp.fft.fft if direction == "fwd" else sp.fft.ifft

    xMat = xBV.getMat()
    xArr = xMat.getDenseArray()
    xArr[:] = sp.fft.fftshift(
        fft_func(sp.fft.ifftshift(xArr, axes=1), axis=1, norm="ortho"),
        axes=1,
    )
    xBV.restoreMat(xMat)
    return xBV
    

def _extract_toeplitz_blocks(Ahat: PETSc.Mat, nblocks: int):
    r"""
    Extract the Fourier coefficient blocks :math:`A_{-m}, \ldots, A_{m}`
    from the harmonic-balanced operator :math:`\hat{A}` defined as
    :math:`\hat{A}_{k,j} = A_{k-j}`.

    The middle block-column (block-column index :math:`m`) is extracted
    via :math:`\hat{A} \hat{I}`, where :math:`\hat{I}` is an
    :math:`nN \times N` selector with :math:`I_N` at block-row
    :math:`m`.  Individual :math:`N \times N` blocks are then
    isolated by left-multiplying the Hermitian transpose of the result
    by per-block selectors.

    :param Ahat: assembled :math:`nN \times nN` PETSc sparse matrix
        with block-Toeplitz structure :math:`\hat{A}_{k,j} = A_{k-j}`
    :type Ahat: PETSc.Mat
    :param nblocks: number of harmonic blocks (:math:`2 m + 1`)
    :type nblocks: int

    :return: list of length ``nblocks`` (:math:`2 m + 1`) of
        PETSc AIJ matrices (N x N),
        :math:`[A_{-m}, \ldots, A_{-1}, A_0, A_{1}, \ldots, A_{m}]`.
    :rtype: list[PETSc.Mat]
    """
    return [
        extract_matrix_block(Ahat, nblocks, rowblock=i, colblock=(nblocks - 1) // 2)
        for i in range (nblocks)
    ]