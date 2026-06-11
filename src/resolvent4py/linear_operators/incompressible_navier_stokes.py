from __future__ import annotations

import typing

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.io import read_coo_matrix, read_harmonic_balanced_matrix
from ..utils.comms import compute_local_size_block_aligned, compute_local_size
from ..utils.ksp import create_mumps_solver, create_gmres_bjacobi_solver
from .linear_operator import LinearOperator
from .matrix import MatrixLinearOperator
from .product import ProductLinearOperator
from .leray_projector import LerayProjectorLinearOperator


class IncompressibleNavierStokesLinearOperator(LinearOperator):
    r"""
    Resolvent operator for the divergence-constrained, linearized
    incompressible Navier-Stokes equations, assembled from COO matrices read
    from file. The system is written in descriptor form

    .. math::

        M \frac{dq}{dt} = A\, q,

    where :math:`q = (u, p)` stacks the :math:`N_v` velocity and :math:`N_c`
    pressure/constraint degrees of freedom (velocity DOFs first), :math:`A` is
    the linearized operator (Jacobian) and :math:`M` is the (singular) mass
    matrix, nonzero only on the velocity block.

    Let :math:`L` be the mass matrix restricted to its velocity columns -- so
    :math:`L` embeds the velocity space into the full state and :math:`L^*`
    restricts it back -- and let

    .. math::

        P = I - G\,(DG)^{-1} D

    be the Leray projector onto the divergence-free velocity subspace, built
    from the divergence :math:`D` and gradient :math:`G`. For a complex shift
    :math:`s`, this class exposes two velocity-space operators:

    .. math::

        \texttt{apply}:\; P\, L^* (sM - A)\, L\, P, \qquad
        \texttt{solve}:\; L^* (sM - A)^{-1} L.

    :meth:`apply` evaluates the (divergence-free-projected) resolvent
    *generator* and :meth:`solve` evaluates the *resolvent* itself; both map the
    velocity subspace to itself. Use :meth:`update_resolvent_generator` and
    :meth:`update_resolvent_operator` to retarget a different shift :math:`s`
    without re-reading the matrices.

    If :code:`freqs` is given, the operator is assembled in harmonic-balanced
    form: :math:`A`, :math:`M`, :math:`D` and :math:`G` are read as the
    non-negative Fourier coefficients of a time-periodic base flow and expanded
    into block-Toeplitz matrices with
    :code:`nblocks = 2 * (len(freqs) - 1) + 1` blocks.

    .. note::

        The matrices are read in COO format. Each :code:`fname_*` argument is a
        :code:`(rows, cols, vals)` triplet of file names in the steady case, or
        a list of such triplets (one per non-negative Fourier coefficient
        :math:`A_0, A_1, \ldots`) in the harmonic-balanced case.

    :param comm: MPI communicator (:code:`PETSc.COMM_WORLD`)
    :type comm: PETSc.Comm
    :param s: complex shift at which the resolvent generator and operator are
        initially built
    :type s: numpy.complex128
    :param fname_A: COO file-name triplet(s) for :math:`A` -- a single triplet,
        or a list of triplets in the harmonic-balanced case
    :type fname_A: Union[Tuple[str, str, str], List[Tuple[str, str, str]]]
    :param fname_M: COO file-name triplet(s) for the mass matrix :math:`M`
    :type fname_M: Union[Tuple[str, str, str], List[Tuple[str, str, str]]]
    :param sizes: :code:`(N_v, N_c)`, the number of velocity and
        pressure/constraint DOFs per block
    :type sizes: Tuple[int, int]
    :param solver_type: solver used for :math:`(sM - A)`, either
        :code:`"MUMPS_DIRECT"` (direct LU) or :code:`"GMRES_BJACOBI"`
        (block-Jacobi-preconditioned GMRES)
    :type solver_type: Optional[str], default is :code:`"MUMPS_DIRECT"`
    :param fname_Dm: COO file-name triplet(s) for the divergence :math:`D`.
        Required, together with :code:`fname_Gm`, to build the Leray projector
        :math:`P` (and hence the generator :meth:`apply`)
    :type fname_Dm: Optional[Union[Tuple[str, str, str],
        List[Tuple[str, str, str]]]], default is None
    :param fname_Gm: COO file-name triplet(s) for the gradient :math:`G`
    :type fname_Gm: Optional[Union[Tuple[str, str, str],
        List[Tuple[str, str, str]]]], default is None
    :param freqs: non-negative base-flow frequencies for the harmonic-balanced
        case; if :code:`None`, the steady operator is built
    :type freqs: Optional[List[float]], default is None
    :param icntl: optional MUMPS ICNTL settings forwarded to the
        :math:`(sM - A)` solver
    :type icntl: Optional[Dict[int, int]], default is None
    :param cntl: optional MUMPS CNTL settings forwarded to the
        :math:`(sM - A)` solver
    :type cntl: Optional[Dict[int, float]], default is None
    """

    def __init__(
        self: "IncompressibleNavierStokesLinearOperator",
        comm: PETSc.Comm,
        s: np.complex128,
        fname_A: typing.List[typing.Tuple[str, str, str]] | typing.Tuple[str, str, str],
        fname_M: typing.Tuple[str, str, str],
        sizes: typing.Tuple[int, int],
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
        self.s = s

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
        N_v, N_c = sizes
        N = N_v + N_c

        if freqs is None:
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
            # nl/nvl/ncl are the per-block local sizes; Nl/Nvl/Ncl are the
            # full-matrix local sizes. They coincide for size > 1 (block-aligned
            # distribution) but differ on a single rank, where the full local
            # size equals the global size — so the full sizes must use Nl/etc.
            nl, Nl = compute_local_size_block_aligned(N, N * nblocks)
            nvl, Nvl = compute_local_size_block_aligned(N_v, N_v * nblocks)
            ncl, Ncl = compute_local_size_block_aligned(N_c, N_c * nblocks)

            blk_size = ((nl, N), (nl, N))
            glb_size = ((Nl, N * nblocks), (Nl, N * nblocks))
            self.Am = read_harmonic_balanced_matrix(fname_A, True, blk_size, glb_size)
            self.Mm = read_harmonic_balanced_matrix(fname_M, True, blk_size, glb_size)

            # Mass matrix restricted to velocity columns: full-space rows,
            # velocity columns (per harmonic block).
            blk_size_L = ((nl, N), (nvl, N_v))
            glb_size_L = ((Nl, N * nblocks), (Nvl, N_v * nblocks))
            self.Lm = read_harmonic_balanced_matrix(
                fname_M, True, blk_size_L, glb_size_L
            )
            self.L = MatrixLinearOperator(self.Lm, None, nblocks)

            if fname_Dm is not None and fname_Gm is not None:
                blk_size_D = ((ncl, N_c), (nvl, N_v))
                glb_size_D = ((Ncl, N_c * nblocks), (Nvl, N_v * nblocks))
                blk_size_G = ((nvl, N_v), (ncl, N_c))
                glb_size_G = ((Nvl, N_v * nblocks), (Ncl, N_c * nblocks))
                self.Dm = read_harmonic_balanced_matrix(
                    fname_Dm, True, blk_size_D, glb_size_D
                )
                self.Gm = read_harmonic_balanced_matrix(
                    fname_Gm, True, blk_size_G, glb_size_G
                )
                self.DmGm = self.Dm.matMult(self.Gm)
                self.kspDmGm = create_mumps_solver(self.DmGm)

                self.D = MatrixLinearOperator(self.Dm, None, nblocks)
                self.G = MatrixLinearOperator(self.Gm, None, nblocks)
                self.DG = MatrixLinearOperator(self.DmGm, self.kspDmGm, nblocks)
                self.P = LerayProjectorLinearOperator(
                    self.D, self.DG, self.G, nblocks
                )

        dims = (self.L.get_dimensions()[1], self.L.get_dimensions()[1])
        super().__init__(
            comm, "IncompressibleNavierStokesLinearOperator", dims, nblocks
        )
        # The real / cc flags are derived from s by the check_*() overrides, so
        # the generator is not needed during super().__init__. Build it now so
        # the operator is ready to apply() after construction.
        self.update_resolvent_generator(s)
        self.update_resolvent_operator(s)

    def check_if_real_valued(
        self: "IncompressibleNavierStokesLinearOperator",
    ) -> bool:
        r"""
        The non-harmonic-balanced operator is assembled from real matrices, so
        it is real-valued iff the shift :math:`s` is real (then :math:`sM - A`
        is real). The harmonic-balanced operator is complex-valued — it carries
        complex-conjugate block structure instead. Overridden to avoid the
        base-class probe, which would call :meth:`apply`.
        """
        if self.get_nblocks() is None:
            return bool(np.imag(self.s) == 0.0)
        return False

    def check_if_complex_conjugate_structure(
        self: "IncompressibleNavierStokesLinearOperator",
    ) -> typing.Optional[bool]:
        r"""
        Complex-conjugate block structure is only defined for the
        harmonic-balanced operator, so this returns :code:`None` for the
        non-harmonic-balanced case. When block-structured, the operator has
        complex-conjugate structure iff the shift :math:`s` is real.
        """
        if self.get_nblocks() is None:
            return None
        return bool(np.imag(self.s) == 0.0)
    

    def update_resolvent_operator(
        self: "IncompressibleNavierStokesLinearOperator",
        s: np.complex128,
    ) -> None:
        r"""
        (Re)build the resolvent operator at shift :math:`s`,

        .. math::

            \texttt{solve}(x) = L^* (sM - A)^{-1} L\, x,

        factorizing :math:`sM - A` with the configured :code:`solver_type` and
        wiring up :meth:`solve`, :meth:`solve_mat` and their
        Hermitian-transpose variants. Any operator built by a previous call is
        destroyed first, so this may be called repeatedly to sweep over
        :math:`s`.

        :param s: complex shift
        :type s: numpy.complex128
        """

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
        r"""
        (Re)build the resolvent generator at shift :math:`s`,

        .. math::

            \texttt{apply}(x) = P\, L^* (sM - A)\, L\, P\, x,

        the divergence-free-projected forward operator. No factorization is
        needed (:math:`sM - A` is only applied), so the Leray projector
        :math:`P` is required. Wires up :meth:`apply`, :meth:`apply_mat` and
        their Hermitian-transpose variants; any generator built by a previous
        call is destroyed first.

        :param s: complex shift
        :type s: numpy.complex128
        """

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
        r"""
        Apply the resolvent generator :math:`P L^* (sM - A) L P` to a
        velocity-space vector :math:`x`. Requires
        :meth:`update_resolvent_generator` to have been called (done at
        construction); raises :class:`ValueError` otherwise.

        :param x: velocity-space PETSc vector
        :type x: PETSc.Vec
        :param y: optional output vector
        :type y: Optional[PETSc.Vec], default is None
        :rtype: PETSc.Vec
        """
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
        r"""
        Apply the resolvent :math:`L^* (sM - A)^{-1} L` to a velocity-space
        vector :math:`x`. Requires :meth:`update_resolvent_operator` to have
        been called (done at construction); raises :class:`ValueError`
        otherwise.

        :param x: velocity-space PETSc vector
        :type x: PETSc.Vec
        :param y: optional output vector
        :type y: Optional[PETSc.Vec], default is None
        :rtype: PETSc.Vec
        """
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
        r"""
        Destroy every PETSc/SLEPc object and sub-operator owned by this
        operator. The user only passes file names, so this operator owns and
        frees all of the assembled matrices, KSPs and wrapper operators.
        """
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


            