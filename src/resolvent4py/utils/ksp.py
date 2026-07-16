__all__ = [
    "create_direct_solver",
    "create_gmres_solver",
    "check_solver",
]

import typing
import warnings

from petsc4py import PETSc

from .miscellaneous import petscprint
from .random import generate_random_petsc_vector


def _apply_mumps_options(
    factor_mat: PETSc.Mat,
    icntl: typing.Optional[dict],
    cntl: typing.Optional[dict],
) -> None:
    r"""Write MUMPS ICNTL/CNTL values onto a factor matrix in place."""
    if icntl:
        for k, v in icntl.items():
            factor_mat.setMumpsIcntl(k, v)
    if cntl:
        for k, v in cntl.items():
            factor_mat.setMumpsCntl(k, v)


def _make_gmres_monitor(comm: PETSc.Comm):
    r"""Return a KSP monitor that prints iteration + residual on rank 0."""

    def _monitor(_ksp, its, rnorm):
        petscprint(
            comm, f"GMRES Iteration {its:3d}, Residual Norm = {rnorm:.3e}"
        )

    return _monitor


def create_direct_solver(
    A: PETSc.Mat,
    icntl: typing.Optional[dict[int, int]] = None,
    cntl: typing.Optional[dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Compute an LU factorization of the matrix :math:`A` using MUMPS.

    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param icntl: optional dict mapping MUMPS ICNTL indices to integer values,
        e.g. ``{14: 50, 7: 5, 28: 2, 29: 2, 35: 2}``. Common knobs:

        * ``ICNTL(14)`` workspace headroom in % (default ~35); raise to
          50/100/200 if MUMPS reports ``INFO(1) = -9`` (real workarray too
          small).
        * ``ICNTL(7)`` sequential ordering (5 = METIS).
        * ``ICNTL(28)`` parallel analysis (2 = on, requires PT-Scotch/ParMETIS).
        * ``ICNTL(29)`` parallel ordering (1 = PT-Scotch, 2 = ParMETIS).
        * ``ICNTL(35)`` Block Low-Rank (2 = factor with BLR; pair with
          ``CNTL(7)``).
    :type icntl: Optional[Dict[int, int]]
    :param cntl: optional dict mapping MUMPS CNTL indices to float values,
        e.g. ``{7: 1e-10}`` for the BLR dropping tolerance.
    :type cntl: Optional[Dict[int, float]]

    :return: PETSc KSP solver
    :rtype: PETSc.KSP
    """
    comm = A.getComm()
    if icntl and icntl.get(35, 0) > 0 and comm.getRank() == 0:
        warnings.warn(
            f"create_direct_solver: ICNTL(35)={icntl[35]} enables BLR "
            f"factorization, so the LU solve is inexact and the resulting "
            f"KSP is no longer a true direct solver.  Either tighten "
            f"CNTL(7) and add iterative refinement (ICNTL(10)>0), or use "
            f"this solver as a preconditioner inside an outer GMRES.",
            UserWarning,
            stacklevel=2,
        )

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    _apply_mumps_options(pc.getFactorMatrix(), icntl, cntl)
    pc.setUp()
    ksp.setUp()
    return ksp


def create_gmres_solver(
    A: PETSc.Mat,
    preconditioner: str,
    nblocks: int,
    n_off_diags: typing.Optional[int] = 1,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
    icntl: typing.Optional[dict[int, int]] = None,
    cntl: typing.Optional[dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Create a GMRES solver for :math:`A x = b` with a MUMPS-based
    preconditioner selected by :code:`preconditioner`.

    Two preconditioning strategies are supported:

    * ``"bjacobi"`` --- block-Jacobi preconditioner: :math:`A` is split
      into :code:`nblocks` on-diagonal blocks and each is LU-factorised
      by MUMPS.  Exact when :math:`A` is genuinely block-diagonal and the
      block partition aligns with rank ownership.

    * ``"block_banded"`` --- keep the block-tridiagonal (default) or
      wider block-banded part of :math:`A` and LU-factor that band via
      MUMPS.  The band is extracted with
      :func:`~resolvent4py.utils.matrix.extract_block_banded` using
      :code:`n_off_diags` block off-diagonals on each side of the main
      block-diagonal (0 = block-diagonal, 1 = block-tridiagonal, ...).

    In both cases the outer Krylov iteration is on :math:`A` (the
    preconditioner is passed as the second argument of
    :meth:`~petsc4py.PETSc.KSP.setOperators`), and the convergence test
    uses the unpreconditioned residual norm so that printed residuals
    match :math:`\lVert b - A x_k \rVert`.

    :param A: PETSc matrix iterated on by GMRES
    :type A: PETSc.Mat
    :param preconditioner: which preconditioner to build --- one of
        :code:`"bjacobi"` or :code:`"block_banded"`
    :type preconditioner: str
    :param nblocks: number of on-diagonal blocks in :math:`A`
    :type nblocks: int
    :param n_off_diags: number of block off-diagonals on each side of the
        main block-diagonal to keep when ``preconditioner == "block_banded"``
        (ignored otherwise)
    :type n_off_diags: Optional[int], default is 1
    :param rtol: relative tolerance for GMRES
    :type rtol: Optional[float], default is :math:`10^{-10}`
    :param atol: absolute tolerance for GMRES
    :type atol: Optional[float], default is :math:`10^{-10}`
    :param monitor: :code:`True` to print GMRES residual history to
        terminal; :code:`False` otherwise
    :type monitor: Optional[bool], default is :code:`False`
    :param icntl: MUMPS ICNTL dict; applied uniformly to every sub-block
        factor when ``preconditioner == "bjacobi"`` and to the single
        band-matrix factor when ``preconditioner == "block_banded"``.
        See :func:`.create_direct_solver` for common knobs.
    :type icntl: Optional[Dict[int, int]]
    :param cntl: MUMPS CNTL dict; same scoping as :code:`icntl`
    :type cntl: Optional[Dict[int, float]]

    :return: PETSc KSP solver
    :rtype: PETSc.KSP
    """
    if preconditioner not in ("bjacobi", "block_banded"):
        raise ValueError(
            f"preconditioner must be 'bjacobi' or 'block_banded', "
            f"got {preconditioner!r}"
        )

    comm = A.getComm()
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    if monitor:
        ksp.setMonitor(_make_gmres_monitor(comm))

    if preconditioner == "bjacobi":
        _configure_bjacobi_preconditioner(ksp, A, nblocks, icntl, cntl)
    else:
        _configure_block_banded_preconditioner(
            ksp, A, nblocks, n_off_diags, icntl, cntl,
        )

    return ksp


def _configure_bjacobi_preconditioner(
    ksp: PETSc.KSP,
    A: PETSc.Mat,
    nblocks: int,
    icntl: typing.Optional[dict[int, int]],
    cntl: typing.Optional[dict[int, float]],
) -> None:
    r"""
    Wire up ``ksp`` (already GMRES) with a block-Jacobi + MUMPS
    preconditioner and push the caller's ICNTL/CNTL onto every sub-block
    factor.

    Block-Jacobi is exact when blocks and ranks align in EITHER direction:
    :math:`n_{\text{blocks}} \bmod n_{\text{procs}} = 0` or
    :math:`n_{\text{procs}} \bmod n_{\text{blocks}} = 0`.  We warn on rank
    0 when neither divides the other.

    The ICNTL/CNTL dance below is required because in PETSc 3.24 the
    ``bjacobi -> sub-LU -> MUMPS`` chain does NOT pull the options-DB
    keys ``sub_pc_factor_mat_mumps_icntl_N`` (they show up as
    ``Option left ... source: code`` at end-of-run).  The reliable path
    is to (1) call :meth:`~petsc4py.PETSc.Mat.setMumpsIcntl` on each
    sub-factor's mumps_id struct, then (2) force a numeric refactor via
    :meth:`~petsc4py.PETSc.PC.setReusePreconditioner`\ (False) followed
    by ``sub_pc.setUp()``.

    JOB=1 (analysis) ICNTLs --- ``ICNTL(7)``, ``ICNTL(28)``,
    ``ICNTL(29)`` --- only take effect during the analysis pass, which
    ran during the initial ``pc.setUp()`` with MUMPS defaults.  To pin
    those, build MUMPS with the desired ordering as default or rely on
    ``ICNTL(7)=7`` (auto), which picks METIS when available.
    """
    comm = A.getComm()
    nprocs = comm.getSize()
    misaligned = (nblocks % nprocs != 0) and (nprocs % nblocks != 0)
    if misaligned and comm.getRank() == 0:
        warnings.warn(
            f"create_gmres_solver(preconditioner='bjacobi'): nblocks="
            f"{nblocks} and comm.size={nprocs} are not aligned (neither "
            f"divides the other).  For a block-exact preconditioner you "
            f"need either nblocks = k * nprocs (one rank per block "
            f"group) or nprocs = k * nblocks (one block per rank "
            f"sub-comm).  Otherwise some bjacobi block straddles a rank "
            f"boundary and a matrix that is genuinely block-diagonal "
            f"will not converge in one iteration.",
            UserWarning,
            stacklevel=3,
        )

    ksp.setOperators(A)
    pc = ksp.getPC()
    pc.setType("bjacobi")

    # petsc4py has no direct binding for PCBJacobiSetTotalBlocks, so we
    # go through the options DB.  Prefix scopes the key to this KSP so we
    # do not leak into other solvers created in the same process.
    prefix = f"_r4py_gmres_{id(ksp):x}_"
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    opts[f"{prefix}pc_bjacobi_blocks"] = str(nblocks)
    ksp.setFromOptions()

    try:
        pc.setUp()  # allocates sub-KSPs; runs the default factor (ILU)
        # Reconfigure every sub-block to LU + MUMPS with the caller's
        # ICNTL/CNTL, then force a refactor so MUMPS consumes the values.
        for sk in pc.getBJacobiSubKSP():
            sk.setType("preonly")
            sub_pc = sk.getPC()
            sub_pc.setType("lu")
            sub_pc.setFactorSolverType("mumps")
            _apply_mumps_options(sub_pc.getFactorMatrix(), icntl, cntl)
            sub_pc.setReusePreconditioner(False)
            sub_pc.setUp()
            sub_pc.setReusePreconditioner(True)
        ksp.setUp()
    finally:
        # Drop our prefixed option so later PETSc runs don't see it.
        opts.delValue(f"{prefix}pc_bjacobi_blocks")


def _configure_block_banded_preconditioner(
    ksp: PETSc.KSP,
    A: PETSc.Mat,
    nblocks: int,
    n_off_diags: int,
    icntl: typing.Optional[dict[int, int]],
    cntl: typing.Optional[dict[int, float]],
) -> None:
    r"""
    Wire up ``ksp`` (already GMRES) with a MUMPS LU preconditioner built
    from the block-banded part of :math:`A`.

    :math:`A_b` = :func:`.extract_block_banded` ``(A, nblocks,
    n_off_diags)`` is assembled, then handed to
    :meth:`~petsc4py.PETSc.KSP.setOperators` as the preconditioner
    operator so it is reference-counted by the KSP and stays alive after
    this call returns.
    """
    from .matrix import extract_block_banded

    Aband = extract_block_banded(A, nblocks, n_off_diags)
    ksp.setOperators(A, Aband)
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    _apply_mumps_options(pc.getFactorMatrix(), icntl, cntl)
    pc.setUp()
    ksp.setUp()


def check_solver(A: PETSc.Mat, ksp: PETSc.KSP) -> None:
    r"""
    Sanity-check a KSP by running one solve against a random RHS.

    Raises :class:`ValueError` if the solve did not converge --- covers
    both a direct-solve failure (MUMPS factorisation error surfacing as
    ``KSP_DIVERGED_PC_FAILED``) and a genuine GMRES divergence.  When the
    PC has a MUMPS factor, the error message is enriched with MUMPS
    ``INFOG(1)`` / ``INFOG(2)`` for diagnostics.

    Replaces the older :code:`check_lu_factorization` (direct solvers)
    and :code:`check_gmres_bjacobi_solver` (Krylov solvers) entry points.

    :param A: matrix the KSP was built against
    :type A: PETSc.Mat
    :param ksp: the KSP to exercise
    :type ksp: PETSc.KSP
    """
    b = generate_random_petsc_vector(A.getSizes()[0])
    x = b.duplicate()
    try:
        ksp.solve(b, x)
        reason = ksp.getConvergedReason()
        if reason < 0:
            extra = _mumps_infog_message(ksp)
            raise ValueError(
                f"KSP solve did not converge (ConvergedReason={reason})."
                + extra
            )
    finally:
        x.destroy()
        b.destroy()


def _mumps_infog_message(ksp: PETSc.KSP) -> str:
    r"""Return a diagnostic string with MUMPS INFOG(1)/INFOG(2) if the
    preconditioner has a MUMPS factor; empty string otherwise."""
    try:
        F = ksp.getPC().getFactorMatrix()
        infog1 = F.getMumpsInfog(1)
        infog2 = F.getMumpsInfog(2)
        return f"  MUMPS INFOG(1)={infog1}, INFOG(2)={infog2}"
    except Exception:
        return ""
