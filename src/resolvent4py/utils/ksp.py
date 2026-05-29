__all__ = [
    "create_mumps_solver",
    "check_lu_factorization",
    "create_gmres_bjacobi_solver",
    "check_gmres_bjacobi_solver",
    "create_gmres_epslu_solver",
    "create_gmres_block_banded_solver",
]

import typing
import warnings

import numpy as np
from petsc4py import PETSc

from .miscellaneous import petscprint
from .random import generate_random_petsc_vector


def create_mumps_solver(
    A: PETSc.Mat,
    icntl: typing.Optional[typing.Dict[int, int]] = None,
    cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Compute an LU factorization of the matrix A using MUMPS.

    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param icntl: optional dict mapping MUMPS ICNTL indices to integer values,
        e.g. ``{14: 50, 7: 5, 28: 2, 29: 2, 35: 2}``. Common knobs:

        * ``ICNTL(14)`` workspace headroom in % (default ~35); raise to 50/100/200
          if MUMPS reports ``INFO(1) = -9`` (real workarray too small).
        * ``ICNTL(7)`` sequential ordering (5 = METIS).
        * ``ICNTL(28)`` parallel analysis (2 = on, requires PT-Scotch/ParMETIS).
        * ``ICNTL(29)`` parallel ordering (1 = PT-Scotch, 2 = ParMETIS).
        * ``ICNTL(35)`` Block Low-Rank (2 = factor with BLR; pair with ``CNTL(7)``).
    :type icntl: Optional[Dict[int, int]]
    :param cntl: optional dict mapping MUMPS CNTL indices to float values,
        e.g. ``{7: 1e-10}`` for the BLR dropping tolerance.
    :type cntl: Optional[Dict[int, float]]

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """
    comm = A.getComm()
    if icntl and icntl.get(35, 0) > 0 and comm.getRank() == 0:
        print("\n")
        warnings.warn(
            f"create_mumps_solver: ICNTL(35)={icntl[35]} enables BLR "
            f"factorization, so the LU solve is inexact and the "
            f"resulting KSP is no longer a true direct solver.  Either "
            f"tighten CNTL(7) and add iterative refinement (ICNTL(10)>0), "
            f"or use this solver as a preconditioner inside an outer "
            f"GMRES.",
            UserWarning,
            stacklevel=2,
        )
        print("\n")

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    if icntl or cntl:
        F = pc.getFactorMatrix()
        if icntl:
            for k, v in icntl.items():
                F.setMumpsIcntl(k, v)
        if cntl:
            for k, v in cntl.items():
                F.setMumpsCntl(k, v)
    pc.setUp()
    ksp.setUp()
    return ksp


def check_lu_factorization(A: PETSc.Mat, ksp: PETSc.KSP) -> None:
    r"""
    Check that the LU factorization computed in :func:`.create_mumps_solver`
    has succeeded.

    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param ksp: PETSc KSP solver
    :type ksp: PETSc.KSP
    """
    sizes = A.getSizes()[0]
    b = generate_random_petsc_vector(sizes)
    x = b.duplicate()
    ksp.solve(b, x)
    pc = ksp.getPC()
    Mat = pc.getFactorMatrix()
    Infog1 = Mat.getMumpsInfog(1)
    Infog2 = Mat.getMumpsInfog(2)
    if Infog1 != 0:
        raise ValueError(
            f"MUMPS factorization failed with INFO(1) = {Infog1}  "
            f"and INFO(2) = {Infog2}'"
        )
    x.destroy()
    b.destroy()


def create_gmres_bjacobi_solver(
    A: PETSc.Mat,
    nblocks: int,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
    sub_icntl: typing.Optional[typing.Dict[int, int]] = None,
    sub_cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Create GMRES solver with block-jacobi preconditioner.

    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param nblocks: number of blocks for the block jacobi preconditioner
    :type nblocks: int
    :param rtol: relative tolerance for GMRES
    :type rtol: Optional[float], default is :math:`10^{-10}`
    :param atol: absolute tolerance for GMRES
    :type atol: Optional[float], default is :math:`10^{-10}`
    :param monitor: :code:`True` to monitor convergence and print residual
        history to terminal. :code:`False` otherwise
    :type monitor: Optional[bool], default is :code:`False`
    :param sub_icntl: optional dict mapping MUMPS ICNTL indices to integer
        values applied uniformly to every bjacobi sub-block (e.g.
        ``{35: 2}`` to enable Block Low-Rank). See
        :func:`.create_mumps_solver` for common knobs.
    :type sub_icntl: Optional[Dict[int, int]]
    :param sub_cntl: optional dict mapping MUMPS CNTL indices to float values
        applied uniformly to every bjacobi sub-block (e.g. ``{7: 1e-10}``
        for the BLR dropping tolerance).
    :type sub_cntl: Optional[Dict[int, float]]

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """

    comm = A.getComm()
    nprocs = comm.getSize()
    # Block-Jacobi is exact when blocks and ranks align in EITHER direction:
    #   (a) nblocks % nprocs == 0  → each rank owns nblocks/nprocs full blocks
    #   (b) nprocs  % nblocks == 0 → each block sits on a sub-comm of size
    #                                nprocs/nblocks ranks (PETSc subcomms).
    # Only warn when neither divides the other.
    mis_aligned = (nblocks % nprocs != 0) and (nprocs % nblocks != 0)
    if mis_aligned and comm.getRank() == 0:
        print("\n")
        warnings.warn(
            f"create_gmres_bjacobi_solver: nblocks={nblocks} and "
            f"comm.size={nprocs} are not aligned (neither divides the "
            f"other).  For a block-exact preconditioner you need either "
            f"nblocks = k * nprocs (one rank per block group) OR "
            f"nprocs = k * nblocks (one block per rank sub-comm).  "
            f"Otherwise some bjacobi block straddles a rank boundary "
            f"and a matrix that is genuinely block-diagonal will not "
            f"converge in one iteration.",
            UserWarning,
            stacklevel=2,
        )
        print("\n")

    monitor_fun = None
    if monitor:

        def monitor_fun(ksp, its, rnorm):
            string = f"GMRES Iteration {its:3d}, Residual Norm = {rnorm:.3e}"
            petscprint(comm, string)

    opts = PETSc.Options()
    opts["pc_type"] = "bjacobi"
    opts["pc_bjacobi_blocks"] = nblocks
    opts["sub_ksp_type"] = "preonly"
    opts["sub_pc_type"] = "lu"
    opts["sub_pc_factor_mat_solver_type"] = "mumps"

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setMonitor(monitor_fun)
    pc = ksp.getPC()
    pc.setFromOptions()
    pc.setUp()
    ksp.setUp()

    if sub_icntl or sub_cntl:
        sub_ksps = pc.getBJacobiSubKSP()
        for sk in sub_ksps:
            F = sk.getPC().getFactorMatrix()
            if sub_icntl:
                for k, v in sub_icntl.items():
                    F.setMumpsIcntl(k, v)
            if sub_cntl:
                for k, v in sub_cntl.items():
                    F.setMumpsCntl(k, v)

    return ksp


def create_gmres_epslu_solver(
    A: PETSc.Mat,
    Aeps: PETSc.Mat,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
    icntl: typing.Optional[typing.Dict[int, int]] = None,
    cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Create a GMRES solver for :math:`A x = b` whose left preconditioner is
    the MUMPS LU factorization of an auxiliary matrix :math:`A_\epsilon`,
    typically a regularized perturbation of :math:`A` (for example
    :math:`A_\epsilon = A + \epsilon P` with :math:`P` a diagonal indicator
    that breaks an ill-conditioned saddle-point structure of :math:`A`).
    Internally calls ``ksp.setOperators(A, Aeps)`` so the outer Krylov
    iteration is on :math:`A` while the preconditioner is built from
    :math:`A_\epsilon`.

    The convergence test uses the unpreconditioned residual norm so that
    printed residuals match :math:`\|b - A x_k\|`.

    :param A: PETSc matrix iterated on by GMRES (the "true" operator).
    :type A: PETSc.Mat
    :param Aeps: PETSc matrix whose MUMPS LU factor is used as the
        preconditioner.  Must share size and parallel layout with
        :code:`A`.
    :type Aeps: PETSc.Mat
    :param rtol: relative tolerance for GMRES
    :type rtol: Optional[float], default :math:`10^{-10}`
    :param atol: absolute tolerance for GMRES
    :type atol: Optional[float], default :math:`10^{-10}`
    :param monitor: :code:`True` to monitor convergence and print residual
        history to terminal. :code:`False` otherwise
    :type monitor: Optional[bool], default :code:`False`
    :param icntl: optional dict mapping MUMPS ICNTL indices to integer
        values for the LU of :math:`A_\epsilon`.  See
        :func:`.create_mumps_solver` for common knobs.
    :type icntl: Optional[Dict[int, int]]
    :param cntl: optional dict mapping MUMPS CNTL indices to float values
        for the LU of :math:`A_\epsilon`.
    :type cntl: Optional[Dict[int, float]]

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """
    comm = A.getComm()

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A, Aeps)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    if monitor:

        def monitor_fun(ksp, its, rnorm):
            string = f"GMRES Iteration {its:3d}, Residual Norm = {rnorm:.3e}"
            petscprint(comm, string)

        ksp.setMonitor(monitor_fun)

    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    if icntl or cntl:
        F = pc.getFactorMatrix()
        if icntl:
            for k, v in icntl.items():
                F.setMumpsIcntl(k, v)
        if cntl:
            for k, v in cntl.items():
                F.setMumpsCntl(k, v)
    pc.setUp()
    ksp.setUp()
    return ksp


def create_gmres_block_banded_solver(
    A: PETSc.Mat,
    nblocks: int,
    n_off_diags: int = 1,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
    icntl: typing.Optional[typing.Dict[int, int]] = None,
    cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Create a GMRES solver for :math:`A x = b` whose left preconditioner
    is the MUMPS LU factorization of the *block-banded* part of
    :math:`A`.  With ``n_off_diags = 1`` (the default) the preconditioner
    is the block-tridiagonal part of :math:`A`; ``n_off_diags = 0`` gives
    the block-diagonal and ``n_off_diags = 2`` the block-pentadiagonal
    part, etc.

    The band matrix
    :math:`A_b =`
    :func:`~resolvent4py.utils.matrix.extract_block_banded`
    ``(A, nblocks, n_off_diags)`` is assembled, MUMPS-factorized, and
    used via ``ksp.setOperators(A, A_b)`` so the outer Krylov iteration
    is on :math:`A` while the preconditioner is built from :math:`A_b`.
    This is useful when :math:`A` has a dominant near-diagonal block
    structure (e.g. a harmonic-balanced operator whose base flow couples
    only a few neighbouring Fourier blocks): the band captures most of
    the operator at a fraction of the factorization cost of the full
    :math:`A`.

    :param A: PETSc matrix iterated on by GMRES (the "true" operator).
        Must be block-structured with ``nblocks`` blocks per dimension.
    :type A: PETSc.Mat
    :param nblocks: number of blocks along each dimension of :math:`A`.
    :type nblocks: int
    :param n_off_diags: number of block off-diagonals to keep on each
        side of the main block-diagonal when forming the preconditioner
        (default 1, i.e. block-tridiagonal).
    :type n_off_diags: int
    :param rtol: relative tolerance for GMRES
    :type rtol: Optional[float], default :math:`10^{-10}`
    :param atol: absolute tolerance for GMRES
    :type atol: Optional[float], default :math:`10^{-10}`
    :param monitor: :code:`True` to monitor convergence and print residual
        history to terminal. :code:`False` otherwise
    :type monitor: Optional[bool], default :code:`False`
    :param icntl: optional dict mapping MUMPS ICNTL indices to integer
        values for the LU of the band matrix.  See
        :func:`.create_mumps_solver` for common knobs.
    :type icntl: Optional[Dict[int, int]]
    :param cntl: optional dict mapping MUMPS CNTL indices to float values
        for the LU of the band matrix.
    :type cntl: Optional[Dict[int, float]]

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """
    from .matrix import extract_block_banded

    # The band matrix is reference-counted by PETSc once handed to
    # ksp.setOperators inside create_gmres_epslu_solver, so it stays
    # alive with the KSP even after this local handle goes out of scope.
    Aband = extract_block_banded(A, nblocks, n_off_diags)
    return create_gmres_epslu_solver(
        A,
        Aband,
        rtol=rtol,
        atol=atol,
        monitor=monitor,
        icntl=icntl,
        cntl=cntl,
    )


def check_gmres_bjacobi_solver(A: PETSc.Mat, ksp: PETSc.KSP) -> None:
    r"""
    Check that the solver computed in :func:`.create_gmres_bjacobi_solver`
    has succeeded.

    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param ksp: PETSc KSP solver
    :type ksp: PETSc.KSP
    """
    sizes = A.getSizes()[0]
    b = generate_random_petsc_vector(sizes)
    x = b.duplicate()
    ksp.solve(b, x)
    x.destroy()
    b.destroy()
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise ValueError(
            f"GMRES solver did not converge. ConvergedReason = {reason}"
        )
