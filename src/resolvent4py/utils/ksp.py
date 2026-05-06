__all__ = [
    "create_mumps_solver",
    "check_lu_factorization",
    "create_gmres_bjacobi_solver",
    "check_gmres_bjacobi_solver",
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
    ksp = PETSc.KSP().create(comm=A.getComm())
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

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """

    comm = A.getComm()
    nprocs = comm.getSize()
    if nprocs % nblocks != 0 and comm.getRank() == 0:
        print("\n")
        warnings.warn(
            f"create_gmres_bjacobi_solver: nblocks={nblocks} is not a "
            f"multiple of comm.size={nprocs}.  PETSc's block-Jacobi "
            f"requires each MPI rank to own a whole number of blocks; "
            f"otherwise some bjacobi block straddles a rank boundary "
            f"and the preconditioner is no longer block-exact (so a "
            f"matrix that is genuinely block diagonal will not converge "
            f"in one iteration).  Use ``nblocks = k * comm.size`` for "
            f"some integer ``k >= 1``.",
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

    return ksp


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
