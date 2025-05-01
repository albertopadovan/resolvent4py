__all__ = [
    "create_mumps_solver",
    "check_lu_factorization",
    "create_gmres_bjacobi_solver",
    "check_gmres_bjacobi_solver",
]

import typing

from mpi4py import MPI
from petsc4py import PETSc

from .miscellaneous import petscprint
from .random import generate_random_petsc_vector


def create_mumps_solver(comm: MPI.Comm, A: PETSc.Mat) -> PETSc.KSP:
    r"""
    Compute an LU factorization of the matrix A using
    `MUMPS <https://mumps-solver.org/index.php?page=doc>`

    :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
        :code:`MPI.COMM_SELF`)
    :type comm: MPI.Comm
    :param A: PETSc matrix
    :type A: PETSc.Mat

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.setUp()
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setUp()
    return ksp


def check_lu_factorization(
    comm: MPI.Comm, A: PETSc.Mat, ksp: PETSc.KSP
) -> None:
    r"""
    Check that the LU factorization computed in :func:`.create_mumps_solver`
    has succeeded.

    :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
        :code:`MPI.COMM_SELF`)
    :type comm: MPI.Comm
    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param ksp: PETSc KSP solver
    :type ksp: PETSc.KSP
    """
    sizes = A.getSizes()[0]
    b = generate_random_petsc_vector(comm, sizes)
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
    comm: MPI.Comm,
    A: PETSc.Mat,
    nblocks: int,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
) -> PETSc.KSP:
    r"""
    Create GMRES solver with block-jacobi preconditioner.

    :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
        :code:`MPI.COMM_SELF`)
    :type comm: MPI.Comm
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


def check_gmres_bjacobi_solver(
    comm: MPI.Comm, A: PETSc.Mat, ksp: PETSc.KSP
) -> None:
    r"""
    Check that the solver computed in :func:`.create_gmres_bjacobi_solver`
    has succeeded.

    :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
        :code:`MPI.COMM_SELF`)
    :type comm: MPI.Comm
    :param A: PETSc matrix
    :type A: PETSc.Mat
    :param ksp: PETSc KSP solver
    :type ksp: PETSc.KSP
    """
    sizes = A.getSizes()[0]
    b = generate_random_petsc_vector(comm, sizes)
    x = b.duplicate()
    ksp.solve(b, x)
    x.destroy()
    b.destroy()
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise ValueError(
            f"GMRES solver did not converge. ConvergedReason = {reason}"
        )
