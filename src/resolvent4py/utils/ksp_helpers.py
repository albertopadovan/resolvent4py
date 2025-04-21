from .. import PETSc
from .. import MPI
from .random_helpers import generate_random_petsc_vector


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
