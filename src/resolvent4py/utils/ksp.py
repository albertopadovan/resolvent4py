from __future__ import annotations

__all__ = [
    "create_mumps_solver",
    "check_lu_factorization",
    "create_gmres_bjacobi_solver",
    "check_gmres_bjacobi_solver",
    "create_gmres_superoptimal_solver",
]

import typing
from typing import TYPE_CHECKING

import numpy as np
from petsc4py import PETSc

if TYPE_CHECKING:
    from ..linear_operators import LinearOperator

from .matrix import create_AIJ_identity, extract_matrix_block
from .miscellaneous import petscprint
from .random import generate_random_petsc_vector


def create_mumps_solver(A: PETSc.Mat) -> PETSc.KSP:
    r"""
    Compute an LU factorization of the matrix A using MUMPS.

    :param A: PETSc matrix
    :type A: PETSc.Mat

    :return ksp: PETSc KSP solver
    :rtype ksp: PETSc.KSP
    """
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    # pc.setReusePreconditioner(True)
    # Mat = pc.getFactorMatrix()
    # Mat.setMumpsIcntl(7, 5)
    # Mat.setMumpsIcntl(28, 1)
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


# ---------------------------------------------------------------------------
# Superoptimal block-circulant preconditioned GMRES
# ---------------------------------------------------------------------------



class _SuperoptimalPC:
    r"""
    PETSc Python shell preconditioner implementing :math:`C^{-1}` where
    :math:`C` is the superoptimal block-circulant approximation.

    Application (Algorithm 1):

    1. Block-FFT the input vector along the harmonic index.
    2. For each harmonic :math:`k = 0, \ldots, n-1`:
       (a) Solve :math:`R_k y_k = \hat{v}_k` (pre-factored).
       (b) Compute :math:`\hat{x}_k = G_{kk} y_k` (sparse mat-vec).
    3. Block-IFFT the result.
    """

    def __init__(self, Gkk_list, Rk_ksp_list, Rk_list, nblocks, N,
                 Ak_list, M0):
        self.Gkk = Gkk_list
        self.Rk_ksp = Rk_ksp_list
        self.Rk = Rk_list
        self.nblocks = nblocks
        self.N = N
        self.Ak_list = Ak_list
        self.M0 = M0

    def setUp(self, pc):
        pass

    def apply(self, pc, x, y):
        nblocks, N = self.nblocks, self.N

        # Gather full vector to numpy, reshape to (nblocks, N)
        x_arr = x.getArray(readonly=True).copy().reshape(nblocks, N)

        # Step 1: block-FFT along harmonic index (axis 0)
        v_hat = np.fft.fft(x_arr, axis=0)

        # Step 2: per-harmonic R_k solve + G_kk mat-vec (PETSc)
        x_hat = np.empty_like(v_hat)
        comm = self.Gkk[0].getComm()
        for k in range(nblocks):
            # Create PETSc vectors for the k-th block
            vk = PETSc.Vec().createWithArray(
                v_hat[k].copy(), comm=comm
            )
            yk = vk.duplicate()
            xk = vk.duplicate()

            # (a) Solve R_k y_k = v_k
            self.Rk_ksp[k].solve(vk, yk)
            # (b) x_k = G_kk y_k
            self.Gkk[k].mult(yk, xk)

            x_hat[k] = xk.getArray().copy()
            vk.destroy()
            yk.destroy()
            xk.destroy()

        # Step 3: block-IFFT
        y.getArray()[:] = np.fft.ifft(x_hat, axis=0).reshape(-1)

    def destroy(self):
        for mat in self.Gkk:
            mat.destroy()
        for ksp in self.Rk_ksp:
            ksp.destroy()
        for mat in self.Rk:
            mat.destroy()
        for mat in self.Ak_list:
            mat.destroy()
        self.M0.destroy()



def create_gmres_superoptimal_solver(
    T: LinearOperator,
    m: int,
    N: int,
    omega: float,
    s: complex = 0.0,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
) -> PETSc.KSP:
    r"""
    Create a GMRES solver preconditioned by the superoptimal block-circulant
    approximation :math:`C^{-1}` of the quasi-block-Toeplitz harmonic-balance
    matrix :math:`T`.

    The Fourier coefficient blocks :math:`A_j` are extracted directly from
    :math:`T` (which has block structure
    :math:`T_{k,j} = (s + ik\omega) M \delta_{k,j} - A_{k-j}`).
    The mdle block-column of :math:`T` gives
    :math:`[\ldots, -A_{-1}, sI - A_0, -A_{1}, \ldots]^T`.

    The preconditioner :math:`C^{-1}` is applied via Algorithm 1:
    block-FFT, n independent N x N solves + sparse mat-vecs, block-IFFT.

    .. note::

        Currently assumes :math:`M = I` (identity mass matrix).
        Sequential (single-MPI-rank) only.

    :param T: linear operator wrapping the nN x nN PETSc sparse matrix
    :type T: LinearOperator
    :param m: number of baseflow Fourier harmonics
    :type m: int
    :param N: block size (spatial DOFs)
    :type N: int
    :param omega: fundamental frequency
    :type omega: float
    :param s: Laplace-domain shift
    :type s: complex, default 0.0
    :param rtol: relative tolerance for GMRES
    :type rtol: Optional[float], default :math:`10^{-10}`
    :param atol: absolute tolerance for GMRES
    :type atol: Optional[float], default :math:`10^{-10}`
    :param monitor: print residual history if True
    :type monitor: Optional[bool], default False

    :return: comigured PETSc KSP solver
    :rtype: PETSc.KSP
    """
    comm = T.get_comm()

    pc_ctx = _setup_superoptimal_pc(T, m, N, omega, s)

    monitor_fun = None
    if monitor:
        def monitor_fun(ksp_obj, its, rnorm):
            petscprint(
                comm,
                f"GMRES (superoptimal) Iteration {its:3d}, "
                f"Residual Norm = {rnorm:.3e}",
            )

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(T.A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setMonitor(monitor_fun)
    pc = ksp.getPC()
    pc.setType("python")
    pc.setPythonContext(pc_ctx)
    pc.setUp()
    ksp.setUp()

    return ksp
