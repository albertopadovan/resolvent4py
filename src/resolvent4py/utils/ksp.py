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

from .matrix import convert_coo_to_csr_v2 as convert_coo_to_csr
from .matrix import create_AIJ_identity
from .miscellaneous import petscprint
from .random import generate_random_petsc_vector
from .comms import scatter_array_from_root_to_all, compute_local_size


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

def _extract_block(Mat: PETSc.Mat, nblocks: int, rowblock: int,
                   colblock: int) -> PETSc.Mat:
    r"""
    Extract a single :math:`N \times N` block from a block-structured
    :math:`nN \times nN` PETSc matrix, where the block at position
    ``(rowblock, colblock)`` occupies rows
    ``rowblock*N .. (rowblock+1)*N - 1`` and columns
    ``colblock*N .. (colblock+1)*N - 1``.

    The extraction is done via two selector multiplications:

    .. math::

        \text{block} = \hat{I}_r^T \; \text{Mat} \; \hat{I}_c

    where :math:`\hat{I}_r` and :math:`\hat{I}_c` are :math:`nN \times N`
    matrices with :math:`I_N` at the appropriate block-row.

    :param Mat: assembled :math:`nN \times nN` PETSc sparse matrix
    :type Mat: PETSc.Mat
    :param nblocks: number of blocks along each dimension
    :type nblocks: int
    :param rowblock: 0-based row-block index
    :type rowblock: int
    :param colblock: 0-based column-block index
    :type colblock: int

    :return: the extracted :math:`N \times N` PETSc sparse matrix
    :rtype: PETSc.Mat
    """
    comm = Mat.getComm()
    size = Mat.getSizes()[0]
    N = size[-1] // nblocks

    # Build selector for the column block: Ic is nN x N with I_N at colblock
    rows_coo, cols_coo, data_coo = None, None, None
    if comm.getRank() == 0:
        data_coo = np.ones(N, dtype=PETSc.ScalarType)
        cols_coo = np.arange(N, dtype=PETSc.IntType)
        rows_coo = cols_coo + colblock * N

    mat_sizes_sel = (size, (compute_local_size(N), N))
    Ic = _assemble_matrix(comm, [rows_coo, cols_coo, data_coo], mat_sizes_sel)

    # Mat @ Ic gives nN x N (the colblock-th block-column)
    MatIc = Mat.matMult(Ic)
    Ic.destroy()

    # Build selector for the row block: Ir is nN x N with I_N at rowblock
    rows_coo, cols_coo, data_coo = None, None, None
    if comm.getRank() == 0:
        data_coo = np.ones(N, dtype=PETSc.ScalarType)
        cols_coo = np.arange(N, dtype=PETSc.IntType)
        rows_coo = cols_coo + rowblock * N

    Ir = _assemble_matrix(comm, [rows_coo, cols_coo, data_coo], mat_sizes_sel)

    # Ir^H @ MatIc = (N x nN) @ (nN x N) = N x N block
    Ir.hermitianTranspose()
    block = Ir.matMult(MatIc)
    Ir.destroy()
    MatIc.destroy()

    return block


def _assemble_matrix(comm, coo_arrays, mat_sizes):
    r"""
    Assemble a PETSc AIJ sparse matrix from COO arrays via CSR conversion.

    :param comm: MPI communicator
    :type comm: PETSc.Comm
    :param coo_arrays: ``[rows, cols, vals]`` in COO format (only populated
        on rank 0; ``None`` on other ranks)
    :type coo_arrays: list
    :param mat_sizes: PETSc size spec
        ``((local_rows, global_rows), (local_cols, global_cols))``
    :type mat_sizes: tuple

    :return: assembled PETSc sparse matrix
    :rtype: PETSc.Mat
    """
    rows_coo, cols_coo, data_coo = coo_arrays
    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    data = scatter_array_from_root_to_all(data_coo)
    rows_ptr, cols, vals = convert_coo_to_csr([rows, cols, data], mat_sizes)

    M = PETSc.Mat().createAIJ(mat_sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble()

    return M


def _extract_toeplitz_blocks(Ahat: PETSc.Mat, nblocks: int):
    r"""
    Extract the Fourier coefficient blocks :math:`A_{-m}, \ldots, A_{m}`
    from the harmonic-balanced operator :math:`\hat{A}` defined as
    :math:`\hat{A}_{k,j} = A_{k-j}`.

    The mdle block-column (block-column index :math:`m`) is extracted
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
        _extract_block(Ahat, nblocks, rowblock=i, colblock=(nblocks - 1) // 2)
        for i in range (nblocks)
    ]



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


def _setup_superoptimal_pc(
    T: LinearOperator,
    A: PETSc.Mat,
    M: PETSc.Mat,
    omega: float,
    s: complex,
) -> _SuperoptimalPC:
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
    
    :param T: linear operator wrapping the :math:`nN \times nN` PETSc
        sparse matrix (the full harmonic-balance system)
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

    :return: configured :class:`_SuperoptimalPC` context
    """
    nblocks = T.get_nblocks()
    nN = T.get_dimensions()[0][-1]
    N = nN // nblocks

    # ------------------------------------------------------------------
    # Extract A_j and M_0 blocks from the block-Toeplitz matrices.
    # _extract_toeplitz_blocks returns nblocks PETSc.Mat objects
    # ordered by block-row index (0 .. nblocks-1).
    # ------------------------------------------------------------------
    comm = T.get_comm()
    Ak_list = _extract_toeplitz_blocks(A, nblocks)
    Mlst_petsc = _extract_toeplitz_blocks(M, nblocks)
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
        Gkk.scale(s + 1j * k * omega)
        for l in range(-m, m + 1):
            Gkk.axpy(-np.exp(2j * np.pi * l * k / nblocks), Ak_list[l + m])
        Gkk_list.append(Gkk.copy())
        Gkk.destroy()

    # ------------------------------------------------------------------
    # Assemble R_{k} as described in the docstrings. To do that, we 
    # first need to assemble B_{k}
    # ------------------------------------------------------------------
    TH = T.A.copy()
    TH.hermitianTranspose()
    TTH = T.A.matMult(TH)
    TH.destroy()
    Bk_list = []
    for k in range (-2 * m, 2 * m + 1):
        Bk_initialized = False
        for p in range (-m, m + 1):
            if -m <= p + k <= m:
                Mpk = _extract_block(TTH, nblocks, rowblock=p + m, colblock=p + k + m)
                if not Bk_initialized:
                    Bk = Mpk.copy()
                    Bk_initialized = True
                else:
                    Bk.axpy(1.0, Mpk)
                Mpk.destroy()
        if not Bk_initialized:
            Bk = _extract_block(TTH, nblocks, rowblock=0, colblock=0)
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
    
    # ------------------------------------------------------------------
    # Factor R_k via PETSc KSP (MUMPS LU).
    # Each entry is a PETSc.KSP ready to solve R_k x = b.
    # ------------------------------------------------------------------
    Rk_ksp_list = []
    for k in range(nblocks):
        ksp_rk = create_mumps_solver(Rk_list[k])
        Rk_ksp_list.append(ksp_rk)

    return _SuperoptimalPC(Gkk_list, Rk_ksp_list, Rk_list, nblocks, N,
                           Ak_list, M0)


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
