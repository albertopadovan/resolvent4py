__all__ = [
    "create_mumps_solver",
    "check_lu_factorization",
    "create_gmres_bjacobi_solver",
    "check_gmres_bjacobi_solver",
    "create_gmres_superoptimal_solver",
]

import typing

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
#
# Algorithm (for a quasi-block-Toeplitz harmonic-resolvent operator
# :math:`T_{k,j} = (s + i k \omega) M \delta_{k,j} - A_{k-j}`):
#
#   Setup (once):
#     - Extract Fourier blocks A_{-m}..A_m and M_0 from the harmonic-balanced
#       operands.
#     - Form DFT-domain blocks
#         G_{kk} = (s + i k \omega) M_0 - sum_l A_l e^{2 pi i l k / n}
#         R_k    = sum_j B_j e^{2 pi i j k / n},
#         B_j    = (1/n) * sum_{p : -m <= p, p+j <= m} (T T^*)_{p, p+j}
#     - Factor each R_k with MUMPS.
#
#   Apply C^{-1} to a global vector x:
#     1. Reshape x (size nN) -> SLEPc BV (N rows x n cols, col j = harmonic j).
#     2. FFT along the harmonic axis (local on each rank).
#     3. For k = 0..n-1: solve R_k y_k = v_k, then x_k = G_{kk} y_k.
#     4. Inverse FFT along the harmonic axis.
#     5. Reshape BV -> output global vector y.
# ---------------------------------------------------------------------------


class _SuperoptimalPC:
    r"""
    PETSc Python shell preconditioner implementing :math:`C^{-1}` where
    :math:`C` is the **Strang block-circulant approximation** of the
    harmonic-resolvent operator.

    For a block-Toeplitz harmonic resolvent
    :math:`T_{k,j} = (s + i k \omega) M_0 \delta_{kj} - A_{k-j}` with
    Fourier coefficients :math:`A_l` for :math:`l = -m, \ldots, m`, the
    Strang block-circulant approximation has DFT-domain eigenvalues

    .. math::

        \hat{c}_k \;=\; s \, M_0 \;-\; \sum_{l=-m}^{m}
            A_l \, \exp\!\left(-\tfrac{2\pi i\, k\, l}{n}\right),
        \qquad k = 0, 1, \ldots, n-1,

    indexed in **storage order** (matching ``numpy.fft.fft``).  The
    position-dependent diagonal shift
    :math:`(s + i k \omega) M_0` averages to :math:`s M_0` over the
    physical harmonics, which is what survives in the block-circulant
    approximation.

    Internally the :math:`\hat{c}_k` are packed into a single
    :math:`nN \times nN` block-diagonal sparse matrix ``C_global`` so
    that PETSc's PCBJACOBI(n) partitions it into ``nblocks`` sub-blocks
    on disjoint sub-communicators. A single ``ksp.solve`` therefore
    triggers ``n`` MUMPS solves concurrently, one per sub-comm of
    ``P / nblocks`` ranks.

    Apply pipeline (per outer GMRES iteration):

    1. ``x`` (stacked, size ``nN``) -> BV -> ``fft`` along the harmonic
       axis (local per-rank).
    2. BV -> ``stacked_in``.
    3. ``C_ksp.solve(stacked_in, stacked_out)`` — n parallel sub-comm
       MUMPS solves.
    4. ``stacked_out`` -> BV -> ``ifft`` -> ``y``.
    """

    def __init__(self, C_global, C_ksp, nblocks):
        self.C_global = C_global       # block_diag(c_hat_0, ..., c_hat_{n-1})
        self.C_ksp = C_ksp             # KSP wrapping C_global with PCBJACOBI(n)
        self.nblocks = nblocks
        # Working storage in the "stacked" layout (size nN).
        self._stacked_in = C_global.createVecLeft()
        self._stacked_out = C_global.createVecLeft()

    def setUp(self, pc):
        pass

    def apply(self, pc, x, y):
        # NB: do NOT reuse a single BV across calls. The reshape utility
        # internally calls setValuesCSR(..., True) which under petsc4py
        # interprets the trailing bool as ADD_VALUES, so a reused BV would
        # accumulate values from previous calls instead of being overwritten.
        # Building a fresh BV per apply is a few % overhead and correct.
        from .vector import reshape_harmonic_balanced_vector_into_bv
        from .bv import reshape_bv_into_harmonic_balanced_vector

        # ---- 1. x (stacked, size nN) -> BV -> FFT along axis=1 (local).
        bv = reshape_harmonic_balanced_vector_into_bv(x, self.nblocks)
        bv_mat = bv.getMat()
        bv_arr = bv_mat.getDenseArray()
        bv_arr[:] = np.fft.fft(bv_arr, axis=1)
        bv.restoreMat(bv_mat)
        reshape_bv_into_harmonic_balanced_vector(bv, self._stacked_in)
        bv.destroy()

        # ---- 2. Solve C_global y = stacked_in.  C_global is block-diagonal
        # with diagonal blocks c_hat_k; PCBJACOBI(n) puts each on its own
        # sub-comm, so all n MUMPS solves happen in parallel.
        self.C_ksp.solve(self._stacked_in, self._stacked_out)

        # ---- 3. stacked_out -> BV -> IFFT along axis=1 -> y.
        bv = reshape_harmonic_balanced_vector_into_bv(
            self._stacked_out, self.nblocks
        )
        bv_mat = bv.getMat()
        bv_arr = bv_mat.getDenseArray()
        bv_arr[:] = np.fft.ifft(bv_arr, axis=1)
        bv.restoreMat(bv_mat)
        reshape_bv_into_harmonic_balanced_vector(bv, y)
        bv.destroy()

    def destroy(self, pc=None):
        self.C_ksp.destroy()
        self.C_global.destroy()
        self._stacked_in.destroy()
        self._stacked_out.destroy()


def _block_diag_from_per_block_mats(
    per_block_mats: typing.List[PETSc.Mat],
    nblocks: int,
    N: int,
    comm: PETSc.Comm,
) -> PETSc.Mat:
    r"""
    Assemble an :math:`nN \times nN` block-diagonal sparse AIJ matrix
    whose :math:`(k, k)` block is ``per_block_mats[k]`` (an
    :math:`N \times N` parallel AIJ on the same comm).
    """
    from .matrix import convert_coo_to_csr
    from .comms import compute_local_size

    nN = nblocks * N
    Nl = compute_local_size(nN)
    sizes = ((Nl, nN), (Nl, nN))

    rows_chunks: typing.List[np.ndarray] = []
    cols_chunks: typing.List[np.ndarray] = []
    vals_chunks: typing.List[np.ndarray] = []
    for k, mat in enumerate(per_block_mats):
        r0, r1 = mat.getOwnershipRange()
        offset = k * N
        for i in range(r0, r1):
            cols, vals = mat.getRow(i)
            n_e = len(cols)
            if n_e == 0:
                continue
            rows_chunks.append(
                np.full(n_e, offset + i, dtype=PETSc.IntType)
            )
            cols_chunks.append(
                np.asarray(cols, dtype=PETSc.IntType) + offset
            )
            vals_chunks.append(np.asarray(vals, dtype=PETSc.ScalarType))

    if rows_chunks:
        rows_arr = np.concatenate(rows_chunks)
        cols_arr = np.concatenate(cols_chunks)
        vals_arr = np.concatenate(vals_chunks)
    else:
        rows_arr = np.empty(0, dtype=PETSc.IntType)
        cols_arr = np.empty(0, dtype=PETSc.IntType)
        vals_arr = np.empty(0, dtype=PETSc.ScalarType)

    rows_ptr, cols_csr, vals_csr = convert_coo_to_csr(
        [rows_arr, cols_arr, vals_arr], sizes
    )
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols_csr))
    M.setValuesCSR(rows_ptr, cols_csr, vals_csr, True)
    M.assemble()
    return M


def _make_block_diag_bjacobi_solver(
    R_global: PETSc.Mat,
    nblocks: int,
    icntl: typing.Optional[typing.Dict[int, int]],
    cntl: typing.Optional[typing.Dict[int, float]],
) -> PETSc.KSP:
    r"""
    Build a ``preonly`` KSP whose PC is ``bjacobi`` with ``nblocks``
    sub-blocks, each handled by ``preonly + LU + MUMPS``. Because
    ``R_global`` is block-diagonal, this is an exact direct solve;
    PCBJACOBI's per-sub-block sub-communicator parallelises the n
    sub-solves.
    """
    comm = R_global.getComm()

    # Use a unique options prefix so this KSP doesn't collide with other
    # KSPs the caller may have created from PETSc.Options. Must start with
    # a letter — PETSc rejects option names beginning with '_'.
    prefix = "r4pysuperoptrk_"
    opts = PETSc.Options()
    opts[prefix + "pc_bjacobi_blocks"] = nblocks
    opts[prefix + "sub_ksp_type"] = "preonly"
    opts[prefix + "sub_pc_type"] = "lu"
    opts[prefix + "sub_pc_factor_mat_solver_type"] = "mumps"
    if icntl:
        for k, v in icntl.items():
            opts[prefix + f"sub_pc_factor_mat_mumps_icntl_{k}"] = v
    if cntl:
        for k, v in cntl.items():
            opts[prefix + f"sub_pc_factor_mat_mumps_cntl_{k}"] = v

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(R_global)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("bjacobi")
    pc.setFromOptions()
    pc.setUp()
    pc.setUpOnBlocks()  # forces MUMPS factor on each sub-block now
    ksp.setUp()
    return ksp


def _setup_superoptimal_pc(
    Tinv: PETSc.Mat,
    A: PETSc.Mat,
    M: PETSc.Mat,
    nblocks: int,
    omega: float,
    s: complex,
    icntl: typing.Optional[typing.Dict[int, int]] = None,
    cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> "_SuperoptimalPC":
    r"""
    One-time setup for the Strang block-circulant preconditioner.

    Builds the DFT-domain eigenvalues of :math:`C` in storage order
    :math:`k = 0, \ldots, n-1`:

    .. math::

        \hat{c}_k = s\, M_0 \;-\; \sum_{l=-m}^{m} A_l \,
            \exp\!\left(-\tfrac{2\pi i\, k\, l}{n}\right).

    These ``n`` matrices are then packed into a single
    :math:`nN \times nN` block-diagonal sparse matrix
    ``C_global = block_diag(c_hat_0, ..., c_hat_{n-1})`` on the parent
    communicator, and a KSP wrapping ``C_global`` with
    PCBJACOBI(nblocks) is configured so each MUMPS factor / solve runs
    on a sub-comm of ``P / nblocks`` ranks.

    .. note::

        ``Tinv`` and ``omega`` are accepted for API compatibility but
        are not used in the Strang construction (the position-dependent
        :math:`i k \omega M_0` shift averages out across harmonics).
    """
    from .matrix import extract_matrix_block

    _ = Tinv  # only needed by potential future "superoptimal" extension
    _ = omega  # ditto: averages to zero over harmonics

    comm = A.getComm()
    nN = A.getSizes()[0][-1]
    N = nN // nblocks
    m = (nblocks - 1) // 2

    # ---- Fourier coefficient blocks A_{-m},...,A_m from block-Toeplitz A.
    # Block (k,j) of A holds A_{k-j}; the middle block-column (col index m)
    # contains [A_{-m},...,A_0,...,A_m] reading down the block-rows.
    Ak_list = [
        extract_matrix_block(A, nblocks, rowblock=i, colblock=m)
        for i in range(nblocks)
    ]

    # ---- Mass block M_0 (middle diagonal block of M).
    M0 = extract_matrix_block(M, nblocks, rowblock=m, colblock=m)

    # ---- DFT-domain blocks in storage order:
    #   c_hat[k] = s * M_0  -  sum_{l=-m..m} A_l * exp(-2 pi i k l / n)
    # Indexed by k = 0..n-1, matching numpy.fft.fft output order.
    c_hat_list: typing.List[PETSc.Mat] = []
    for k in range(nblocks):
        ck = M0.copy()
        ck.scale(s)
        for l in range(-m, m + 1):
            ck.axpy(
                -np.exp(-2j * np.pi * k * l / nblocks),
                Ak_list[l + m],
            )
        c_hat_list.append(ck)

    M0.destroy()
    for Ak in Ak_list:
        Ak.destroy()

    # ---- Pack into one block-diagonal nN x nN AIJ on the parent comm.
    C_global = _block_diag_from_per_block_mats(c_hat_list, nblocks, N, comm)
    for ck in c_hat_list:
        ck.destroy()

    # ---- KSP for C_global with PCBJACOBI(n): n parallel sub-comm solves.
    C_ksp = _make_block_diag_bjacobi_solver(C_global, nblocks, icntl, cntl)

    return _SuperoptimalPC(C_global, C_ksp, nblocks)


def create_gmres_superoptimal_solver(
    Tinv: PETSc.Mat,
    A: PETSc.Mat,
    M: PETSc.Mat,
    nblocks: int,
    omega: float,
    s: complex = 0.0,
    rtol: typing.Optional[float] = 1e-10,
    atol: typing.Optional[float] = 1e-10,
    monitor: typing.Optional[bool] = False,
    icntl: typing.Optional[typing.Dict[int, int]] = None,
    cntl: typing.Optional[typing.Dict[int, float]] = None,
) -> PETSc.KSP:
    r"""
    Create a GMRES solver preconditioned by the **Strang block-circulant
    approximation** :math:`C^{-1}` of the quasi-block-Toeplitz
    harmonic-resolvent operator
    :math:`T_{k,j} = (s + i k \omega) M \delta_{k,j} - A_{k-j}`.

    The DFT-domain eigenvalues of :math:`C` (in storage order matching
    ``numpy.fft.fft``) are

    .. math::

        \hat{c}_k \;=\; s\, M_0 \;-\; \sum_{l=-m}^{m} A_l \,
            \exp\!\left(-\tfrac{2\pi i\, k\, l}{n}\right),
        \qquad k = 0, \ldots, n-1.

    Apply pipeline per outer GMRES iteration: ``fft`` along the
    harmonic axis, one parallel block-Jacobi solve against
    ``block_diag(c_hat_0, ..., c_hat_{n-1})``, ``ifft``.

    :param Tinv: assembled :math:`nN \times nN` operator to invert
        (e.g. ``s*M - T``). Used as the GMRES outer operator.
    :type Tinv: PETSc.Mat
    :param A: assembled :math:`nN \times nN` block-Toeplitz Jacobian with
        block :math:`(k,j) = A_{k-j}` (the input to
        :func:`assemble_harmonic_resolvent_generator`).
    :type A: PETSc.Mat
    :param M: assembled :math:`nN \times nN` block-diagonal mass matrix
        (only the middle diagonal block :math:`M_0` is used).
    :type M: PETSc.Mat
    :param nblocks: number of harmonic blocks :math:`= 2 m + 1`.
    :type nblocks: int
    :param omega: fundamental frequency. Accepted for API compatibility
        but does not appear in the Strang construction (the
        :math:`i k \omega M_0` shift averages to zero over the
        physical harmonics).
    :type omega: float
    :param s: Laplace-domain shift.
    :type s: complex, default ``0.0``
    :param rtol: GMRES relative tolerance.
    :type rtol: Optional[float], default :math:`10^{-10}`
    :param atol: GMRES absolute tolerance.
    :type atol: Optional[float], default :math:`10^{-10}`
    :param monitor: print residual history if ``True``.
    :type monitor: Optional[bool], default ``False``
    :param icntl: MUMPS ICNTL settings forwarded to each :math:`\hat{c}_k` factor.
    :type icntl: Optional[Dict[int, int]]
    :param cntl: MUMPS CNTL settings forwarded to each :math:`\hat{c}_k` factor.
    :type cntl: Optional[Dict[int, float]]

    :return: configured PETSc KSP whose :code:`solve` applies
        GMRES(C^{-1} T_inv).
    :rtype: PETSc.KSP

    .. note::

        Setup cost is ``nblocks`` MUMPS factorizations of size
        :math:`N \times N`, each on a sub-comm of ``P / nblocks``
        ranks (via PCBJACOBI), plus the cost of extracting the Fourier
        blocks :math:`A_l` and assembling
        ``block_diag(c_hat_0, ..., c_hat_{n-1})``. Much cheaper than
        the previous "superoptimal" version, which required the full
        :math:`T T^*` product.
    """
    comm = Tinv.getComm()

    pc_ctx = _setup_superoptimal_pc(
        Tinv, A, M, nblocks, omega, s, icntl=icntl, cntl=cntl
    )

    monitor_fun = None
    if monitor:
        def monitor_fun(ksp_obj, its, rnorm):
            petscprint(
                comm,
                f"GMRES (superopt) Iteration {its:3d}, "
                f"Residual Norm = {rnorm:.3e}",
            )

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(Tinv)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol)
    if monitor_fun is not None:
        ksp.setMonitor(monitor_fun)
    pc = ksp.getPC()
    pc.setType("python")
    pc.setPythonContext(pc_ctx)
    pc.setUp()
    ksp.setUp()

    return ksp
