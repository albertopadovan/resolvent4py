import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.utils.comms import scatter_array_from_root_to_all, compute_local_size
from resolvent4py.utils.matrix import convert_coo_to_csr_v2, create_AIJ_identity
from resolvent4py.linear_operators.superoptimal_block_circulant import (
    SuperoptimalBlockCirculantLinearOperator,
)
from .. import pytest_utils


def _numpy_to_petsc(comm, A_np):
    """Convert a dense numpy matrix (known on all ranks) to a distributed
    PETSc AIJ matrix."""
    N = A_np.shape[0]
    rows_coo, cols_coo, vals_coo = None, None, None
    if comm.getRank() == 0:
        r, c = np.nonzero(A_np)
        rows_coo = np.asarray(r, dtype=PETSc.IntType)
        cols_coo = np.asarray(c, dtype=PETSc.IntType)
        vals_coo = np.asarray(A_np[r, c], dtype=PETSc.ScalarType)

    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    vals = scatter_array_from_root_to_all(vals_coo)

    Nl = compute_local_size(N)
    sizes = ((Nl, N), (Nl, N))
    rows_ptr, cols_csr, vals_csr = convert_coo_to_csr_v2(
        [rows, cols, vals], sizes
    )
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols_csr))
    M.setValuesCSR(rows_ptr, cols_csr, vals_csr, True)
    M.assemble()
    return M


def _build_block_circulant_np(n, nblocks, rng):
    r"""Build a block-circulant nN x nN numpy matrix.

    Blocks A_j (j = 0, ..., nblocks-1) are random, and
    T_{k,j} = A_{(k-j) mod nblocks}.
    """
    nN = n * nblocks
    Aj_blocks = [
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        for _ in range(nblocks)
    ]
    T_np = np.zeros((nN, nN), dtype=np.complex128)
    for k in range(nblocks):
        for j in range(nblocks):
            diff = (k - j) % nblocks
            T_np[k * n : (k + 1) * n, j * n : (j + 1) * n] = Aj_blocks[diff]
    return T_np


def test_apply_matches_numpy_reference(comm):
    r"""Test that SBCLO.apply(x) matches a pure-numpy implementation of
    C x = F^{-1} diag(Lambda_k) F x, using the G_kk and R_k computed
    by setup()."""
    import scipy as sp

    n = 4
    nblocks = 5
    m = (nblocks - 1) // 2
    nN = n * nblocks

    T_np = comm.tompi4py().bcast(
        _build_block_circulant_np(n, nblocks, np.random.default_rng(42))
        if comm.getRank() == 0
        else None,
        root=0,
    )

    T_petsc = _numpy_to_petsc(comm, T_np)
    ksp = res4py.create_mumps_solver(T_petsc)
    T_linop = res4py.linear_operators.MatrixLinearOperator(
        T_petsc, ksp, nblocks
    )
    A_petsc = T_petsc.copy()
    A_petsc.scale(-1.0)
    Nl = compute_local_size(nN)
    M_petsc = PETSc.Mat().createAIJ(
        ((Nl, nN), (Nl, nN)), comm=comm
    )
    M_petsc.setUp()
    M_petsc.assemble()

    sbclo = SuperoptimalBlockCirculantLinearOperator(
        T_linop, A_petsc, M_petsc, omega=0.0, s=0.0
    )

    # Extract G_kk and R_k as numpy arrays
    Gkk_np_list = []
    Rk_np_list = []
    for k in range(nblocks):
        Gk = sbclo.Gkk_list[k].copy()
        Gk.convert(PETSc.Mat.Type.DENSE)
        Gs = res4py.distributed_to_sequential_matrix(Gk)
        Gkk_np_list.append(Gs.getDenseArray().copy())
        Gs.destroy()
        Gk.destroy()

        Rk = sbclo.Rk_list[k].copy()
        Rk.convert(PETSc.Mat.Type.DENSE)
        Rs = res4py.distributed_to_sequential_matrix(Rk)
        Rk_np_list.append(Rs.getDenseArray().copy())
        Rs.destroy()
        Rk.destroy()

    # Pure numpy C*x:
    # 1. Reshape x -> n x nblocks BV
    # 2. IFFT (bwd) along columns (with ifftshift/fftshift)
    # 3. Per-block: Lambda_k = R_k @ inv(G_kk), apply Lambda_k
    # 4. FFT (fwd) along columns
    # 5. Reshape back to vector
    x_vec, x_np = pytest_utils.generate_random_vector(comm, nN)

    x_bv = x_np.reshape(nblocks, n).T  # n x nblocks
    x_hat = sp.fft.fftshift(
        sp.fft.ifft(sp.fft.ifftshift(x_bv, axes=1), axis=1, norm="ortho"),
        axes=1,
    )
    y_hat = np.zeros_like(x_hat)
    for k in range(nblocks):
        w = np.linalg.solve(Gkk_np_list[k], x_hat[:, k])
        y_hat[:, k] = Rk_np_list[k] @ w
    y_bv = sp.fft.fftshift(
        sp.fft.fft(sp.fft.ifftshift(y_hat, axes=1), axis=1, norm="ortho"),
        axes=1,
    )
    y_expected = y_bv.T.reshape(-1)

    # PETSc apply
    y_C = sbclo.apply(x_vec)
    y_C_seq = res4py.distributed_to_sequential_vector(y_C)
    y_C_np = y_C_seq.getArray().copy()
    y_C_seq.destroy()

    error = np.linalg.norm(y_C_np - y_expected) / np.linalg.norm(y_expected)

    x_vec.destroy()
    y_C.destroy()
    ksp.destroy()
    T_petsc.destroy()
    A_petsc.destroy()
    M_petsc.destroy()

    assert error < 1e-10, f"apply vs numpy reference error: {error:.2e}"


def test_Gkk_matches_fft_block_diagonalization(comm):
    r"""Verify that G_{kk} from setup() matches the diagonal blocks of
    F T F^H, where F is the block-FFT."""
    import scipy as sp

    n = 4
    nblocks = 5
    m = (nblocks - 1) // 2
    nN = n * nblocks

    T_np = comm.tompi4py().bcast(
        _build_block_circulant_np(n, nblocks, np.random.default_rng(42))
        if comm.getRank() == 0
        else None,
        root=0,
    )

    T_petsc = _numpy_to_petsc(comm, T_np)
    ksp = res4py.create_mumps_solver(T_petsc)
    T_linop = res4py.linear_operators.MatrixLinearOperator(
        T_petsc, ksp, nblocks
    )
    A_petsc = T_petsc.copy()
    A_petsc.scale(-1.0)
    Nl = compute_local_size(nN)
    M_petsc = PETSc.Mat().createAIJ(
        ((Nl, nN), (Nl, nN)), comm=comm
    )
    M_petsc.setUp()
    M_petsc.assemble()

    sbclo = SuperoptimalBlockCirculantLinearOperator(
        T_linop, A_petsc, M_petsc, omega=0.0, s=0.0
    )

    # Build F T F^H in numpy using _fft convention
    # F is defined by applying _fft("bwd") to each basis vector
    FTFh = np.zeros((nN, nN), dtype=np.complex128)
    for j in range(nN):
        e = np.zeros((n, nblocks), dtype=np.complex128)
        e[j % n, j // n] = 1.0
        e_hat = sp.fft.fftshift(
            sp.fft.ifft(
                sp.fft.ifftshift(e, axes=1), axis=1, norm="ortho"
            ),
            axes=1,
        )
        # F^* e_j (bwd = ifft)
        col = e_hat.T.reshape(-1)
        # T @ F^* e_j
        T_col = T_np @ col
        # F @ T @ F^* e_j  (fwd = fft)
        T_col_bv = T_col.reshape(nblocks, n).T
        result = sp.fft.fftshift(
            sp.fft.fft(
                sp.fft.ifftshift(T_col_bv, axes=1), axis=1, norm="ortho"
            ),
            axes=1,
        )
        FTFh[:, j] = result.T.reshape(-1)

    # Extract diagonal blocks of F T F^H
    diag_blocks_fft = []
    for k in range(nblocks):
        diag_blocks_fft.append(
            FTFh[k * n : (k + 1) * n, k * n : (k + 1) * n]
        )

    # Compare with G_kk from setup
    # G_kk are stored as PETSc matrices in sbclo.Gkk_list[0..nblocks-1]
    # For the block-circulant case with s=omega=0, M=0:
    #   G_kk = -sum_l Ak_list[l+m] * exp(-2pi i l k / n)
    # which should equal the diagonal block of F T F^H
    # (since T = -A, so G_kk = sum_l Ak_list[l+m] * exp(...) = diag block)
    max_error = 0.0
    for k in range(nblocks):
        Gkk_petsc = sbclo.Gkk_list[k]
        Gkk_dense = Gkk_petsc.copy()
        Gkk_dense.convert(PETSc.Mat.Type.DENSE)
        Gkk_seq = res4py.distributed_to_sequential_matrix(Gkk_dense)
        Gkk_np = Gkk_seq.getDenseArray().copy()
        Gkk_seq.destroy()
        Gkk_dense.destroy()

        err = np.linalg.norm(Gkk_np - diag_blocks_fft[k]) / np.linalg.norm(
            diag_blocks_fft[k]
        )
        max_error = max(max_error, err)

    ksp.destroy()
    T_petsc.destroy()
    A_petsc.destroy()
    M_petsc.destroy()

    assert max_error < 1e-10, (
        f"G_kk vs FFT diag blocks max relative error: {max_error:.2e}"
    )


def test_Rk_matches_definition(comm):
    r"""Verify that R_k from setup() matches R_k = Σ_j G_{k,j} G_{k,j}^*
    computed directly from the full G = F T F^*.

    Uses a block-Toeplitz (non-circulant) T so that the diagonal averaging
    in setup() is exact (no wrapping truncation)."""
    import scipy as sp

    n = 4
    nf = 2
    nblocks = 2 * nf + 3  # nblocks > 2*nf+1 so there are zero blocks
    m = (nblocks - 1) // 2
    nN = n * nblocks

    # Build block-Toeplitz T (non-circulant): T_{k,j} = -A_{k-j} for |k-j|<=nf
    rng = np.random.default_rng(42)
    Aj = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
          for _ in range(2 * nf + 1)]
    T_np = np.zeros((nN, nN), dtype=np.complex128)
    for k in range(nblocks):
        for j in range(nblocks):
            diff = k - j
            if abs(diff) <= nf:
                T_np[k * n : (k + 1) * n, j * n : (j + 1) * n] = -Aj[diff + nf]
    T_np = comm.tompi4py().bcast(T_np, root=0)

    T_petsc = _numpy_to_petsc(comm, T_np)
    ksp = res4py.create_mumps_solver(T_petsc)
    T_linop = res4py.linear_operators.MatrixLinearOperator(
        T_petsc, ksp, nblocks
    )
    A_petsc = T_petsc.copy()
    A_petsc.scale(-1.0)
    Nl = compute_local_size(nN)
    M_petsc = PETSc.Mat().createAIJ(
        ((Nl, nN), (Nl, nN)), comm=comm
    )
    M_petsc.setUp()
    M_petsc.assemble()

    sbclo = SuperoptimalBlockCirculantLinearOperator(
        T_linop, A_petsc, M_petsc, omega=0.0, s=0.0
    )

    # Build full G = F^* T F using our _fft convention
    # F^* = ifft (bwd), F = fft (fwd)
    G_full = np.zeros((nN, nN), dtype=np.complex128)
    for j in range(nN):
        e = np.zeros((n, nblocks), dtype=np.complex128)
        e[j % n, j // n] = 1.0
        # F^* e_j (bwd = ifft)
        e_hat = sp.fft.fftshift(
            sp.fft.ifft(
                sp.fft.ifftshift(e, axes=1), axis=1, norm="ortho"
            ),
            axes=1,
        )
        col = e_hat.T.reshape(-1)
        # T (F^* e_j)
        T_col = T_np @ col
        # F (T F^* e_j)  (fwd = fft), giving column j of G = F T F^*
        T_col_bv = T_col.reshape(nblocks, n).T
        result = sp.fft.fftshift(
            sp.fft.fft(
                sp.fft.ifftshift(T_col_bv, axes=1), axis=1, norm="ortho"
            ),
            axes=1,
        )
        G_full[:, j] = result.T.reshape(-1)

    # Compute R_k = Σ_j G_{k,j} G_{k,j}^* from full G
    Rk_from_G = []
    for k in range(nblocks):
        Rk = np.zeros((n, n), dtype=np.complex128)
        for j in range(nblocks):
            Gkj = G_full[k * n : (k + 1) * n, j * n : (j + 1) * n]
            Rk += Gkj @ Gkj.conj().T
        Rk_from_G.append(Rk)

    # Compare with R_k from setup()
    max_error = 0.0
    for k in range(nblocks):
        Rk_petsc = sbclo.Rk_list[k].copy()
        Rk_petsc.convert(PETSc.Mat.Type.DENSE)
        Rk_seq = res4py.distributed_to_sequential_matrix(Rk_petsc)
        Rk_np = Rk_seq.getDenseArray().copy()
        Rk_seq.destroy()
        Rk_petsc.destroy()

        err = np.linalg.norm(Rk_np - Rk_from_G[k]) / np.linalg.norm(
            Rk_from_G[k]
        )
        max_error = max(max_error, err)

    ksp.destroy()
    T_petsc.destroy()
    A_petsc.destroy()
    M_petsc.destroy()

    assert max_error < 1e-10, (
        f"R_k vs definition max relative error: {max_error:.2e}"
    )


def test_apply_hermitian_transpose_adjoint_identity(comm):
    r"""Test that <x, C y> = <C^* x, y> for random x, y."""
    sbclo, nN = _make_sbclo(comm)
    x, _ = pytest_utils.generate_random_vector(comm, nN)
    y, _ = pytest_utils.generate_random_vector(comm, nN)

    Cy = sbclo.apply(y)
    Cstar_x = sbclo.apply_hermitian_transpose(x)

    # <x, Cy>
    lhs = x.dot(Cy)
    # <C^*x, y>
    rhs = Cstar_x.dot(y)

    error = abs(lhs - rhs) / abs(lhs)

    x.destroy()
    y.destroy()
    Cy.destroy()
    Cstar_x.destroy()

    assert error < 1e-10, (
        f"<x, Cy> vs <C^*x, y> relative error: {error:.2e}"
    )


def _make_sbclo(comm, n=4, nblocks=5):
    """Helper: build a SBCLO from a random block-circulant T."""
    nN = n * nblocks
    T_np = comm.tompi4py().bcast(
        _build_block_circulant_np(n, nblocks, np.random.default_rng(42))
        if comm.getRank() == 0
        else None,
        root=0,
    )
    T_petsc = _numpy_to_petsc(comm, T_np)
    ksp = res4py.create_mumps_solver(T_petsc)
    T_linop = res4py.linear_operators.MatrixLinearOperator(
        T_petsc, ksp, nblocks
    )
    A_petsc = T_petsc.copy()
    A_petsc.scale(-1.0)
    Nl = compute_local_size(nN)
    M_petsc = PETSc.Mat().createAIJ(
        ((Nl, nN), (Nl, nN)), comm=comm
    )
    M_petsc.setUp()
    M_petsc.assemble()
    sbclo = SuperoptimalBlockCirculantLinearOperator(
        T_linop, A_petsc, M_petsc, omega=0.0, s=0.0
    )
    return sbclo, nN


def test_solve_inverts_apply(comm):
    r"""Test that solve(apply(x)) == x."""
    sbclo, nN = _make_sbclo(comm)
    x, _ = pytest_utils.generate_random_vector(comm, nN)

    Cx = sbclo.apply(x)
    x_back = sbclo.solve(Cx)

    x_seq = res4py.distributed_to_sequential_vector(x)
    x_back_seq = res4py.distributed_to_sequential_vector(x_back)
    error = np.linalg.norm(
        x_back_seq.getArray() - x_seq.getArray()
    ) / np.linalg.norm(x_seq.getArray())

    x_seq.destroy()
    x_back_seq.destroy()
    x.destroy()
    Cx.destroy()
    x_back.destroy()

    assert error < 1e-10, f"solve(apply(x)) roundtrip error: {error:.2e}"


def test_apply_inverts_solve(comm):
    r"""Test that apply(solve(x)) == x."""
    sbclo, nN = _make_sbclo(comm)
    x, _ = pytest_utils.generate_random_vector(comm, nN)

    Cinv_x = sbclo.solve(x)
    x_back = sbclo.apply(Cinv_x)

    x_seq = res4py.distributed_to_sequential_vector(x)
    x_back_seq = res4py.distributed_to_sequential_vector(x_back)
    error = np.linalg.norm(
        x_back_seq.getArray() - x_seq.getArray()
    ) / np.linalg.norm(x_seq.getArray())

    x_seq.destroy()
    x_back_seq.destroy()
    x.destroy()
    Cinv_x.destroy()
    x_back.destroy()

    assert error < 1e-10, f"apply(solve(x)) roundtrip error: {error:.2e}"


def test_solve_hermitian_transpose_inverts_apply_hermitian_transpose(comm):
    r"""Test that solve_hermitian_transpose(apply_hermitian_transpose(x)) == x."""
    sbclo, nN = _make_sbclo(comm)
    x, _ = pytest_utils.generate_random_vector(comm, nN)

    Cstar_x = sbclo.apply_hermitian_transpose(x)
    x_back = sbclo.solve_hermitian_transpose(Cstar_x)

    x_seq = res4py.distributed_to_sequential_vector(x)
    x_back_seq = res4py.distributed_to_sequential_vector(x_back)
    error = np.linalg.norm(
        x_back_seq.getArray() - x_seq.getArray()
    ) / np.linalg.norm(x_seq.getArray())

    x_seq.destroy()
    x_back_seq.destroy()
    x.destroy()
    Cstar_x.destroy()
    x_back.destroy()

    assert error < 1e-10, (
        f"solve_hermitian_transpose roundtrip error: {error:.2e}"
    )


def test_apply_hermitian_transpose_inverts_solve_hermitian_transpose(comm):
    r"""Test that apply_hermitian_transpose(solve_hermitian_transpose(x)) == x."""
    sbclo, nN = _make_sbclo(comm)
    x, _ = pytest_utils.generate_random_vector(comm, nN)

    Cinvstar_x = sbclo.solve_hermitian_transpose(x)
    x_back = sbclo.apply_hermitian_transpose(Cinvstar_x)

    x_seq = res4py.distributed_to_sequential_vector(x)
    x_back_seq = res4py.distributed_to_sequential_vector(x_back)
    error = np.linalg.norm(
        x_back_seq.getArray() - x_seq.getArray()
    ) / np.linalg.norm(x_seq.getArray())

    x_seq.destroy()
    x_back_seq.destroy()
    x.destroy()
    Cinvstar_x.destroy()
    x_back.destroy()

    assert error < 1e-10, (
        f"apply_hermitian_transpose(solve_hermitian_transpose(x)) "
        f"roundtrip error: {error:.2e}"
    )
