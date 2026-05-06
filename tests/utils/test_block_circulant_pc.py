"""
Tests for the Strang block-circulant preconditioner exposed by
``create_gmres_superoptimal_solver`` in ``utils/ksp.py``.

Two layers:

  * ``test_numpy_block_circulant_inverse_is_exact`` — pure numpy.
    Pins down the FFT convention: for a *true* block-circulant matrix
    built from random storage-order blocks, the FFT-diagonalisation
    algorithm (fft → solve diag → ifft) reproduces the exact inverse.
    No PETSc; runs on a single rank.

  * ``test_strang_pc_matches_numpy_reference`` — PETSc-level.
    Builds a small harmonic-resolvent operator (block-Toeplitz
    + position-dependent diagonal shift), runs the actual
    ``create_gmres_superoptimal_solver`` PC apply against a numpy
    reference, and checks they agree to ~floating-point precision.

  * ``test_strang_pc_accelerates_gmres`` — PETSc-level.
    Same operator. With the PC, GMRES should converge in much fewer
    iterations than without. Just verifies the PC is *useful*; the
    exact factor is problem-dependent.
"""
import numpy as np
import pytest
import resolvent4py as res4py
from petsc4py import PETSc

from resolvent4py.utils.comms import (
    compute_local_size,
    scatter_array_from_root_to_all,
)
from resolvent4py.utils.matrix import (
    convert_coo_to_csr,
    assemble_matrix_from_coo,
)


# ---------------------------------------------------------------------------
# Pure numpy: pins down the FFT convention.
# ---------------------------------------------------------------------------


def _numpy_block_circulant_inverse(c_blocks, v):
    """Apply C^{-1} v where C is a true block-circulant matrix with blocks
    ``c_blocks[d]`` for ``d = 0..n-1`` (storage order, so that
    ``C[i, j] = c_blocks[(i - j) mod n]``).

    Uses the FFT diagonalisation that matches the Strang PC implementation:
    eigenvalue at storage frequency k is ``c_hat[k] = sum_d c_blocks[d] *
    exp(-2 pi i k d / n)``, i.e. ``numpy.fft.fft(c_blocks, axis=0)[k]``.
    """
    n = len(c_blocks)
    N = c_blocks[0].shape[0]

    # DFT of the per-block matrix sequence along axis 0 (storage axis).
    c_stack = np.stack(c_blocks, axis=0)            # (n, N, N)
    c_hat = np.fft.fft(c_stack, axis=0)             # (n, N, N)

    v_blocks = v.reshape(n, N)
    v_hat = np.fft.fft(v_blocks, axis=0)
    x_hat = np.empty_like(v_hat)
    for k in range(n):
        x_hat[k] = np.linalg.solve(c_hat[k], v_hat[k])
    return np.fft.ifft(x_hat, axis=0).reshape(-1)


def test_numpy_block_circulant_inverse_is_exact():
    """For a true block-circulant matrix, fft → solve diag → ifft IS the
    exact inverse (up to floating-point)."""
    rng = np.random.default_rng(42)
    n, N = 5, 4

    c_blocks = [
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        # diagonally-dominant so each c_hat[k] is well-conditioned
        + 5.0 * np.eye(N) * (1 if d == 0 else 0)
        for d in range(n)
    ]

    # Build C explicitly: C[i, j] = c_blocks[(i - j) mod n]
    C = np.zeros((n * N, n * N), dtype=complex)
    for i in range(n):
        for j in range(n):
            d = (i - j) % n
            C[i * N : (i + 1) * N, j * N : (j + 1) * N] = c_blocks[d]

    v = rng.standard_normal(n * N) + 1j * rng.standard_normal(n * N)

    x_alg = _numpy_block_circulant_inverse(c_blocks, v)
    x_direct = np.linalg.solve(C, v)

    rel = np.linalg.norm(x_alg - x_direct) / np.linalg.norm(x_direct)
    assert rel < 1e-10, f"FFT-diag algorithm error: {rel:.3e}"


# ---------------------------------------------------------------------------
# Helpers for the PETSc-level tests.
# ---------------------------------------------------------------------------


def _numpy_to_petsc_aij(comm, A_np):
    """Build a distributed PETSc AIJ from a numpy matrix known on all ranks."""
    Nr, Nc = A_np.shape
    rows_coo, cols_coo, vals_coo = None, None, None
    if comm.getRank() == 0:
        # Force contiguous, correctly-typed arrays. mpi4py >=4.1 uses
        # DLPack to view the buffers and rejects non-contiguous inputs;
        # np.nonzero / fancy indexing don't always produce contiguous
        # arrays on the path through np.asarray.
        r, c = np.nonzero(A_np)
        rows_coo = np.ascontiguousarray(r, dtype=PETSc.IntType)
        cols_coo = np.ascontiguousarray(c, dtype=PETSc.IntType)
        vals_coo = np.ascontiguousarray(
            np.asarray(A_np[r, c]), dtype=PETSc.ScalarType
        )
    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    vals = scatter_array_from_root_to_all(vals_coo)
    Nrl = compute_local_size(Nr)
    Ncl = compute_local_size(Nc)
    sizes = ((Nrl, Nr), (Ncl, Nc))
    rp, cs, vs = convert_coo_to_csr([rows, cols, vals], sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rp, cs))
    M.setValuesCSR(rp, cs, vs, True)
    M.assemble()
    return M


def _build_harmonic_resolvent_setup(comm, N, nblocks, omega, s, seed=0):
    """Build a small harmonic-resolvent test problem.

    Returns (A_petsc, M_petsc, Tinv_petsc, A_l_list_np, M0_np, T_np_dense),
    where:
      * ``A_petsc`` is the block-Toeplitz Jacobian with block (k, j) = A_{k-j}
      * ``M_petsc`` is block-diagonal with M_0 on every block
      * ``Tinv_petsc`` is ``s*M - T`` where
            T_{kj} = -i k omega M_0 delta_{kj} + (signed) A blocks
        i.e. the operator that the user's pipeline produces
      * the numpy artefacts let us cross-check.
    """
    m = (nblocks - 1) // 2
    rng = np.random.default_rng(seed)

    # Diagonally-dominant random Fourier coefficients A_{-m}, ..., A_m.
    A_l = []
    for l in range(-m, m + 1):
        Al = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        if l == 0:
            Al += 4.0 * np.eye(N)
        A_l.append(Al)

    M0 = np.eye(N) + 0.05 * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )

    nN = nblocks * N
    A_block_toep = np.zeros((nN, nN), dtype=complex)
    for k_phys in range(-m, m + 1):
        for j_phys in range(-m, m + 1):
            d = k_phys - j_phys
            if -m <= d <= m:
                ks = k_phys + m
                js = j_phys + m
                A_block_toep[ks * N : (ks + 1) * N, js * N : (js + 1) * N] = (
                    A_l[d + m]
                )

    M_block_diag = np.zeros((nN, nN), dtype=complex)
    for ks in range(nblocks):
        M_block_diag[ks * N : (ks + 1) * N, ks * N : (ks + 1) * N] = M0

    # Tinv = sM - T, with T_{kj} = -i k omega M0 delta_{kj} + A_{k-j}
    # (matches what assemble_harmonic_resolvent_generator + sM - T produces
    # when the user does A.scale(-1.0); we replicate that overall sign.)
    Tinv_np = np.zeros_like(A_block_toep)
    for k_phys in range(-m, m + 1):
        ks = k_phys + m
        Tinv_np[ks * N : (ks + 1) * N, ks * N : (ks + 1) * N] += (
            (s + 1j * k_phys * omega) * M0
        )
    Tinv_np += A_block_toep

    # Bcast for determinism
    mpi = comm.tompi4py()
    A_block_toep = mpi.bcast(A_block_toep, root=0)
    M_block_diag = mpi.bcast(M_block_diag, root=0)
    Tinv_np = mpi.bcast(Tinv_np, root=0)
    A_l = [mpi.bcast(a, root=0) for a in A_l]
    M0 = mpi.bcast(M0, root=0)

    A_petsc = _numpy_to_petsc_aij(comm, A_block_toep)
    M_petsc = _numpy_to_petsc_aij(comm, M_block_diag)
    Tinv_petsc = _numpy_to_petsc_aij(comm, Tinv_np)
    return A_petsc, M_petsc, Tinv_petsc, A_l, M0, Tinv_np


def _strang_apply_numpy(A_l, M0, nblocks, s, v):
    """Reference numpy implementation of the Strang block-circulant PC apply.
    Mirrors the implementation in ``_setup_superoptimal_pc`` exactly so the
    PETSc apply can be cross-checked against it."""
    m = (nblocks - 1) // 2
    N = M0.shape[0]

    # c_hat[k] = s*M0  -  sum_{l=-m..m} A_l * exp(-2 pi i k l / n), k=0..n-1
    c_hat = []
    for k in range(nblocks):
        ck = s * M0.copy()
        for l in range(-m, m + 1):
            ck -= np.exp(-2j * np.pi * k * l / nblocks) * A_l[l + m]
        c_hat.append(ck)

    # The user's storage convention puts block j at physical harmonic j-m,
    # but the FFT/IFFT does not care about that interpretation: it just
    # transforms across the storage axis. Both the algorithm and this
    # reference apply fft to the raw storage layout, so they must agree.
    v_blocks = v.reshape(nblocks, N)
    v_hat = np.fft.fft(v_blocks, axis=0)
    x_hat = np.empty_like(v_hat)
    for k in range(nblocks):
        x_hat[k] = np.linalg.solve(c_hat[k], v_hat[k])
    return np.fft.ifft(x_hat, axis=0).reshape(-1)


# ---------------------------------------------------------------------------
# PETSc-level tests.
# ---------------------------------------------------------------------------


def test_strang_pc_matches_numpy_reference(comm):
    """The PC apply (PETSc, possibly multi-rank) must agree with a pure
    numpy Strang reference on the same operator and RHS."""
    N, nblocks = 6, 5
    omega = 1.7
    s = 0.5 + 1.2j

    A, M, Tinv, A_l, M0, Tinv_np = _build_harmonic_resolvent_setup(
        comm, N, nblocks, omega, s, seed=0
    )

    # Build the PC by going through the public factory, then yank out the
    # underlying KSP's PCSHELL and apply it directly on a known RHS.
    ksp = res4py.create_gmres_superoptimal_solver(
        Tinv, A, M, nblocks, omega, s, monitor=False
    )

    nN = N * nblocks
    rng = np.random.default_rng(7)
    b_np = rng.standard_normal(nN) + 1j * rng.standard_normal(nN)
    b_np = comm.tompi4py().bcast(b_np, root=0)

    # Distribute b to a PETSc vec matching Tinv's layout.
    b = Tinv.createVecRight()
    r0, r1 = b.getOwnershipRange()
    b.setValues(
        np.arange(r0, r1, dtype=PETSc.IntType),
        np.asarray(b_np[r0:r1], dtype=PETSc.ScalarType),
    )
    b.assemble()

    # Apply the PC (only) to b: PC = pc.apply
    pc = ksp.getPC()
    y = b.duplicate()
    pc.apply(b, y)

    # Reference numpy result.
    y_ref = _strang_apply_numpy(A_l, M0, nblocks, s, b_np)

    # Gather y to all ranks and compare.
    y_seq = res4py.distributed_to_sequential_vector(y)
    y_arr = y_seq.getArray().copy()
    y_seq.destroy()

    rel = np.linalg.norm(y_arr - y_ref) / np.linalg.norm(y_ref)

    b.destroy()
    y.destroy()
    A.destroy()
    M.destroy()
    Tinv.destroy()
    ksp.destroy()

    assert rel < 1e-8, (
        f"PETSc PC apply does not match numpy reference: rel = {rel:.3e}"
    )


def test_strang_pc_accelerates_gmres(comm):
    """With the Strang PC, GMRES on a small harmonic-resolvent operator
    should converge much faster than without. We don't assert a precise
    iteration count — just that the PC actually accelerates."""
    N, nblocks = 6, 5
    omega = 1.7
    s = 0.5 + 1.2j

    A, M, Tinv, A_l, M0, Tinv_np = _build_harmonic_resolvent_setup(
        comm, N, nblocks, omega, s, seed=1
    )
    nN = N * nblocks
    Nl = compute_local_size(nN)
    rtol = 1e-8

    # ---- Baseline: GMRES on Tinv with no preconditioner.
    ksp_nopc = PETSc.KSP().create(comm=comm)
    ksp_nopc.setOperators(Tinv)
    ksp_nopc.setType("gmres")
    ksp_nopc.setTolerances(rtol=rtol, atol=rtol, max_it=2000)
    ksp_nopc.getPC().setType("none")
    ksp_nopc.setUp()

    b = res4py.generate_random_petsc_vector((Nl, nN))
    x = b.duplicate()
    ksp_nopc.solve(b, x)
    its_nopc = ksp_nopc.getIterationNumber()
    reason_nopc = ksp_nopc.getConvergedReason()
    ksp_nopc.destroy()

    # ---- With the Strang block-circulant PC.
    ksp_pc = res4py.create_gmres_superoptimal_solver(
        Tinv, A, M, nblocks, omega, s, rtol=rtol, atol=rtol, monitor=False
    )
    ksp_pc.setTolerances(rtol=rtol, atol=rtol, max_it=2000)
    x_pc = b.duplicate()
    ksp_pc.solve(b, x_pc)
    its_pc = ksp_pc.getIterationNumber()
    reason_pc = ksp_pc.getConvergedReason()

    b.destroy()
    x.destroy()
    x_pc.destroy()
    A.destroy()
    M.destroy()
    Tinv.destroy()
    ksp_pc.destroy()

    assert reason_pc > 0, (
        f"Strang-PC GMRES did not converge: reason={reason_pc}, its={its_pc}"
    )
    assert reason_nopc > 0 or its_nopc >= its_pc, (
        f"Baseline GMRES status: reason={reason_nopc} its={its_nopc}; "
        f"PC GMRES: its={its_pc}"
    )
    # PC should beat baseline by a healthy margin (very loose: 2x).
    assert its_pc * 2 <= its_nopc, (
        f"Strang PC did not accelerate GMRES: nopc={its_nopc}, pc={its_pc}"
    )
