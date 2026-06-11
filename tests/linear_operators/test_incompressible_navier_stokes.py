r"""Tests for :class:`.IncompressibleNavierStokesLinearOperator`.

The class reads its matrices from COO ``.dat`` files, so each test writes small
(non-physical) matrices to a shared temporary directory created/destroyed by
the root rank, builds the operator, and checks its action against a dense numpy
reference. Both the non-harmonic-balanced (``freqs is None``) and the
harmonic-balanced branches are exercised.

Conventions used by the class (see incompressible_navier_stokes.py):

- velocity DOFs are the first ``N_v`` indices, constraint DOFs the last
  ``N_c`` (``N = N_v + N_c``);
- the mass matrix is read both full (``Mm``) and restricted to its first
  ``N_v`` columns (``Lm``), so the test mass matrix is block-diagonal
  ``blkdiag(M_vv, 0)`` (no nonzeros in the constraint columns);
- the generator (``apply``) is ``P L^* (sM - A) L P`` and the resolvent
  (``solve``, after ``update_resolvent_operator``) is ``L^* (sM - A)^{-1} L``,
  both mapping velocity space to itself.
"""

import os
import shutil
import tempfile

import numpy as np
import scipy as sp
import pytest
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils

N_V, N_C = 10, 5
N = N_V + N_C


# ── shared temporary directory (root creates and destroys) ──────────────────
@pytest.fixture
def tmpdir_shared(comm):
    mpicomm = comm.tompi4py()
    path = tempfile.mkdtemp(prefix="res4py_ns_") if mpicomm.Get_rank() == 0 else None
    path = mpicomm.bcast(path, root=0)
    yield path
    mpicomm.Barrier()
    if mpicomm.Get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


# ── COO file writers (root writes the .dat files) ───────────────────────────
def _write_coo(comm, dirpath, stem, Anp):
    r"""Write a dense numpy matrix as COO triplet files and return the
    ``(rows, cols, vals)`` filename tuple. Exact zeros are fine: read_coo_matrix
    drops entries with ``|val| <= 1e-16``."""
    fnames = tuple(
        os.path.join(dirpath, f"{p}_{stem}.dat") for p in ("rows", "cols", "vals")
    )
    mpicomm = comm.tompi4py()
    if mpicomm.Get_rank() == 0:
        Nr, Nc = Anp.shape
        ii, jj = np.meshgrid(np.arange(Nr), np.arange(Nc), indexing="ij")
        arrays = (
            ii.reshape(-1).astype(PETSc.ScalarType),
            jj.reshape(-1).astype(PETSc.ScalarType),
            Anp.reshape(-1).astype(PETSc.ScalarType),
        )
        for fname, arr in zip(fnames, arrays):
            arr = np.ascontiguousarray(arr)
            vec = PETSc.Vec().createWithArray(arr, comm=PETSc.COMM_SELF)
            res4py.write_to_file(fname, vec)
            vec.destroy()
    mpicomm.Barrier()
    return fnames


def _write_coo_list(comm, dirpath, stem, coeffs):
    r"""Write a list of Fourier-coefficient matrices as numbered COO triplets
    (the ``filenames_lst`` layout read_harmonic_balanced_matrix expects)."""
    return [
        _write_coo(comm, dirpath, f"{stem}_{k:02d}", C)
        for k, C in enumerate(coeffs)
    ]


# ── numpy references ────────────────────────────────────────────────────────
def _block_toeplitz(coeffs, nblocks):
    r"""Block-Toeplitz matrix from one-sided coefficients ``[C0, C1, ...]`` with
    ``C_{-k} = conj(C_k)`` (the real_bflow=True convention). For 2 coefficients
    and nblocks=3 this is ``[[C0, conj(C1), 0], [C1, C0, conj(C1)], [0, C1,
    C0]]`` — the positive coefficient on the sub-diagonal."""
    nfb = len(coeffs) - 1
    br, bc = coeffs[0].shape
    M = np.zeros((br * nblocks, bc * nblocks), dtype=complex)
    for i in range(nblocks):
        for j in range(nblocks):
            d = i - j
            if 0 <= d <= nfb:
                blk = coeffs[d]
            elif 0 < -d <= nfb:
                blk = coeffs[-d].conj()
            else:
                continue
            M[i * br : (i + 1) * br, j * bc : (j + 1) * bc] = blk
    return M


def _reference(M, A, D, G, s, nblocks):
    r"""Dense reference for the generator (apply) and resolvent (solve) of the
    full operator. ``M, A, D, G`` are the full (block-Toeplitz, for HB)
    matrices. ``nblocks`` is 1 for the non-HB case."""
    nv = N_V * nblocks
    # Velocity columns: the first N_v columns of every block.
    vel_cols = np.concatenate([b * N + np.arange(N_V) for b in range(nblocks)])
    Lm = M[:, vel_cols]  # (N*nblocks) x (N_v*nblocks)

    DG = D @ G
    assert np.linalg.cond(DG) < 1e8, "DG must be invertible"
    Pp = np.eye(nv) - G @ sp.linalg.inv(DG) @ D

    sMA = s * M - A
    assert np.linalg.cond(sMA) < 1e8, "sM - A must be invertible"

    apply_ref = Pp @ Lm.conj().T @ sMA @ Lm @ Pp
    solve_ref = Lm.conj().T @ sp.linalg.inv(sMA) @ Lm
    return apply_ref, solve_ref


# ── matrix builders ─────────────────────────────────────────────────────────
def _vel_mass(rng, scale):
    r"""blkdiag(M_vv, 0): mass confined to the velocity block."""
    Mvv = scale * (np.eye(N_V) + 0.1 * rng.standard_normal((N_V, N_V)))
    M = np.zeros((N, N), dtype=complex)
    M[:N_V, :N_V] = Mvv
    return M


def _diag_dominant_A(rng, scale=1.0):
    r"""A with a strong negative diagonal so (sM - A) is invertible."""
    A = scale * 0.1 * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    A -= 8.0 * np.eye(N)
    return A


def _div_grad(rng, scale=1.0):
    D = scale * (
        rng.standard_normal((N_C, N_V)) + 1j * rng.standard_normal((N_C, N_V))
    )
    G = scale * (
        rng.standard_normal((N_V, N_C)) + 1j * rng.standard_normal((N_V, N_C))
    )
    return D, G


def _nonhb_matrices(seed=1):
    rng = np.random.default_rng(seed)
    M = _vel_mass(rng, 1.0)
    A = _diag_dominant_A(rng)
    D, G = _div_grad(rng)
    return M, A, D, G


def _hb_coeffs(seed=2):
    rng = np.random.default_rng(seed)
    A = [_diag_dominant_A(rng), _diag_dominant_A(rng, scale=0.05)]
    M = [_vel_mass(rng, 1.0), _vel_mass(rng, 0.05)]
    D0, G0 = _div_grad(rng)
    D1, G1 = _div_grad(rng, scale=0.05)
    return A, M, [D0, D1], [G0, G1]


def _saddle_A(A_uu, D, G):
    r"""Assemble the saddle-point block matrix A = [[A_uu, -G], [-D, 0]] so that
    sM - A = [[sI - A_uu, G], [D, 0]] when M = blkdiag(I, 0)."""
    A = np.zeros((N, N), dtype=complex)
    A[:N_V, :N_V] = A_uu
    A[:N_V, N_V:] = -G
    A[N_V:, :N_V] = -D
    return A


def _identity_mass():
    M = np.zeros((N, N), dtype=complex)
    M[:N_V, :N_V] = np.eye(N_V)
    return M


def _physical_matrices(seed=3):
    r"""Physically-structured (Stokes/saddle-point) matrices with identity
    velocity mass. With this structure the generator inverts the resolvent on
    the divergence-free subspace (GEN o RES equals the Leray projector P)."""
    rng = np.random.default_rng(seed)
    D, G = _div_grad(rng)
    assert np.linalg.cond(D @ G) < 1e8
    A_uu = -8.0 * np.eye(N_V) + 0.1 * (
        rng.standard_normal((N_V, N_V)) + 1j * rng.standard_normal((N_V, N_V))
    )
    return _saddle_A(A_uu, D, G), _identity_mass(), D, G


def _physical_hb_coeffs(seed=4):
    r"""Per-Fourier-mode saddle-point structure with constant (time-independent)
    velocity mass: A_k = [[A_uu_k, -G_k], [-D_k, 0]], M_0 = blkdiag(I, 0),
    M_1 = 0. GEN o RES then equals the harmonic-balanced Leray projector."""
    rng = np.random.default_rng(seed)
    A_c, D_c, G_c = [], [], []
    for k in range(2):
        D, G = _div_grad(rng, scale=(1.0 if k == 0 else 0.05))
        if k == 0:
            assert np.linalg.cond(D @ G) < 1e8
            A_uu = -8.0 * np.eye(N_V) + 0.1 * (
                rng.standard_normal((N_V, N_V))
                + 1j * rng.standard_normal((N_V, N_V))
            )
        else:
            A_uu = 0.05 * (
                rng.standard_normal((N_V, N_V))
                + 1j * rng.standard_normal((N_V, N_V))
            )
        A_c.append(_saddle_A(A_uu, D, G))
        D_c.append(D)
        G_c.append(G)
    # Constant mass: M_0 = blkdiag(I, 0), M_1 = 0.
    return A_c, [_identity_mass(), np.zeros((N, N), dtype=complex)], D_c, G_c


def _gen_inverts_res_error(comm, linop, vel):
    r"""Relative error of GEN(RES x) against x for a divergence-free x. Since
    GEN o RES = P (the Leray projector), the identity holds on the
    divergence-free subspace, so x is first projected with the operator's own P."""
    x, _ = pytest_utils.generate_random_vector(comm, vel)
    xdf = linop.P.apply(x)  # project onto the divergence-free subspace
    y = linop.solve(xdf)  # RES: L^* (sM - A)^{-1} L
    z = linop.apply(y)  # GEN: P L^* (sM - A) L P
    zs = res4py.distributed_to_sequential_vector(z)
    xs = res4py.distributed_to_sequential_vector(xdf)
    err = np.linalg.norm(
        zs.getArray() - xs.getArray()
    ) / np.linalg.norm(xs.getArray())
    for v in (x, xdf, y, z, zs, xs):
        v.destroy()
    return err


# ── non-harmonic-balanced branch ────────────────────────────────────────────
def test_ns_nonhb_actions(comm, tmpdir_shared):
    r"""apply (generator) and solve (resolvent) vs dense reference, complex s."""
    s = 0.7 + 0.3j
    M, A, D, G = _nonhb_matrices()
    apply_ref, solve_ref = _reference(M, A, D, G, s, nblocks=1)

    fA = _write_coo(comm, tmpdir_shared, "A", A)
    fM = _write_coo(comm, tmpdir_shared, "M", M)
    fD = _write_coo(comm, tmpdir_shared, "D", D)
    fG = _write_coo(comm, tmpdir_shared, "G", G)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG
    )
    linop.update_resolvent_operator(s)

    assert linop.get_nblocks() is None
    assert linop.get_dimensions()[0][-1] == N_V
    assert linop.get_real_flag() is False  # complex shift
    assert linop.get_block_cc_flag() is None

    x, xnp = pytest_utils.generate_random_vector(comm, N_V)
    y = linop.create_left_vector()
    err = [
        pytest_utils.compute_error_vector(comm, linop.apply, x, y, apply_ref.dot, xnp),
        pytest_utils.compute_error_vector(comm, linop.solve, x, y, solve_ref.dot, xnp),
    ]
    x.destroy()
    y.destroy()
    linop.destroy()
    assert np.linalg.norm(err) < 1e-8


def test_ns_nonhb_real_shift_flags(comm, tmpdir_shared):
    r"""With a real shift the non-HB operator reports real-valued; cc is None."""
    s = 0.7
    M, A, D, G = _nonhb_matrices()
    fA = _write_coo(comm, tmpdir_shared, "A", A)
    fM = _write_coo(comm, tmpdir_shared, "M", M)
    fD = _write_coo(comm, tmpdir_shared, "D", D)
    fG = _write_coo(comm, tmpdir_shared, "G", G)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG
    )
    assert linop.get_real_flag() is True
    assert linop.get_block_cc_flag() is None
    linop.destroy()


# ── harmonic-balanced branch ────────────────────────────────────────────────
def test_ns_hb_actions(comm, tmpdir_shared):
    r"""apply and solve vs the dense block-Toeplitz reference, complex s."""
    freqs = [0.0, 1.0]  # two (one-sided) frequencies -> nblocks = 3
    nblocks = 2 * (len(freqs) - 1) + 1
    if comm.getSize() != 1 and comm.getSize() % nblocks != 0:
        pytest.skip(
            f"HB branch needs MPI pool size 1 or divisible by nblocks={nblocks} "
            f"(run on 1, {nblocks}, {2 * nblocks}, ... ranks)"
        )
    s = 0.5 + 0.2j

    A_c, M_c, D_c, G_c = _hb_coeffs()
    A_full = _block_toeplitz(A_c, nblocks)
    M_full = _block_toeplitz(M_c, nblocks)
    D_full = _block_toeplitz(D_c, nblocks)
    G_full = _block_toeplitz(G_c, nblocks)
    apply_ref, solve_ref = _reference(M_full, A_full, D_full, G_full, s, nblocks)

    fA = _write_coo_list(comm, tmpdir_shared, "A", A_c)
    fM = _write_coo_list(comm, tmpdir_shared, "M", M_c)
    fD = _write_coo_list(comm, tmpdir_shared, "D", D_c)
    fG = _write_coo_list(comm, tmpdir_shared, "G", G_c)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG, freqs=freqs
    )
    linop.update_resolvent_operator(s)

    vel = N_V * nblocks
    assert linop.get_nblocks() == nblocks
    assert linop.get_dimensions()[0][-1] == vel
    assert linop.get_real_flag() is False  # HB is complex-valued
    assert linop.get_block_cc_flag() is False  # complex shift -> no cc

    x, xnp = pytest_utils.generate_random_vector(comm, vel)
    y = linop.create_left_vector()
    err = [
        pytest_utils.compute_error_vector(comm, linop.apply, x, y, apply_ref.dot, xnp),
        pytest_utils.compute_error_vector(comm, linop.solve, x, y, solve_ref.dot, xnp),
    ]
    x.destroy()
    y.destroy()
    linop.destroy()
    assert np.linalg.norm(err) < 1e-8


def test_ns_hb_real_shift_flags(comm, tmpdir_shared):
    r"""With a real shift the HB operator reports cc block structure (and is
    still complex-valued)."""
    freqs = [0.0, 1.0]
    nblocks = 2 * (len(freqs) - 1) + 1
    if comm.getSize() != 1 and comm.getSize() % nblocks != 0:
        pytest.skip(
            f"HB branch needs MPI pool size 1 or divisible by nblocks={nblocks} "
            f"(run on 1, {nblocks}, {2 * nblocks}, ... ranks)"
        )
    s = 0.5

    A_c, M_c, D_c, G_c = _hb_coeffs()
    fA = _write_coo_list(comm, tmpdir_shared, "A", A_c)
    fM = _write_coo_list(comm, tmpdir_shared, "M", M_c)
    fD = _write_coo_list(comm, tmpdir_shared, "D", D_c)
    fG = _write_coo_list(comm, tmpdir_shared, "G", G_c)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG, freqs=freqs
    )
    assert linop.get_nblocks() == nblocks
    assert linop.get_real_flag() is False
    assert linop.get_block_cc_flag() is True
    linop.destroy()


# ── generator inverts resolvent (on the divergence-free subspace) ────────────
def test_ns_nonhb_generator_inverts_resolvent(comm, tmpdir_shared):
    r"""The generator inverts the resolvent on the divergence-free subspace:
    GEN(RES x) = x for x = P x. Over the full velocity space GEN o RES is the
    Leray projector P, so x is projected first; this requires the physical
    saddle-point structure (A = [[A_uu, -G], [-D, 0]], M = blkdiag(I, 0))."""
    s = 0.6 + 0.4j
    A, M, D, G = _physical_matrices()
    fA = _write_coo(comm, tmpdir_shared, "A", A)
    fM = _write_coo(comm, tmpdir_shared, "M", M)
    fD = _write_coo(comm, tmpdir_shared, "D", D)
    fG = _write_coo(comm, tmpdir_shared, "G", G)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG
    )
    err = _gen_inverts_res_error(comm, linop, N_V)
    linop.destroy()
    assert err < 1e-8


def test_ns_hb_generator_inverts_resolvent(comm, tmpdir_shared):
    r"""Same generator/resolvent inverse check, harmonic-balanced case."""
    freqs = [0.0, 1.0]
    nblocks = 2 * (len(freqs) - 1) + 1
    if comm.getSize() != 1 and comm.getSize() % nblocks != 0:
        pytest.skip(
            f"HB branch needs MPI pool size 1 or divisible by nblocks={nblocks} "
            f"(run on 1, {nblocks}, {2 * nblocks}, ... ranks)"
        )
    s = 0.4 + 0.3j
    A_c, M_c, D_c, G_c = _physical_hb_coeffs()
    fA = _write_coo_list(comm, tmpdir_shared, "A", A_c)
    fM = _write_coo_list(comm, tmpdir_shared, "M", M_c)
    fD = _write_coo_list(comm, tmpdir_shared, "D", D_c)
    fG = _write_coo_list(comm, tmpdir_shared, "G", G_c)

    linop = res4py.linear_operators.IncompressibleNavierStokesLinearOperator(
        comm, s, fA, fM, (N_V, N_C), fname_Dm=fD, fname_Gm=fG, freqs=freqs
    )
    err = _gen_inverts_res_error(comm, linop, N_V * nblocks)
    linop.destroy()
    assert err < 1e-8
