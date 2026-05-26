"""
Side module that computes (and caches) the Floquet eigendecomposition
and neutral-projection operator for ``RosslerPeriodic``.

The RosslerPeriodic class does not auto-compute these; the user is
responsible for either calling :func:`compute_eigendecomposition` to
recompute or :func:`load` to read a previously cached result, and
then setting

    eq.L, eq.Phi, eq.Psi, eq._neutral_proj

on the equation instance.

A typical recompute-and-save flow:

    from eigendecomp_rossler import compute_eigendecomposition, save
    L, Phi, Psi, neutral_proj = compute_eigendecomposition(eq)
    save("data/eigendecomp_cache.npz", L, Phi, Psi, neutral_proj, eq)

A typical load flow:

    from eigendecomp_rossler import load
    L, Phi, Psi, neutral_proj = load("data/eigendecomp_cache.npz")
    eq.L, eq.Phi, eq.Psi, eq._neutral_proj = L, Phi, Psi, neutral_proj
"""

import os
from typing import Tuple, Optional

import numpy as np

from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py


# ── Shift-invert Arnoldi block (HB) ──────────────────────────────────────────

def _shift_invert_eig(
    A_op,
    sigma: complex,
    n_evals: int,
    krylov_dim: int,
    n_harmonics: int,
    icntl: Optional[dict] = None,
) -> Tuple[np.ndarray, SLEPc.BV, np.ndarray, SLEPc.BV]:
    """Shift-and-invert Arnoldi about ``sigma``."""
    M = A_op.A.copy()
    M.scale(-1.0)
    size = M.getSizes()[0]
    I = res4py.create_AIJ_identity(A_op.A.getComm(), (size, size))
    M.axpy(sigma, I)
    I.destroy()
    ksp = res4py.create_mumps_solver(M, icntl=icntl if icntl is not None else {13: 1})
    res4py.check_lu_factorization(M, ksp)
    Linv = res4py.linear_operators.MatrixLinearOperator(
        M, ksp, nblocks=n_harmonics,
    )

    Dv, V = res4py.linalg.eig(
        Linv, Linv.solve, krylov_dim, n_evals,
        process_evals=lambda mu: sigma - 1.0 / mu,
    )
    Dw, W = res4py.linalg.eig(
        Linv, Linv.solve_hermitian_transpose, krylov_dim, n_evals,
        process_evals=lambda mu: np.conj(sigma) - 1.0 / mu,
    )
    Linv.destroy()

    V, W, Dv, Dw = res4py.linalg.match_right_and_left_eigenvectors(
        V, W, Dv, Dw,
    )
    return Dv, V, Dw, W


# ── Neutral Floquet eigentriples & projection operator ──────────────────────

def _compute_neutral_eigentriples(
    eq,
    n_strips: int = 4,
    n_evals: int = 4,
    krylov_dim: int = 50,
    neutral_omega: Optional[float] = None,
):
    """Find the neutrally stable Floquet directions at
    :math:`\\pm i k \\omega_n` for ``k = 0, ..., n_strips-1``, where
    :math:`\\omega_n = ` ``neutral_omega`` if provided, else
    :math:`\\omega = ` ``eq.omega``.

    Pass ``neutral_omega = 2 * eq.omega`` when building a 2T-periodic
    representation around a T-periodic base flow: the neutral subspace
    (orbit-tangent + its T-periodic harmonics) lives on integer
    multiples of the base-flow frequency :math:`\\omega_{\\rm base}`,
    which equals :math:`2\\,\\omega_{\\rm 2T}`.
    """
    comm = eq.A.A.getComm()
    n_harmonics = 2 * eq.nf + 1
    omega_strip = neutral_omega if neutral_omega is not None else eq.omega

    v_list, w_list = [], []
    for k in range(n_strips):
        for sign in ([0] if k == 0 else [1, -1]):
            sigma = sign * k * 1j * omega_strip
            Dv, V, _, W = _shift_invert_eig(
                eq.A, sigma, n_evals, krylov_dim, n_harmonics,
            )
            evals = np.diag(Dv)
            idx = np.argmin(np.abs(evals - sigma))

            v_col = V.getColumn(idx)
            v_list.append(v_col.copy())
            V.restoreColumn(idx, v_col)

            w_col = W.getColumn(idx)
            w_list.append(w_col.copy())
            W.restoreColumn(idx, w_col)

    n_neutral = len(v_list)
    state_dim = eq.A.A.getSizes()[0]

    V_neutral = SLEPc.BV().create(comm=comm)
    V_neutral.setSizes(state_dim, n_neutral)
    V_neutral.setType("mat")
    W_neutral = SLEPc.BV().create(comm=comm)
    W_neutral.setSizes(state_dim, n_neutral)
    W_neutral.setType("mat")

    for i in range(n_neutral):
        V_neutral.insertVec(i, v_list[i])
        W_neutral.insertVec(i, w_list[i])
        v_list[i].destroy()
        w_list[i].destroy()

    AV = V_neutral.copy()
    for i in range(n_neutral):
        v = V_neutral.getColumn(i)
        av = AV.getColumn(i)
        av = eq.A.apply(v, av)
        AV.restoreColumn(i, av)
        V_neutral.restoreColumn(i, v)
    WtAV = AV.dot(W_neutral)
    res4py.petscprint(comm, "W^* A V (neutral directions):")
    res4py.petscprint(comm, np.diag(WtAV.getDenseArray()))
    WtAV.destroy()
    AV.destroy()

    neutral_proj = res4py.linear_operators.ProjectionLinearOperator(
        V_neutral, W_neutral, complement=True, nblocks=n_harmonics,
    )
    return V_neutral, W_neutral, neutral_proj


# ── Full eigendecomposition ─────────────────────────────────────────────────

def compute_eigendecomposition(
    eq,
    n_evals: int = 20,
    krylov_dim: int = 60,
    sigma: complex = 0.0,
    n_neutral_strips: int = 4,
    neutral_omega: Optional[float] = None,
):
    """Compute the full Floquet eigendecomposition for ``eq``.

    Returns ``(L, Phi, Psi, neutral_proj)``.

    Pass ``neutral_omega = 2 * eq.omega`` for the 2T-periodic case so
    the neutral-strip loop hits the T-periodic neutrals at integer
    multiples of the base-flow frequency, not the 2T-basis frequency.
    """
    n_harmonics = 2 * eq.nf + 1

    _V_neut, _W_neut, neutral_proj = _compute_neutral_eigentriples(
        eq, n_strips=n_neutral_strips, neutral_omega=neutral_omega,
    )

    Dv, V, _, W = _shift_invert_eig(
        eq.A, sigma, n_evals, krylov_dim, n_harmonics,
    )

    evals = np.diag(Dv)
    half_omega = eq.omega / 2.0
    # Relative tolerance on the strip boundary: period-doubled modes
    # sit at |Im λ| = ω/2 by construction, and Arnoldi's converged
    # imaginary part can be ~1e-3 noisy, so a hard +1e-10 cutoff would
    # drop them.
    in_strip = np.abs(evals.imag) <= half_omega * (1.0 + 1e-2)
    idces = np.where(in_strip)[0]
    idces = idces[np.argsort(-evals[idces].real)]
    Dv = np.diag(evals[idces])
    V = res4py.bv_slice(V, idces.astype(np.int32))
    W = res4py.bv_slice(W, idces.astype(np.int32))

    # Only strip out the orbit-tangent neutral if one is actually
    # present in the surviving set — otherwise we'd accidentally
    # discard the master pair when the strip caught only them.
    evals_strip = np.diag(Dv)
    idx_neutral = int(np.argmin(np.abs(evals_strip)))
    if np.abs(evals_strip[idx_neutral]) < 1e-6:
        keep = np.delete(np.arange(len(evals_strip)), idx_neutral)
    else:
        keep = np.arange(len(evals_strip))
    Dv = np.diag(evals_strip[keep])
    V = res4py.bv_slice(V, keep.astype(np.int32))
    W = res4py.bv_slice(W, keep.astype(np.int32))

    L = np.diag(Dv)
    return L, V, W, neutral_proj


# ── On-disk cache I/O ───────────────────────────────────────────────────────

def _gather_bv_to_array(bv: SLEPc.BV) -> np.ndarray:
    ncols = bv.getSizes()[-1]
    N_hb = bv.getSizes()[0][-1]
    out = np.zeros((N_hb, ncols), dtype=np.complex128)
    for j in range(ncols):
        col = bv.getColumn(j)
        col_seq = res4py.distributed_to_sequential_vector(col)
        out[:, j] = col_seq.getArray()
        col_seq.destroy()
        bv.restoreColumn(j, col)
    return out


def _bv_from_array(
    arr: np.ndarray,
    state_dim: Tuple[int, int],
    comm: PETSc.Comm,
) -> SLEPc.BV:
    ncols = arr.shape[-1]
    bv = SLEPc.BV().create(comm=comm)
    bv.setSizes(state_dim, ncols)
    bv.setType("mat")
    for j in range(ncols):
        col = bv.getColumn(j)
        r0, r1 = col.getOwnershipRange()
        col.getArray()[:] = arr[r0:r1, j]
        bv.restoreColumn(j, col)
    return bv


def save(
    path: str,
    L: np.ndarray,
    Phi: SLEPc.BV,
    Psi: SLEPc.BV,
    neutral_proj,
    eq,
) -> None:
    """Save the eigendecomposition cache to ``path`` (numpy npz)."""
    comm = eq.A.A.getComm()

    Phi_arr = _gather_bv_to_array(Phi)
    Psi_arr = _gather_bv_to_array(Psi)
    Phi_neutral_arr = _gather_bv_to_array(neutral_proj.L.L.U)
    Psi_neutral_arr = _gather_bv_to_array(neutral_proj.L.L.V)

    if comm.getRank() == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            L=np.asarray(L, dtype=np.complex128),
            Phi=Phi_arr,
            Psi=Psi_arr,
            Phi_neutral=Phi_neutral_arr,
            Psi_neutral=Psi_neutral_arr,
            nf=int(eq.nf),
            nfb=int(eq.nfb),
            n=int(eq.n),
            c=float(eq.c),
            n_harmonics=int(2 * eq.nf + 1),
        )
        print(f"Saved eigendecomp cache → {path}")


def load(
    path: str,
    comm: Optional[PETSc.Comm] = None,
) -> Tuple[np.ndarray, SLEPc.BV, SLEPc.BV, "res4py.linear_operators.ProjectionLinearOperator"]:
    """Load an eigendecomposition cache from ``path``.

    Reconstructs the SLEPc BVs and the ProjectionLinearOperator.
    Returns ``(L, Phi, Psi, neutral_proj)`` ready to assign onto a
    freshly constructed ``RosslerPeriodic``.
    """
    if comm is None:
        comm = PETSc.COMM_WORLD

    cache = np.load(path)
    L = np.asarray(cache["L"], dtype=np.complex128)
    Phi_arr = cache["Phi"]
    Psi_arr = cache["Psi"]
    Phi_neutral_arr = cache["Phi_neutral"]
    Psi_neutral_arr = cache["Psi_neutral"]
    n_harmonics = int(cache["n_harmonics"])

    N_hb = Phi_arr.shape[0]
    state_dim = (res4py.compute_local_size(N_hb), N_hb)

    Phi = _bv_from_array(Phi_arr, state_dim, comm)
    Psi = _bv_from_array(Psi_arr, state_dim, comm)
    V_neutral = _bv_from_array(Phi_neutral_arr, state_dim, comm)
    W_neutral = _bv_from_array(Psi_neutral_arr, state_dim, comm)

    neutral_proj = res4py.linear_operators.ProjectionLinearOperator(
        V_neutral, W_neutral, complement=True, nblocks=n_harmonics,
    )
    return L, Phi, Psi, neutral_proj
