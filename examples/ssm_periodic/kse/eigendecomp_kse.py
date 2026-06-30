"""
Side module that computes (and caches) the Floquet eigendecomposition
and neutral-projection operator for ``KuramotoSivashinskyPeriodic``.

The KSE class itself no longer auto-computes these; the user is
responsible for either calling :func:`compute_eigendecomposition` to
recompute or :func:`load` to read a previously cached result, and
then setting

    eq.L, eq.Phi, eq.Psi, eq._neutral_proj

on the equation instance.

A typical recompute-and-save flow:

    from eigendecomp_kse import compute_eigendecomposition, save
    L, Phi, Psi, neutral_proj = compute_eigendecomposition(eq)
    save("data/eigendecomp_cache.npz", L, Phi, Psi, neutral_proj, eq)

A typical load flow:

    from eigendecomp_kse import load
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
    """Shift-and-invert Arnoldi about ``sigma``.

    Builds ``M = (sigma*I - L_HB)`` from the assembled harmonic
    resolvent generator ``A_op``, MUMPS-factorizes it, then runs
    Arnoldi on ``Linv = M^{-1}`` for both ``Linv`` and its Hermitian
    transpose.  Returns matched and biorthogonalised ``(Dv, V, Dw, W)``.
    """
    M = A_op.A.copy()
    M.scale(-1.0)
    size = M.getSizes()[0]
    I = res4py.create_AIJ_identity(A_op.A.getComm(), (size, size))
    M.axpy(sigma, I)
    I.destroy()
    # ICNTL(13)=1 disables MUMPS's parallel root-node factorization
    # (ScaLAPACK), which was triggering an MPICH assertion on macOS.
    ksp = res4py.create_mumps_solver(
        M, icntl=icntl if icntl is not None else {13: 1}
    )
    res4py.check_lu_factorization(M, ksp)
    Linv = res4py.linear_operators.MatrixLinearOperator(
        M,
        ksp,
        nblocks=n_harmonics,
    )

    Dv, V = res4py.linalg.eig(
        Linv,
        Linv.solve,
        krylov_dim,
        n_evals,
        process_evals=lambda mu: sigma - 1.0 / mu,
    )
    Dw, W = res4py.linalg.eig(
        Linv,
        Linv.solve_hermitian_transpose,
        krylov_dim,
        n_evals,
        process_evals=lambda mu: np.conj(sigma) - 1.0 / mu,
    )
    Linv.destroy()

    V, W, Dv, Dw = res4py.linalg.match_right_and_left_eigenvectors(
        V,
        W,
        Dv,
        Dw,
    )
    return Dv, V, Dw, W


# ── Neutral Floquet eigentriples & projection operator ──────────────────────


def _compute_neutral_eigentriples(
    eq,
    n_strips: int = 6,
    n_evals: int = 4,
    krylov_dim: int = 50,
    neutral_omega: Optional[float] = None,
):
    """Find the neutrally stable Floquet directions at
    :math:`\\pm i k \\omega_n` for ``k = 0, ..., n_strips-1``, where
    :math:`\\omega_n = ` ``neutral_omega`` if provided, else
    :math:`\\omega = ` ``eq.omega``.

    Pass ``neutral_omega = 2 * eq.omega`` when building a 2T-periodic
    eigendecomposition: the T-periodic orbit-tangent neutrals sit at
    integer multiples of the *original* orbit frequency ω_base, which
    in the 2T basis is ``2 * eq.omega``.  Using the default would
    deflate at half-integer multiples of ω_base — including the
    period-doubling mode at ``λ = i·ω_base/2 = i·eq.omega``.

    Returns ``(V_neutral, W_neutral, neutral_proj)`` where the BVs hold
    the biorthogonalised right/left neutral eigenvectors and
    ``neutral_proj`` is the corresponding complement
    :class:`ProjectionLinearOperator`.
    """
    comm = eq.A.A.getComm()
    n_harmonics = 2 * eq.nf + 1
    omega_strip = neutral_omega if neutral_omega is not None else eq.omega

    # σ for every strip carries a tiny real offset (1e-8) so that
    # L_HB − σ·I is non-singular even when the orbit-tangent direction is
    # an exact kernel mode of L_HB at λ ≡ 0 (mod i·ω_HB).  Without the
    # offset, MUMPS factorisation of an exactly-singular matrix returns
    # garbage Krylov vectors instead of true eigenpairs.  1e-8 is far
    # below any meaningful eigvalue, so accuracy is unaffected.
    sigma_offset = 1e-8
    v_list, w_list = [], []
    for k in range(n_strips):
        for sign in [0] if k == 0 else [1, -1]:
            sigma = sign * k * 1j * omega_strip + sigma_offset
            Dv, V, _, W = _shift_invert_eig(
                eq.A,
                sigma,
                n_evals,
                krylov_dim,
                n_harmonics,
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

    # Sanity print: diagonal of W^* A V should be the neutral eigenvalues.
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
        V_neutral,
        W_neutral,
        complement=True,
        nblocks=n_harmonics,
    )
    return V_neutral, W_neutral, neutral_proj


# ── Full eigendecomposition (neutrals + principal strip + remove DC) ────────


def compute_eigendecomposition(
    eq,
    n_evals: int = 20,
    krylov_dim: int = 60,
    sigma: complex = 0.0,
    n_neutral_strips: int = 6,
    neutral_omega: Optional[float] = None,
):
    """Compute the full Floquet eigendecomposition for ``eq``.

    Returns ``(L, Phi, Psi, neutral_proj)``:
      * ``L`` — 1D numpy array of complex Floquet exponents.
      * ``Phi``, ``Psi`` — right / left HB eigenvectors as SLEPc BVs,
        biorthogonalised.
      * ``neutral_proj`` — complement
        :class:`ProjectionLinearOperator` for the neutral directions.

    Does the neutral-strip computation first (used by
    ``eq.solve_linear_system`` to deflate gauge directions), then a
    main shift-invert Arnoldi at ``sigma``, filters to the principal
    Floquet strip ``|Im λ| <= ω/2``, sorts by descending real part,
    and removes the residual neutral.
    """
    n_harmonics = 2 * eq.nf + 1

    if n_neutral_strips > 0:
        _V_neut, _W_neut, neutral_proj = _compute_neutral_eigentriples(
            eq,
            n_strips=n_neutral_strips,
            neutral_omega=neutral_omega,
        )
    else:
        # No neutral-direction deflation — caller is responsible for
        # making sure the shift-invert target ``sigma`` is far enough
        # from i·k·ω for the eig() to converge to non-neutral modes.
        neutral_proj = None

    Dv, V, _, W = _shift_invert_eig(
        eq.A,
        sigma,
        n_evals,
        krylov_dim,
        n_harmonics,
    )

    # Principal strip filter and descending Re(λ) sort
    evals = np.diag(Dv)
    half_omega = eq.omega / 2.0
    in_strip = np.abs(evals.imag) <= half_omega * (1.0 + 1e-2)
    idces = np.where(in_strip)[0]
    idces = idces[np.argsort(-evals[idces].real)]
    Dv = np.diag(evals[idces])
    V = res4py.bv_slice(V, idces.astype(np.int32))
    W = res4py.bv_slice(W, idces.astype(np.int32))

    # σ=0 + neutral-strip path: the orbit-tangent neutral that was
    # deflated in shift-invert still leaks back into the principal-strip
    # result with significant magnitude (~1e-4 for KSE — much larger than
    # rossler's ~1e-10 because of integration-error differences).  We
    # cannot gate on a fixed |λ| threshold; instead, *unconditionally*
    # drop the smallest-|λ| mode in the strip whenever neutral deflation
    # was used.  The actual period-doubling mode lives at slightly larger
    # |λ| (negative real) and survives this delete cleanly.
    if n_neutral_strips > 0:
        evals_strip = np.diag(Dv)
        idx_neutral = int(np.argmin(np.abs(evals_strip)))
        keep = np.delete(np.arange(len(evals_strip)), idx_neutral)
        Dv = np.diag(evals_strip[keep])
        V = res4py.bv_slice(V, keep.astype(np.int32))
        W = res4py.bv_slice(W, keep.astype(np.int32))

    L = np.diag(Dv)
    return L, V, W, neutral_proj


# ── On-disk cache I/O ───────────────────────────────────────────────────────


def _gather_bv_to_array(bv: SLEPc.BV) -> np.ndarray:
    """Gather an HB SLEPc BV to a numpy array of shape ``(N_hb, ncols)``.

    ``distributed_to_sequential_vector`` replicates each column on all
    ranks, so the resulting array is identical on every rank.
    """
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
    """Build a parallel SLEPc BV from a replicated ``(N_hb, ncols)`` numpy
    array.  Each rank fills its owned slice from the array's rows.
    """
    ncols = arr.shape[-1]
    bv = SLEPc.BV().create(comm=comm)
    bv.setSizes(state_dim, ncols)
    bv.setType("mat")

    # petsc4py's Vec.getArray() returns a writable view; writes are
    # in-place and the array is auto-restored when the view is GC'd
    # (no explicit Vec.restoreArray method exists).
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
    """Save the eigendecomposition cache to ``path`` (numpy npz).

    The BVs are gathered to identical numpy arrays on every rank;
    only rank 0 writes to disk.
    """
    comm = eq.A.A.getComm()

    Phi_arr = _gather_bv_to_array(Phi)
    Psi_arr = _gather_bv_to_array(Psi)
    if neutral_proj is not None:
        Phi_neutral_arr = _gather_bv_to_array(neutral_proj.L.L.U)
        Psi_neutral_arr = _gather_bv_to_array(neutral_proj.L.L.V)
    else:
        # No neutral subspace requested — stash zero-column placeholders
        # so the cache shape is still parseable on load.
        N_hb = Phi_arr.shape[0]
        Phi_neutral_arr = np.zeros((N_hb, 0), dtype=np.complex128)
        Psi_neutral_arr = np.zeros((N_hb, 0), dtype=np.complex128)

    if comm.getRank() == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            L=np.asarray(L, dtype=np.complex128),
            Phi=Phi_arr,
            Psi=Psi_arr,
            Phi_neutral=Phi_neutral_arr,
            Psi_neutral=Psi_neutral_arr,
            # Metadata for validation
            nf=int(eq.nf),
            nfb=int(eq.nfb),
            n=int(eq.n),
            n_pts=int(eq.n_pts),
            n_harmonics=int(2 * eq.nf + 1),
        )
        print(f"Saved eigendecomp cache → {path}")


def load(
    path: str,
    comm: Optional[PETSc.Comm] = None,
) -> Tuple[
    np.ndarray,
    SLEPc.BV,
    SLEPc.BV,
    "res4py.linear_operators.ProjectionLinearOperator",
]:
    """Load an eigendecomposition cache from ``path``.

    Reconstructs the SLEPc BVs and the
    :class:`ProjectionLinearOperator` using the current MPI
    communicator.  Returns ``(L, Phi, Psi, neutral_proj)`` ready to
    assign onto a freshly constructed ``KuramotoSivashinskyPeriodic``.
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
    if Phi_neutral_arr.shape[-1] == 0:
        # Cache was written without a neutral subspace.
        neutral_proj = None
    else:
        V_neutral = _bv_from_array(Phi_neutral_arr, state_dim, comm)
        W_neutral = _bv_from_array(Psi_neutral_arr, state_dim, comm)
        neutral_proj = res4py.linear_operators.ProjectionLinearOperator(
            V_neutral,
            W_neutral,
            complement=True,
            nblocks=n_harmonics,
        )
    return L, Phi, Psi, neutral_proj
