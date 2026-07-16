"""
Compute the 2-D spectral submanifold of the Hopf3D toy model and dump
the SSM polynomial + latent-space dynamics coefficients to
``data/ssm_cache.npz`` for downstream reuse in ``generate_figures.py``.

Mirrors the SSM-construction block of ``demonstrate_ssm.py`` — same
model parameters, master indices, polynomial order, and gauge — but
runs no trajectory comparison here.  The cache contents match the
schema :class:`SpectralSubmanifoldROM` consumes (``PS``, ``W``, ``gs``,
``multiindices``, ``Lams``), plus everything else you need to rebuild
the diff-eq (mu, alpha, beta) and re-derive derived quantities
(``R``, ``rho_domain``, geometric-decay-fit) later without re-solving
the SSM.

Run with:
    mpirun -n <nprocs> python save_ssm.py
"""

import os
import numpy as np

from petsc4py import PETSc
import resolvent4py as res4py

from toy_model_differential_equation import Hopf3D


comm = PETSc.COMM_WORLD


# ── Model + SSM parameters (kept in sync with demonstrate_ssm.py) ──────────
mu = 1.0 / 20.0
alpha = 1
beta = 0.0
r = 2                          # 2-D SSM (Hopf complex-conjugate pair)
m = 30                         # polynomial expansion order
scaling_ssm = 0.2              # gauge scaling used inside SSM.solve
# `latent_space_components = None`  → natural parameterisation, every order
# of g(s) retained (== the old `conj_to_linear_dynamics = False`).
# Use `[1]` to force a linear latent flow (legacy conj-to-linear), or e.g.
# `[1, 3]` for a cubic normal form.
latent_space_components = None
rho_fraction = 0.5             # domain radius as fraction of R


# ── Build the differential equation and its eigendecomposition ────────────
diff_eq = Hopf3D(mu=mu, alpha=alpha, beta=beta)
L, Phi, Psi = diff_eq.compute_eigendecomposition()

# Slice the two master modes (Hopf pair at indices 0, 1).
idces = [0, 1]
V = res4py.bv_slice(Phi, idces)
W = res4py.bv_slice(Psi, idces)
L_master = np.diag(L)[idces]


# ── Compute the SSM ───────────────────────────────────────────────────────
SSM = res4py.spectral_submanifold.SpectralSubmanifold(
    diff_eq, r, m,
    latent_space_components=latent_space_components,
)
SSM.solve(V, W, L_master, scaling=scaling_ssm)

R, orders, coeff_sums, slope, intercept = SSM.estimate_convergence_radius()
rho_domain = rho_fraction * R
res4py.petscprint(
    comm,
    f"Convergence radius R = {R:.4f}, "
    f"rho_domain = {rho_fraction}·R = {rho_domain:.4f}",
)
res4py.petscprint(comm, f"Master eigenvalues Λ = {SSM.Lams}")


# ── Gather PETSc/SLEPc data to numpy for on-disk storage ─────────────────
n_state_pair = diff_eq.get_state_dimension()
n_state = int(n_state_pair[-1])                    # global state dim


def _gather_vec(vec):
    """Distributed PETSc.Vec → 1-D numpy array on every rank."""
    v_seq = res4py.distributed_to_sequential_vector(vec)
    arr = v_seq.getArray().copy()
    v_seq.destroy()
    return arr


def _gather_bv(bv):
    """SLEPc.BV → (n_state, ncols) numpy array on every rank."""
    ncols = bv.getSizes()[-1]
    out = np.zeros((n_state, ncols), dtype=np.complex128)
    for j in range(ncols):
        col = bv.getColumn(j)
        out[:, j] = _gather_vec(col)
        bv.restoreColumn(j, col)
    return out


# Polynomial coefficients:  PS has shape (n_terms, n_state)  (autonomous).
PS = np.stack([_gather_vec(p) for p in SSM.ps], axis=0)

# Left master eigenvectors used inside the ROM's encoder.
W_np = _gather_bv(SSM.W)

# Multi-indices, master eigenvalues, latent dynamics coefficients.
multiindices = np.array(SSM.ssm_multiindices, dtype=np.int64)   # (n_terms, r)
Lams = np.asarray(SSM.Lams)                                     # (r,)
gs = np.asarray(SSM.gs)                                         # (n_terms, r)


# ── Save ─────────────────────────────────────────────────────────────────
if comm.getRank() == 0:
    os.makedirs("data", exist_ok=True)
    outpath = "data/ssm_cache.npz"
    np.savez_compressed(
        outpath,
        # SSM arrays consumed by SpectralSubmanifoldROM
        PS=PS,
        W=W_np,
        multiindices=multiindices,
        Lams=Lams,
        gs=gs,
        conj_to_linear_dynamics=bool(SSM.conj_to_linear_dynamics),
        # Model parameters (to rebuild Hopf3D or plot annotations)
        mu=mu, alpha=alpha, beta=beta,
        n_state=n_state,
        # SSM-construction hyperparameters
        r=r, m=m, scaling=scaling_ssm,
        # Convergence-radius diagnostics
        R=R,
        rho_domain=rho_domain,
        rho_fraction=rho_fraction,
        orders=np.asarray(orders),
        coeff_sums=np.asarray(coeff_sums),
        slope=slope,
        intercept=intercept,
    )
    print(f"Saved SSM cache → {outpath}")
    print(f"  PS shape          = {PS.shape}")
    print(f"  W shape           = {W_np.shape}")
    print(f"  gs shape          = {gs.shape}")
    print(f"  multiindices shape= {multiindices.shape}")


comm.Barrier()
os._exit(0)
