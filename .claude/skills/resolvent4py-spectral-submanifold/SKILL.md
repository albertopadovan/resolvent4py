---
name: resolvent4py-spectral-submanifold
description: Spectral submanifold reduction for quadratic dynamical systems q'=Aq+B(q,q) — polynomial parametrization P(s) tangent to a chosen spectral subspace, with intrinsic dynamics ds/dt=Lambda*s+g(s). Use when the user asks about SSMs, invariant manifolds, latent-space dynamics, or polynomial expansions around an eigenspace.
---

# `resolvent4py.spectral_submanifold`

Compute spectral submanifolds (SSMs) for quadratic dynamical systems
`q̇ = Aq + B(q,q)`. An SSM is an invariant manifold tangent to an
`r`-dimensional spectral subspace of `A`; the SSM polynomial
`P(s) = Σ p_j s^j` and the intrinsic dynamics
`ṡ = Λs + g(s) = Σ g_j s^j` are computed order-by-order from the
homological equations.

## Files

| File | Class / API |
|---|---|
| [differential_equation.py](../../../src/resolvent4py/spectral_submanifold/differential_equation.py) | `DifferentialEquation` ABC — user supplies `evaluate_linear_term`, `evaluate_quadratic_term`, `solve_linear_system`. |
| [spectral_submanifold.py](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py) | `SpectralSubmanifold` — solver. Public methods: `solve`, `decode`, `encode`, `latent_space_dynamics`, `estimate_convergence_radius`. |

## How it works

### `DifferentialEquation` (you implement this)

The abstract base requires three methods.  **The signatures are
mixed**: `evaluate_quadratic_term` takes/returns **rank-replicated
numpy arrays**; the other two are PETSc Vec.  This split is intentional
— it makes the per-pair quadratic loop in `solve` parallelisable
across pairs (see "Parallel quadratic-RHS evaluation" below).

- `evaluate_linear_term(t, q, y=None) -> PETSc.Vec` — `Aq` on a
  distributed Vec.  Free to use any PETSc parallel machinery
  internally.
- `evaluate_quadratic_term(t, q1, q2, y=None) -> np.ndarray` — `B(q1,
  q2)` on **numpy arrays of length `state_dim`**.  Must be symmetric
  in `q1, q2`.  Callers gather inputs to numpy before calling and
  scatter the result back — *you don't do any gather/scatter inside
  this method*.  A `y` buffer is provided for in-place writes when
  the caller wants to avoid allocation.
- `solve_linear_system(s, b, x=None) -> PETSc.Vec` — `(sI − A)⁻¹ b`
  for arbitrary complex shift `s`, on distributed Vecs.

`evaluate_dynamics(t, q, y=None) -> PETSc.Vec` is provided by the
base and handles the contract bridge: it calls
`evaluate_linear_term` (PETSc Vec interface) and then gathers `q`
to rank 0 via `gather_vec_to_rank`, evaluates the bilinear in
numpy on rank 0 only, and `scatter_vec_from_rank`s the result into
a temp Vec for the final `axpy`.  Override this only if your RHS
has terms beyond `Aq + B(q,q)`.

`poly_deg` must be 2 (only quadratic nonlinearities are currently
supported — see the `__init__` guard in
[spectral_submanifold.py:75](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py#L75)).

### `PeriodicDifferentialEquation` mixin (HB lift)

For time-periodic systems the factory in
`DifferentialEquation.__new__` inserts this mixin ahead of the
user's subclass in the MRO when `periodic_diffeq=(omegas, time,
is_period_doubling)` is provided.  The mixin overrides
`evaluate_linear_term` (PETSc Vec interface, internal SLEPc BV
workspace) and `evaluate_quadratic_term` (**pure numpy**, cached
IDFT weight matrix `_E_idft[i, k] = exp(1j·ω_k·t_i)`).  The
bilinear pipeline is

```
q1.reshape((nblocks, N)).T  → (N, nblocks)
@ E.T                       → (N, nt)   # IDFT
per-t super().evaluate_quadratic_term(t_i, qk1_arr, qk2_arr)
@ E.conj() / nt             → (N, nblocks)  # DFT
.T.reshape(-1)              → (N·nblocks,)
```

No SLEPc BV or PETSc Vec is allocated for the bilinear; the only
remaining collectives are inside the user's
`evaluate_linear_term`.  Subclasses must implement
`evaluate_quadratic_term` on numpy state-vectors of length `N` =
the per-time-instant state dimension (not `N·nblocks`).

### `SpectralSubmanifold(diff_eq, r, m, conj_to_linear_dynamics=False, n_workers=None)`

Constructor builds the multi-index combinatorics and (if `n_workers
> 1`) the per-rank partition + routing table for the parallel pair
loop.  See **Parallel quadratic-RHS evaluation** below.

### `SpectralSubmanifold.solve(Phi, Psi, Lams, scaling, verbose)`

Given right eigenvectors `Φ`, left eigenvectors `Ψ`, and eigenvalues
`Λ` of the chosen master subspace (precomputed via
[res4py.linalg.eig](../resolvent4py-linalg/SKILL.md) +
`match_right_and_left_eigenvectors`), this drives the SSM expansion
to order `m`:

1. Enumerate all multi-indices `j ∈ ℕ^r` with `|j| ≤ m`
   (`compute_multiindices`).
2. For each multi-index in increasing-order order:
   - assemble RHS contributions from quadratic interactions
     (`ssm_quad_rhs_idc`) and from the previously-known nonlinear
     dynamics terms (`ssm_nonlin_dynmc`),
   - project: `g_j = Ψ* rhs` (or zero if conjugacy to linear
     dynamics is requested),
   - solve `(λ_j I − A) p_j = rhs − Φ g_j` via
     `solve_linear_system`,
   - sanity-check `|Ψ* p_j| < 1e-6` (raises `ValueError` if not —
     usually means you need to adjust `scaling`).
3. Cache `ps`, `gs`, `Lams`, `V`, `W` on the object for
   `decode/encode/latent_space_dynamics` to use later.

`scaling` rescales eigenvectors as `V = Φ·s, W = Ψ/s`. This is the
knob to push around if convergence is poor or
`|Ψ* p_j|` exceeds tolerance.

### Parallel quadratic-RHS evaluation (`n_workers > 1`)

The inner pair loop on line 384 of
[spectral_submanifold.py](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py)
is a reduction `rhs = Σ_{(i,j) ∈ pairs} B(ps[i], ps[j])` with no
inter-iteration dependency, so it's embarrassingly parallel.  Pass
`n_workers ∈ [1, world_size]` to the constructor to distribute the
pairs across a subset of ranks (default = `world_size`).

- **Worker placement**: *decimation* across the world rank space —
  `worker w` is at `world_rank = w · world_size // n_workers`.  On
  a typical launch with adjacent ranks on the same node, this puts
  one worker per node, which balances inbound bandwidth during the
  gather phase.
- **Pair partition** (deterministic, computed once at `__init__`):
  `self.my_ssm_quad_rhs_idc[j_idx] = full_pairs[worker_id::n_workers]`.
  Empty on non-worker ranks.
- **Routing table** `self._workers_needing[j_idx][ps_index] = [world
  ranks that need it]`: identical on every rank by deterministic
  construction; no MPI to sync.

The `solve` loop branches:

- `n_workers == 1`: serial path.  Each pair is processed via
  `_evaluate_quadratic_rhs_root(t, q1_vec, q2_vec, y_vec)`, a
  PETSc-Vec ↔ numpy adapter that does
  `gather_vec_to_rank(·, 0)` → numpy `evaluate_quadratic_term` on
  rank 0 only → `scatter_vec_from_rank(·, y_vec, 0)`.  Non-root
  ranks participate in collectives only; no redundant compute, no
  per-rank workspace blowup.
- `n_workers > 1`: parallel path in
  `_evaluate_quadratic_rhs_parallel(j_idx, ps, rhs)`.
  1. **Gather**: per `ps[i] ∈ routing.keys()`, one
     `gather_vec_to_rank` to the *primary* worker
     (`routing[i][0]`), then the primary issues blocking `Send`s to
     any secondary workers.
  2. **Local pair loop**: workers iterate their `my_ssm_quad_rhs_idc[j_idx]`
     and accumulate into a numpy buffer via numpy
     `evaluate_quadratic_term`.
  3. **Allreduce** the per-worker buffer on `world_comm`.
  4. **Inject** each rank's local share into `rhs` via `setValues`
     (no Scatterv needed — every rank already has the full numpy
     after Allreduce).

The same multi-index recursion uses the result, so the choice of
`n_workers` only changes wire/compute distribution, not the
numerical answer (modulo ULP-level reordering of the sum across
pair partitions).

**When to drop `r` below `world_size`**: only when
`evaluate_quadratic_term` allocates so much workspace per call that
you cannot afford one copy per rank (NUMA / memory-bandwidth
saturation, or PDE-discretization bilinears with large FFT
buffers).  Otherwise `n_workers = world_size` wins.

### Other methods

- `decode(s)` → full-state `Σ_j p_j Π_k s_k^{j_k}`.
- `encode(q)` → latent-space coordinates `Ψ* q`.
- `latent_space_dynamics(t, s)` → `Λs + Σ_j g_j s^j`, suitable for
  `scipy.integrate.solve_ivp`.
- `estimate_convergence_radius()` — fits `log10(C_k)` vs `k` (where
  `C_k = Σ_{|j|=k} ||p_j||_1`) using the **Theil–Sen** estimator.
  This is robust to near-resonance outliers; ordinary least squares
  is skewed by a single near-resonant `p_j` blowing up. Skips the
  first 33% of orders (low-order terms aren't asymptotic).

## Multi-index combinatorics (helpers, internal)

The private helpers `_generate_constrained_compositions`,
`_generate_quadratic_pairs`, and `_generate_nonlinear_dynamics_pairs`
enumerate all `(i, l)` multi-index pairs for the polynomial
convolutions in the homological equations. They're cached on the
object as `ssm_multiindices`, `ssm_quad_rhs_idc`,
`ssm_nonlin_dynmc` at construction time so `solve` doesn't redo
combinatorics per call.

## Common pitfalls

- `ps[0]` is a zero vector (manifold passes through the origin); the
  next `r` entries `ps[1..r]` are the columns of `V`. Order-`k>1`
  terms start at index `r+1`. Don't index by integer if you can
  avoid it — go through `_get_multiindex_index` like
  [spectral_submanifold.py:376](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py#L376).
- `_check_eigen_triplets` is a *warning*, not an error — it prints
  if biorthogonality / eigenvalue residuals exceed tolerance but
  proceeds. Watch the stdout when running `solve`.
- The intercept and slope from `estimate_convergence_radius` feed
  into `proper_radius()` in
  [utils/ssm.py](../../../src/resolvent4py/utils/ssm.py) to decide
  what fraction of `R_estimate` is safe for a given `manifold_tol`.
- **`evaluate_quadratic_term` takes numpy arrays, NOT PETSc Vecs.**
  This is a recent contract change.  Subclasses that still gather +
  do numpy work + scatter inside the method will work but waste a
  full round-trip per call (the caller already gathers).  Audit
  your subclass before benchmarking.
  Updated: `Hopf3D`, autonomous KSE, periodic KSE, Rössler.
  **Not yet updated**: `JetFlowPeriodic` (bespoke rank-0 + Bcast
  HB pipeline; needs its own conversion before it works with the
  new SSM path).
- For `n_workers > 1`, the parallel pair loop runs the per-time-
  instant bilinear on **COMM_SELF-equivalent inputs** (numpy
  arrays).  Subclasses that allocate per-call workspace on a
  COMM_WORLD BV / Vec inside the bilinear will deadlock or produce
  wrong results.  Operate only on the input numpy arrays.

## Tests

[resolvent4py-tests-spectral-submanifold](../resolvent4py-tests-spectral-submanifold/SKILL.md)
exists but contains only an empty `__init__.py` at present —
SSM functionality is exercised through the
[examples/ssm/](../../../examples/ssm/) and
[examples/ssm_periodic/](../../../examples/ssm_periodic/)
use-cases instead.
