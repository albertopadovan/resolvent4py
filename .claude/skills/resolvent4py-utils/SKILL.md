---
name: resolvent4py-utils
description: Cross-cutting utilities — BV/Vec/Mat helpers, MPI scatter/gather, parallel I/O for harmonic-balanced data, KSP factories (MUMPS, GMRES+bjacobi), random data, time-stepping, complex-conjugate enforcement. Use when the user asks about parallel layout, COO/CSR assembly, harmonic-balanced matrices, MUMPS/PCBJACOBI options, or any low-level helper that the rest of resolvent4py builds on.
---

# `resolvent4py.utils`

The `utils` package is the foundation: every other module imports from
here. The top-level `from .utils import *` in
[__init__.py](../../../src/resolvent4py/__init__.py) re-exports
everything, so most of this surface lives at `res4py.<name>` (e.g.
`res4py.compute_local_size`, `res4py.create_mumps_solver`,
`res4py.write_to_file`, …).

## Files at a glance

| File | Responsibility |
|---|---|
| [bv.py](../../../src/resolvent4py/utils/bv.py) | Operations on `SLEPc.BV` (`bv_add`, `bv_conj/real/imag`, `bv_slice`, harmonic-balanced reshape) |
| [comms.py](../../../src/resolvent4py/utils/comms.py) | Local/global size math, dist⇄seq scatter for vec/mat, root-broadcast scatter |
| [errors.py](../../../src/resolvent4py/utils/errors.py) | `raise_not_implemented_error` decorator for LinearOperator stubs |
| [io.py](../../../src/resolvent4py/utils/io.py) | PETSc binary I/O for vec/mat/BV; harmonic-balanced matrix/vector/BV assembly from disk |
| [ksp.py](../../../src/resolvent4py/utils/ksp.py) | `create_mumps_solver`, `create_gmres_bjacobi_solver`, paired `check_*` helpers |
| [matrix.py](../../../src/resolvent4py/utils/matrix.py) | Mat factories, `convert_coo_to_csr`, harmonic resolvent generator, block extractors |
| [miscellaneous.py](../../../src/resolvent4py/utils/miscellaneous.py) | `get_mpi_type`, `petscprint` (rank-0 only), `get_memory_usage` |
| [random.py](../../../src/resolvent4py/utils/random.py) | Random PETSc sparse matrix / vector generators |
| [ssm.py](../../../src/resolvent4py/utils/ssm.py) | SSM-specific: `plot_convergence_radius`, `proper_radius` Newton solver |
| [time_stepping.py](../../../src/resolvent4py/utils/time_stepping.py) | `solve_ivp` (RK2/RK3, forward & adjoint), `compute_post_transient_solution`, FFT helpers |
| [vector.py](../../../src/resolvent4py/utils/vector.py) | `vec_real/imag`, `enforce_complex_conjugacy`, `check_complex_conjugacy`, vec⇄BV reshape |

## Themes

### Parallel layout (`comms.py`)

`compute_local_size(N)` is the canonical block-distribution rule:
`N//P` per rank, with the first `N mod P` ranks getting one extra.
Use this any time you create a `PETSc.Vec` / `Mat` and need the
local part to match what other library calls expect.

`compute_local_size_block_aligned(n, N)` is a **separate** rule for
operators that will be preconditioned with `PCBJACOBI(nblocks)`:
PETSc snaps sub-block boundaries to rank boundaries, so a
misaligned distribution silently produces the wrong sub-blocks.
This routine distributes per-block instead, requiring
`pool_size % nblocks == 0`. Use this only when building
block-Jacobi-preconditioned matrices.

### Single-rank gather / scatter (`comms.py`)

Three Vec-to-numpy distribution patterns coexist; pick by who
consumes the result:

- `distributed_to_sequential_vector(vec) → COMM_SELF Vec` — Allgather.
  Every rank ends up with the full vector.  Use when every rank
  *redundantly* runs the same compute on the gathered data.
- `gather_vec_to_rank(vec, dest_rank) → np.ndarray | None` — Gatherv
  to one root.  Returns the full numpy array on `dest_rank` and
  `None` everywhere else.  Use when *one* rank does the compute.
- `scatter_vec_from_rank(arr_on_source, target_vec, source_rank) →
  target_vec` — the inverse of `gather_vec_to_rank`.  Scatterv from
  one source so each rank receives only its `target_vec` ownership
  slice.

The pair `(gather_vec_to_rank, scatter_vec_from_rank)` round-trips
exactly (modulo dtype) and replaces the heavier
Allgather-then-everyone-computes pattern when the per-rank
compute would be redundant or memory-heavy.  This is what
`DifferentialEquation.evaluate_dynamics` and
`SpectralSubmanifold._evaluate_quadratic_rhs_root` use.

### KSP factories (`ksp.py`)

- `create_mumps_solver(A, icntl=None, cntl=None)` → KSP with
  `preonly + lu + mumps`. Pass `icntl={14: 50}` to bump workspace
  if MUMPS reports `INFO(1) = -9`. Setting `ICNTL(35)>0` triggers a
  user warning because BLR makes the solve inexact.
- `create_gmres_bjacobi_solver(A, nblocks, ...)` → GMRES with
  bjacobi(nblocks) preconditioner, each sub-block factored by
  MUMPS. Warns when `nblocks` is not a multiple of `comm.size`
  (block boundary misalignment again).
- Always pair with `check_lu_factorization` /
  `check_gmres_bjacobi_solver` to fail fast if MUMPS / GMRES did
  not actually converge.

### I/O (`io.py`)

PETSc binary format throughout (`PETSc.Viewer().createMPIIO`).
Top-level entry points:

- `write_to_file(filename, obj)` — accepts `Mat`, `Vec`, or `BV`
  (BV is unwrapped to its underlying `Mat`).
- `read_vector` / `read_dense_matrix` / `read_bv` — straightforward
  loaders requiring local/global sizes.
- `read_coo_matrix(filenames, sizes)` — reads `(rows.dat, cols.dat,
  vals.dat)` triples and assembles a sparse AIJ Mat via
  `convert_coo_to_csr`.
- `read_harmonic_balanced_{matrix,bv,vector}` — assembles a block
  Toeplitz from the Fourier coefficients of a time-periodic
  matrix/BV/vector. `real_bflow=True` means the file list contains
  only non-negative coefficients and the negative ones are filled
  in as conjugates.

### Matrix utilities (`matrix.py`)

- `convert_coo_to_csr` does the cross-rank shuffle from COO triples
  to CSR pointers/cols/vals. Uses `Alltoall` for counts and pairwise
  `Sendrecv` for data — chosen over `Alltoallv` to avoid request
  exhaustion / tag overflow on large machines.
- `assemble_harmonic_resolvent_generator(A, freqs, M=None)` builds
  `T = -i diag(ω_k) M + A` (with `M = I` by default) — the
  generator of harmonic resolvent dynamics.
- `extract_matrix_block(Mat, nblocks, rowblock, colblock)` pulls a
  single `(rowblock, colblock)` block out via two selector
  matMults (`Ir^H · Mat · Ic`).
- `extract_block_diagonal(Mat, nblocks)` zeros out off-diagonal
  blocks via `Σ_k E_k · A · E_k`.
- `extract_block_banded(Mat, nblocks, n_off_diags=0)` generalizes
  to a band: sums `Σ_{|i−j| ≤ n_off_diags} E_i · A · E_j`. Useful
  for block-tridiagonal preconditioners on harmonic-balanced matrices.

### Time stepping (`time_stepping.py`)

`solve_ivp(v, L, t0, tf, nsteps, method, m, adjoint,
X, periodic_forcing)` is a custom RK2/RK3 that:

- Integrates `dx/dt = L(t) x + f(t)` (or backward in time when
  `adjoint=True`, using `L.apply_hermitian_transpose`). **`L` is a
  `LinearOperator`, not a raw callable.** At every RK stage,
  `solve_ivp` calls `L.set_evaluation_time(t)` so any time-periodic
  inner operator (`TimePeriodicMatrixLinearOperator`, plus anything
  composed on top of one) is automatically kept in sync.
- Accepts a periodic forcing as `(F_hat, omegas)` Fourier modes;
  uses `ifft` internally per step.
- Saves every `m` steps into a SLEPc.BV `X` (or returns just the
  final-time vec when `m = -1`).

`compute_post_transient_solution(L, B, C, adjoint, tsim, nsave,
nperiods, omegas, x, Fhat, Yhat, X, ..., method='donothing', ...)`
computes the `T`-periodic steady state of `dx/dt = L(t) x + B(t)
f(t), y = C(t) x` for the user-supplied operators. All of `L`,
`B`, `C` may be time-varying (the routine threads
`set_evaluation_time(t_i)` through `B`'s pre-FFT sampling and `C`'s
per-snapshot projection). Two strategies via the `method` flag:

- `method='donothing'` (default, backward-compat): post-transient
  iteration. Cheap per call but requires the *shifted* Floquet
  spectrum to be strictly stable; diverges otherwise.
- `method='gmres'`: shoot-and-solve. Wraps `(I − Φ(T, 0))` as a
  PETSc shell (one homogeneous `solve_ivp` shot per `mult`), GMRES-
  solves `(I − Φ) x(0) = ∫₀ᵀ Φ(T, τ) (B f)(τ) dτ`, then integrates
  one more period from `x(0)` to fill `X`. Works for **any** shift
  where `I − Φ(T, 0)` is non-singular — including SSM-style shifts
  whose `s` lands inside the Floquet spectrum of `L_HB` (where
  `'donothing'` would diverge). Costs one extra integration up
  front plus one per GMRES iter.

Output FFT format follows `omegas`:
`np.min(omegas) == 0` ⇒ rfft (one-sided real signal); two-sided
omegas ⇒ full fft. Pass `harmonic_balancing_ordering=True` to
permute the two-sided output into `[-m, …, -1, 0, 1, …, m]` (the
`PeriodicDifferentialEquation` convention) instead of numpy's
`[0, 1, …, m, -m, …, -1]`.

`create_time_and_frequency_arrays(dt, omega, n_omegas, real)`
builds the matching `(tsim, nsave, omegas)` triple. The save grid
has `2·(n_omegas + 4)` samples per period — `+4` is a dealiasing
buffer for products of bandlimited signals (the `B(t) f(t)` and
`C(t) x(t)` convolutions). If `B(t)`/`C(t)` carry more than ~4
harmonics each, bump `n_omegas` accordingly.

### Vector utilities (`vector.py`)

- `enforce_complex_conjugacy(vec, nblocks)` / `check_complex_conjugacy`
  operate on vectors with block structure
  `[v_-m, ..., v_-1, v_0, v_1, ..., v_m]` (so `v_-i = conj(v_i)`).
  `nblocks` must be odd; `v_0` is forced real. **The communicator
  is inferred from `vec.getComm()`** — there is no `comm` parameter.
- `reshape_harmonic_balanced_vector_into_bv` and its inverse in
  [bv.py](../../../src/resolvent4py/utils/bv.py)
  (`reshape_bv_into_harmonic_balanced_vector`) switch between the
  stacked-vector and `n × nblocks` BV representations of the same
  harmonic-balanced state. Use the BV form when you need column-wise
  FFTs; use the vec form when you feed the state into a
  harmonic-balanced linear operator.

## Common pitfalls

- `enforce_complex_conjugacy` requires `nblocks` to be **odd** —
  it raises if you pass an even count.
- Avoid `compute_local_size(n*nblocks)` when the matrix will be
  preconditioned with `PCBJACOBI(nblocks)` — see
  `compute_local_size_block_aligned` above.
- `bv_roll(axis=0)` is currently broken with a circular-import
  guard ([bv.py:214](../../../src/resolvent4py/utils/bv.py#L214))
  — only `axis=-1` works today.
- Routines with optional output buffers (`y` for vec actions, `Y`
  for BV actions) allocate when given `None`. For tight loops, hand
  in a pre-allocated buffer.

## Tests

See [resolvent4py-tests](../resolvent4py-tests/SKILL.md).
Every file here has a matching `test_<file>.py` that compares
against numpy / scipy reference computations.
