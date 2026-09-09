---
name: resolvent4py-tests
description: The resolvent4py pytest suite — layout, conftest fixtures, pytest_utils error helpers, the MPI execution caveat, and per-directory coverage (linear operators, linalg, model reduction, utils) with the tolerance bands and pitfalls each one depends on. Use when the user asks how the suite is organized, how a class or routine is validated, what tolerance a test uses, or how to add a new test.
---

# `tests/`

Pytest suite for resolvent4py. Mirrors the source tree one-to-one:
`src/resolvent4py/<x>/` is exercised by `tests/<x>/`. Runs under MPI and
silences stdout on non-root ranks.

```
tests/
├── conftest.py              # fixtures (comm, sizes, random matrices)
├── pytest_utils.py          # random data + numpy/PETSc error metrics
├── linalg/  linear_operators/  model_reduction/  utils/
└── spectral_submanifold/    # empty — see "SSM" below
```

## Execution

**Never run bare `pytest`** — `mpiexec -n 2 pytest tests/`. Tests assert on
parallel layout. On Stampede3, `petsc4py`/`mpi4py` come from environment
modules, so launch through the user's wrapper script rather than directly.

## Fixtures (`conftest.py`)

- `comm` / `rank_size` (session-scoped) — `PETSc.COMM_WORLD`, `(rank, size)`.
- `square_matrix_size` `(50, 50)`, `rectangular_matrix_size` `(50, 20)` —
  parametrized, currently single-valued, marked `pytest.mark.local`.
- `square_random_matrix`, `square_stable_random_matrix`,
  `rectangular_random_matrix` — yield `(Apetsc, Apython)` pairs. The
  "stable" variant shifts eigenvalues to `Re(λ) < 0` (needed by propagator
  and time-stepping tests).
- `test_output_dir` — tmp-path directory for I/O round-trips.

A `pytest_configure` hook unregisters the terminal reporter on non-root ranks
so only rank 0 prints. Fixtures own the PETSc objects they create — don't
destroy a fixture-provided matrix inside a test.

## Error helpers (`pytest_utils.py`)

| Helper | Returns |
|---|---|
| `generate_random_matrix(comm, (Nr, Nc), complex=True)` | `(Apetsc, Apython)` — distributed sparse + sequential dense, identical contents |
| `generate_stable_random_matrix(...)` | Same, eigenvalues shifted to negative real part |
| `generate_random_bv` / `generate_random_vector` | `(X_BV, Xnumpy)` / `(x_petsc, xnumpy)` |
| `compute_error_vector(comm, petsc_action, x, y, python_action, xpython)` | Relative `‖y − f(x)‖ / ‖f(x)‖` |
| `compute_error_vector_shell_operator(...)` | Same, for `mult(A, x, y)`-style shell ops |
| `compute_error_bv(...)` | Relative error between a BV action and its numpy reference |

Nearly every assertion is one relative norm against a tolerance of `1e-8` to
`1e-14`. These helpers gather to rank 0 via `distributed_to_sequential_*`;
the numpy reference is computed redundantly on every rank by design.

## Dominant pattern

```python
def test_X_on_Y(comm, square_random_matrix):
    Apetsc, Apython = square_random_matrix
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp=...)
    x, xpython = pytest_utils.generate_random_vector(comm, N)
    y = linop.create_left_vector()
    error = pytest_utils.compute_error_vector(
        comm, linop.apply, x, y, Apython.dot, xpython,
    )
    # ... destroy resources ...
    assert error < 1e-10
```

Variants: `*_on_vectors` vs `*_on_bvs` (`apply` vs `apply_mat`), `*_y_none`
(omitting the output buffer auto-allocates), `*_repeated_apply` and
`*_varying_column_counts` (cached intermediate buffers).

---

## `tests/linear_operators/` — one file per concrete subclass

Build the operator, build the equivalent dense numpy matrix, compare each
`apply*`/`solve*` against `numpy.dot` / `scipy.linalg.inv`.

| File | Class | Notes |
|---|---|---|
| `test_matrix.py` | `MatrixLinearOperator` | square + rectangular; `Y=None`; repeated-apply consistency |
| `test_low_rank.py` | `LowRankLinearOperator` | `Σ` is **not** square (`(rr, rc)`) — the general factorization |
| `test_low_rank_updated.py` | `LowRankUpdatedLinearOperator` | Woodbury solve; cached intermediate BV across repeated calls and varying column counts |
| `test_propagator.py` | `PropagatorLinearOperator` | RK2 vs RK3 bands on `square_stable_random_matrix` (LTI case); time-periodic forward/adjoint vs `scipy.integrate.solve_ivp` RK45. `_make_periodic_A_coeffs` builds a stable `A(t)` |
| `test_time_periodic_matrix.py` | `TimePeriodicMatrixLinearOperator` | `{vec, BV} × {real, complex}` for real-A and complex-A; `set_evaluation_time(t)`; the `⟨v, Lw⟩ = ⟨L*v, w⟩` identity in all 8 combinations |
| `test_petsc_python.py` | `PetscPythonLinearOperator` | shell `mult`/`multHermitian` only; no `solve` |
| `test_product.py` | `ProductLinearOperator` | mixed `solve` + `apply` + `apply_hermitian_transpose` + `apply` over 4 ops |
| `test_projection.py` | `ProjectionLinearOperator` | `complement=False/True`; idempotency `P²=P` |
| `test_real_flag.py` | `check_if_real_valued` heuristic | real/complex `Matrix` and `LowRank`; `ShiftAndScale` with real vs complex `α` |
| `test_shift_and_scale.py` | `ShiftAndScaleLinearOperator` | complex `α=2+3j`, `β=−1.5+0.5j`; vec + BV |

**The action-list pattern.** Most files build parallel lists
`actions_petsc = [linop.apply, linop.apply_hermitian_transpose, linop.solve,
linop.solve_hermitian_transpose]` and `actions_python = [A.dot,
A.conj().T.dot, Ainv.dot, Ainv.conj().T.dot]`, then iterate.

| Operator | Typical tol |
|---|---|
| `Matrix`, `Projection`, `Shift+Scale`, `Product`, `PetscPython` | `1e-10` |
| `LowRank` | `1e-8` |
| `LowRankUpdated` | `1e-8` (Woodbury accumulates error) |
| `Propagator` RK3 (`dt=1e-5`, `tf=2.1`, LTI) / RK2 / time-periodic | `1e-11` / `1e-7` / `1e-6` |
| `TimePeriodicMatrix` (apply, and the adjoint identity) | `1e-12` |

Two tests go beyond "does it match": `test_low_rank_updated_repeated_apply_mat`
calls `apply_mat` twice with the same `X` and asserts bitwise-identical
results (`< 1e-14`), proving the cached BV carries no state between calls;
`test_low_rank_updated_varying_column_counts` uses counts `[3, 7, 3]` to
exercise the lazy-resize path in `_get_intermediate_bv`.

## `tests/linalg/`

| File | Cases |
|---|---|
| `test_eigendecomposition.py` | Shift-invert Arnoldi vs `scipy.linalg.eig`; biorthogonalization (`W*V = I`, `W*AV = diag(λ)`) |
| `test_randomized_svd.py` | Singular values and triplet residuals vs `scipy.linalg.svd` |
| `test_resolvent_analysis_rsvd_dt.py` | `compute_post_transient_solution` and the full RSVD-dt SVD |
| `test_harmonic_resolvent_time_vs_freq.py` | Time-domain post-transient integration vs the HB solve `(iΩ − A_HB)⁻¹ F̂`, for real T-periodic `A(t)`. Four tests: forward/adjoint × `B=C=I` / time-periodic `B(t), C(t)`. Tol `1e-4` on the well-resolved range (`n_compare = 10` interior modes at `n_omegas = n_pert = 24`) |

`_build_resolvent_operator(comm, A, omega)` builds `L = (iωI − A)` with MUMPS
attached so `L.solve` acts as `R(iω)`. Eigenvalue tests then call
`res4py.linalg.eig(L, L.solve, krylov_dim, r,
process_evals=lambda x: 1j*omega - 1/x)` to recover eigenvalues of `A`.

Tolerances: `eig` `5e-1` **percent** on dominant eigenvalues;
`match_right_and_left_eigenvectors` `1e-12` for `W*V − I` but `1e-6` for
`W*AV − diag(λ)` (two solver calls); `randomized_svd` `5e-1` percent on
singular values, `1e-10` on triplet residuals; RSVD-dt `1e-2` (RK truncation).
These assume the small `(50, 50)` fixture — tighten them if it grows.

`test_resolvent_analysis_rsvd_dt.py` randomizes `omega`, `n_omegas`, `dt` per
run and broadcasts from rank 0 so all ranks agree; it covers `real=True`
(positive freqs) and `real=False` (two-sided), forward and adjoint.

## `tests/model_reduction/`

One file, `test_balanced_truncation.py`: build `(A, B, C)` with random `5×5`
`A`; piecewise Gauss–Legendre quadrature (10-point per sub-interval over the
first `30 · domega` band with `domega = 0.05`, then 5-point over 50
sub-intervals with `domega *= 10`, refining near the resolvent peak at
`ω = 0`); an `L_generator` closing over `A` that yields
`(L, L.solve_mat, (L.destroy,))` per `ω`; then `compute_gramian_factors` →
`compute_balanced_projection(r=1)` → `assemble_reduced_order_tensors`.
Reference is scipy's `solve_continuous_lyapunov` for `X` and `Y`, then
`eig(X·Y)` for Hankel singular values. Both assertions are `< 5%` **relative
percent** (the `100 *` factor is intentional) — loose because Gauss–Legendre
on a truncated interval can't exactly reproduce `∫_{-∞}^∞ R(ω)BB*R(ω)*`.

Canonical generator pattern: `L_gen = partial(L_generator, A=Apetsc)`
replicated once per quadrature point.

## `tests/utils/` — largest module

| File | What's covered |
|---|---|
| `test_bv.py` | `bv_add/conj/real/imag/slice/roll`; harmonic-balanced reshape (vec ↔ BV) round-trips |
| `test_comms.py` | `compute_local_size`; dist⇄seq round-trips; scatter from root with/without explicit `locsize` |
| `test_io.py` | vec/mat/BV round-trips; harmonic-balanced matrix in 3 modes; error on too-few blocks |
| `test_ksp.py` | MUMPS accuracy + `check_lu_factorization`; GMRES+bjacobi one-iteration property; tighter rtol → smaller residual |
| `test_matrix.py` | Mat factories, hermitian-transpose, `mat_solve_hermitian_transpose`, harmonic resolvent generator, `extract_block_diagonal` |
| `test_miscellaneous.py` | `get_mpi_type` for every supported dtype; raises on unsupported |
| `test_random.py` | sizes, real/complex flag, square + rectangular |
| `test_time_stepping.py` | RK3 forced/adjoint vs `scipy.integrate.solve_ivp`; `compute_post_transient_solution` `gmres` ↔ `donothing` cross-check on stable LTI and LTP, forward + adjoint |
| `test_vector.py` | `vec_real/imag` (in-place + copy), `enforce_complex_conjugacy` round-trip, even-`nblocks` raises, harmonic-balanced reshape |

**Block-Jacobi alignment.** `test_gmres_bjacobi_block_diagonal_one_iter`
asserts GMRES converges in *exactly* one iteration when bjacobi is exact. To
keep sub-blocks aligned with rank ownership it picks `nblocks = 5` in serial
and `nblocks = comm.size` otherwise (one block per rank). Change that rule and
GMRES will silently take more than one iteration.

**Harmonic-balanced I/O** has three flavors plus an error case:
`len(filenames_lst) == 1, real_bflow=True` ⇒ block-diagonal (all blocks
`A_0`); `real_bflow=True` with `[A_0 … A_{nfb}]` ⇒ Toeplitz with
`A_{−k} = conj(A_k)` filled automatically; `real_bflow=False` with
`[A_{−nfb} … A_{nfb}]` ⇒ Toeplitz without conjugacy; and `nfp < nfb` must
raise `ValueError`. `test_io.py` uses a `_shared_tmpdir(comm)` helper that
`mkdtemp`s on rank 0 and broadcasts — PETSc binary I/O is collective.

**Time stepping** compares `res4py.solve_ivp` (RK3, `nsteps=10000`) against
scipy at `rtol=atol=1e-13` with the same forcing assembled in numpy, tol
`1e-8`, across `complex × adjoint`. The post-transient group runs
`donothing` (`tol=1e-10`, ≤500 periods) against `gmres` (`gmres_rtol=1e-12`)
on the same operator and asserts agreement `< 1e-6`, for LTI and LTP, forward
and adjoint.

## SSM

`tests/spectral_submanifold/` holds only an empty `__init__.py`. SSM is
exercised through `examples/ssm/` (autonomous) and `examples/ssm_periodic/`
(time-periodic) instead — see
[resolvent4py-examples](../resolvent4py-examples/SKILL.md).

A first unit test would target the multi-index combinatorics, which need no
PETSc: `compute_multiindices(m=2)` for `r=2` should give
`[(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)]`, and
`_generate_quadratic_pairs((2,0), m=2)` should enumerate `[((1,0), (1,0))]`.
For the full pipeline, follow the `test_balanced_truncation.py` template.

## Common pitfalls

- Tests assume **complex** PETSc — `numpy_to_petsc` helpers cast through
  `complex128`. A real-scalar PETSc build breaks most of the suite, and
  reference comparisons in `test_io.py` must match that dtype.
- `LowRankLinearOperator.S` must be broadcast from rank 0
  (`S = comm.tompi4py().bcast(S, root=0)`). Otherwise each rank holds a
  different `Σ` and the action is undefined.
- `Propagator` tests use `square_stable_random_matrix` — for unstable `A` the
  propagated solution blows up and the comparison is meaningless.
- `Product` tests reverse-build the operator: `linops=[L1, L2, L3, L4]` with
  `actions=[L1.solve, L2.apply, L3.apply_hermitian_transpose, L4.apply]`
  yields `L = L1⁻¹ L2 L3* L4` (left-to-right), matching the docstring in
  `product.py`.
- `enforce_complex_conjugacy` requires **odd** `nblocks` — the frequent bug is
  computing `nblocks = 2 * nfp` instead of `2 * nfp + 1`.
- `test_gmres_bjacobi_solver_custom_tolerances` allows
  `residuals[1] <= residuals[0] + 1e-15`; tightening it causes intermittent
  failures.
- The piecewise quadrature boundaries in `test_balanced_truncation.py` are
  sensitive to off-by-one errors where `domega` changes — re-check that
  `omegas` and `wlgs` line up before rewriting that loop.
- `r = np.min([r, krylov_dim - 1])` — `eig()` can never return more modes than
  `krylov_dim`; tests clamp for safety with small fixtures.
