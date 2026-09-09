---
name: resolvent4py-linear-operators
description: Abstract LinearOperator base class plus concrete subclasses (matrix, low-rank, low-rank-updated, product, projection, shift-and-scale, propagator, time-periodic-matrix, PETSc-shell). Use when the user asks how to build, compose, or extend resolvent4py linear operators, or when they need to choose the right subclass for a given problem.
---

# `resolvent4py.linear_operators`

This is the *core abstraction* of resolvent4py. Almost every routine
elsewhere in the library — [linalg](../resolvent4py-linalg/SKILL.md),
[model_reduction](../resolvent4py-model-reduction/SKILL.md),
[spectral_submanifold](../resolvent4py-spectral-submanifold/SKILL.md)
— is written against the `LinearOperator` interface so it can drive
any concrete operator without knowing how the operator is stored.

## The base class

[linear_operator.py](../../../src/resolvent4py/linear_operators/linear_operator.py)
defines the abstract `LinearOperator`. Subclasses must implement
`apply`, `apply_mat`, and `destroy`. The optional methods
(`apply_hermitian_transpose`, `apply_hermitian_transpose_mat`,
`solve`, `solve_mat`, `solve_hermitian_transpose`,
`solve_hermitian_transpose_mat`) are decorated with
`raise_not_implemented_error` — calling them on a subclass that
didn't override them raises a clean `NotImplementedError` with the
class name attached.

Key responsibilities of the base class:

- Store dims as `((local_rows, global_rows), (local_cols,
  global_cols))` and `nblocks` for block-structured (harmonic
  balanced) operators.
- Auto-detect `real_flag` (operator preserves real subspace) and
  `block_cc_flag` (operator preserves complex-conjugate block
  structure) at construction time. Most subclasses inherit this
  behavior; **the time-related ones override `check_if_real_valued`
  / `check_if_complex_conjugate_structure` to delegate to the inner
  operator** rather than paying the random-vector probe cost
  (`PropagatorLinearOperator`, `ShiftAndScaleLinearOperator`).
- Provide BV/Vec factories: `create_left_vector`,
  `create_right_vector`, `create_left_bv(n)`, `create_right_bv(n)`.
- Provide a default **`set_evaluation_time(t)`** that walks every
  attribute and, for any attribute that is itself a `LinearOperator`
  (or a `list`/`tuple` thereof), recursively forwards the time call.
  Composite operators (`ShiftAndScale`, `Product`,
  `LowRankUpdated`, `Projection`, `Propagator`) get time-propagation
  *for free* on top of a time-periodic core
  (`TimePeriodicMatrixLinearOperator`). Only the leaf class that
  actually depends on time
  (`TimePeriodicMatrixLinearOperator.set_evaluation_time` overrides
  `self.time = time`); everyone else inherits the walking default.

## Concrete subclasses

| File | Class | Operator |
|---|---|---|
| [matrix.py](../../../src/resolvent4py/linear_operators/matrix.py) | `MatrixLinearOperator` | `L = A` (any PETSc.Mat). Optional KSP enables `solve*`. |
| [low_rank.py](../../../src/resolvent4py/linear_operators/low_rank.py) | `LowRankLinearOperator` | `L = U Σ V*` (Σ need not be diagonal). |
| [low_rank_updated.py](../../../src/resolvent4py/linear_operators/low_rank_updated.py) | `LowRankUpdatedLinearOperator` | `L = A + B K C*`; `solve*` via Woodbury. |
| [propagator.py](../../../src/resolvent4py/linear_operators/propagator.py) | `PropagatorLinearOperator` | `L = Φ(t_f, t_0)` — solution-operator of `dx/dt = A(t) x` over `[t_0, t_f]`, integrated by RK2/RK3. Collapses to `exp(A(t_f − t_0))` when `A` is time-invariant; gives the monodromy when `t_f − t_0 = T` and `A` is `T`-periodic. Inherits real/cc flags from `A`. |
| [time_periodic_matrix.py](../../../src/resolvent4py/linear_operators/time_periodic_matrix.py) | `TimePeriodicMatrixLinearOperator` | `L(t) = Σ_k A_k exp(i ω_k t)` — time-periodic operator with explicit Fourier coefficients. One-sided `freqs` (min == 0) signals a real `A(t)` and triggers a complex-conjugate fast-path (`A_{−k} = conj(A_k)`). `set_evaluation_time(t)` sets `self.time`; subsequent `apply`/`apply_mat`/HT all use it. |
| [petsc_python.py](../../../src/resolvent4py/linear_operators/petsc_python.py) | `PetscPythonLinearOperator` | Wraps a LinOp as a PETSc shell `Mat` so KSPs can use it. **Not** a `LinearOperator` subclass. |
| [product.py](../../../src/resolvent4py/linear_operators/product.py) | `ProductLinearOperator` | `L = L_r ... L_2 L_1` for any mix of `apply`/`solve`/HT actions. |
| [projection.py](../../../src/resolvent4py/linear_operators/projection.py) | `ProjectionLinearOperator` | `L = Φ(Ψ*Φ)⁻¹Ψ*` or `I − P` (idempotent). |
| [shift_and_scale.py](../../../src/resolvent4py/linear_operators/shift_and_scale.py) | `ShiftAndScaleLinearOperator` | `L = αI + βA`. Cheap; useful for resolvent shifts. **Inherits `block_cc_flag` from `A`** and **`real_flag` from `A` AND** `np.imag(α) == np.imag(β) == 0` — no random-vector probe at construction. |

## Composition patterns

These operators compose freely. Common idioms:

- **Resolvent**: build `R⁻¹ = ShiftAndScale(A, alpha=iω, beta=-1)`, give
  it a KSP via `MatrixLinearOperator(R_inv_mat, ksp)`, and you have a
  `solve` that acts as `R(iω)`.
- **Updated dynamics**: `LowRankUpdatedLinearOperator(A, B, K, C)`
  where `A.solve` is enabled gives you `(A + BKC*)⁻¹` for free via
  Woodbury — used for control-design pipelines.
- **Sequential application**: `ProductLinearOperator([L1, L2, L3],
  [L1.solve, L2.apply, L3.apply_hermitian_transpose])` builds
  `L3* L2 L1⁻¹` with caching of intermediate vectors.
- **Matrix-free Krylov**: `PetscPythonLinearOperator.create_shell(L)`
  lets PETSc's KSP/PC infrastructure consume your custom LinOp.
- **Shifted periodic ODE**: `shifted = ShiftAndScale(A_t, α=−s, β=1)`
  where `A_t` is a `TimePeriodicMatrixLinearOperator` gives you
  `A(t) − sI`. Feed to `solve_ivp` / `PropagatorLinearOperator` /
  `compute_post_transient_solution` — the recursive
  `set_evaluation_time` walks through `shifted.A → A_t` automatically.
- **Monodromy as a shell**: `Φ = PropagatorLinearOperator(shifted, 0,
  T, dt)`; then `I − Φ = ShiftAndScale(Φ, α=1, β=−1)`;
  `PetscPythonLinearOperator.create_shell(I − Φ)` is what
  `compute_post_transient_solution(method='gmres')` does internally
  to solve `(s I − L_HB) x = b` via shoot-and-solve.

## Conventions

- All `*_mat` variants take/return `SLEPc.BV`; all non-`_mat`
  variants take/return `PETSc.Vec`.
- `y` (or `Y`) is always optional. If `None`, a fresh output is
  allocated. Pass one in for hot loops to avoid allocations.
- Hermitian transpose is true `L*`, not transpose. `solve_hermitian_transpose`
  on `MatrixLinearOperator` does the conjugate dance around
  `ksp.solveTranspose` automatically.
- Operators that allocate intermediate BVs (e.g. `LowRankUpdated`,
  `Product`) cache them keyed by column count and resize lazily —
  see `_get_intermediate_bv` in
  [low_rank_updated.py:123](../../../src/resolvent4py/linear_operators/low_rank_updated.py#L123).

## Common pitfalls

- `ProductLinearOperator` reverses `linops` and `linops_actions` in
  `__init__` — that's intentional but surprising. The first element
  of the input list is the *outermost* operator, written
  left-to-right as in the math.
- `PropagatorLinearOperator` does not implement `solve`; if you need
  it, wrap with `ShiftAndScale` or invert externally. (Same goes for
  `PetscPythonLinearOperator`.)
- `PetscPythonLinearOperator` is the one class here that does **not**
  inherit from `LinearOperator` — it's a duck-typed shim for PETSc.
- `TimePeriodicMatrixLinearOperator`'s `freqs` argument determines
  the path: one-sided (`min(freqs) == 0`) means "real `A(t)`,
  conjugate-symmetric coefficients implicit". Two-sided means
  general complex. Mixing the two will give silently wrong outputs
  on the complex-conjugate fast-path.
- `set_evaluation_time(t)` must be called on the outermost composite
  before each `apply` if you want the time to advance — `solve_ivp`
  and `compute_post_transient_solution` do this for you at every RK
  stage. If you call `apply` directly on a composite that wraps a
  `TimePeriodicMatrixLinearOperator`, you'll get whatever time it
  was last set to (default `0.0` at construction).

## Tests

See
[resolvent4py-tests](../resolvent4py-tests/SKILL.md).
Every concrete class has a per-file test that compares all four
actions (apply / solve / their HT variants, both vec and BV forms)
against a numpy reference assembled from the same data.
