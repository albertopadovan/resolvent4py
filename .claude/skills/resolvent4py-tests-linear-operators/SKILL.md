---
name: resolvent4py-tests-linear-operators
description: One test file per concrete LinearOperator class — vec/BV actions, hermitian-transpose variants, solve variants, real_flag detection, idempotency where applicable. Use when the user asks how a particular operator class is validated or what the expected tolerance for its actions is.
---

# `tests/linear_operators/`

Tests for every concrete subclass of
[LinearOperator](../resolvent4py-linear-operators/SKILL.md). The
pattern is identical across files: build the operator, build the
equivalent dense numpy matrix, compare each `apply* / solve*` action
against `numpy.dot` / `scipy.linalg.inv`.

## Test files

| File | Class under test | Notes |
|---|---|---|
| [test_matrix.py](../../../tests/linear_operators/test_matrix.py) | `MatrixLinearOperator` | square + rectangular; `Y=None`; repeated apply consistency |
| [test_low_rank.py](../../../tests/linear_operators/test_low_rank.py) | `LowRankLinearOperator` | `Σ` is **not** square (shapes `(rr, rc)`) — tests the general factorization |
| [test_low_rank_updated.py](../../../tests/linear_operators/test_low_rank_updated.py) | `LowRankUpdatedLinearOperator` | Woodbury solve; cached intermediate BV across repeated calls and varying column counts |
| [test_propagator.py](../../../tests/linear_operators/test_propagator.py) | `PropagatorLinearOperator` | RK2 vs RK3 accuracy bands on `square_stable_random_matrix` for the LTI case (= matrix exponential); plus **time-periodic forward/adjoint propagator** vs `scipy.integrate.solve_ivp` RK45 reference. Helper `_make_periodic_A_coeffs` builds a stable time-periodic `A(t)` (eigenvalue-shifted DC block + small ε on AC blocks). |
| [test_time_periodic_matrix.py](../../../tests/linear_operators/test_time_periodic_matrix.py) | `TimePeriodicMatrixLinearOperator` | All 6 combinations of `{vec, BV} × {real input, complex input}` for real-A and complex-A; `set_evaluation_time(t)` correctness; the `⟨v, Lw⟩ = ⟨L*v, w⟩` adjoint identity in all 8 real/complex combinations. Helpers `_make_real_A_coeffs(freqs=[0, 1.2, 2.4])` and `_make_complex_A_coeffs(freqs=[-1.2, 0, 1.2])`. |
| [test_petsc_python.py](../../../tests/linear_operators/test_petsc_python.py) | `PetscPythonLinearOperator` | Shell `mult` / `multHermitian` only; no `solve` |
| [test_product.py](../../../tests/linear_operators/test_product.py) | `ProductLinearOperator` | Mixed actions: `solve` + `apply` + `apply_hermitian_transpose` + `apply` over 4 ops |
| [test_projection.py](../../../tests/linear_operators/test_projection.py) | `ProjectionLinearOperator` | Both `complement=False/True`; idempotency `P²=P` |
| [test_real_flag.py](../../../tests/linear_operators/test_real_flag.py) | The auto-detection in `LinearOperator.__init__` | Real/complex matrices; `ShiftAndScale` with real vs complex `α/β` |
| [test_shift_and_scale.py](../../../tests/linear_operators/test_shift_and_scale.py) | `ShiftAndScaleLinearOperator` | Complex `α=2+3j`, `β=−1.5+0.5j`; vec + BV |

## Common patterns

### The action-list pattern

Most tests build parallel lists `actions_petsc = [linop.apply,
linop.apply_hermitian_transpose, linop.solve,
linop.solve_hermitian_transpose]` and `actions_python = [A.dot,
A.conj().T.dot, Ainv.dot, Ainv.conj().T.dot]`, then iterate. The
norm of the per-action error vector is the assertion target.

### Tolerance bands by operator

| Operator | Typical tol |
|---|---|
| `Matrix`, `LowRank`, `Projection`, `Shift+Scale`, `Product`, `PetscPython` | `1e-10` (or `1e-8` for `LowRank`) |
| `LowRankUpdated` | `1e-8` (Woodbury accumulates error) |
| `Propagator` (RK3, `dt=1e-5`, `tf=2.1`, LTI = matrix exp) | `1e-11` |
| `Propagator` (RK2, same params) | `1e-7` |
| `Propagator` (RK3, time-periodic vs scipy RK45) | `1e-6` |
| `TimePeriodicMatrix` (vec/BV apply vs numpy reference) | `1e-12` |
| `TimePeriodicMatrix` (adjoint identity `⟨v,Lw⟩=⟨L*v,w⟩`) | `1e-12` |

### `LowRankUpdated` — caching tests

[test_low_rank_updated.py](../../../tests/linear_operators/test_low_rank_updated.py)
has two tests that are not just "does it match":

- `test_low_rank_updated_repeated_apply_mat` calls `apply_mat` twice
  with the same `X` and asserts the two results are bitwise
  identical (`< 1e-14`) — proves the cached intermediate BV doesn't
  accumulate state between calls.
- `test_low_rank_updated_varying_column_counts` calls with column
  counts `[3, 7, 3]` to exercise the lazy-resize path
  (`_get_intermediate_bv`).

### Projection — idempotency tests

[test_projection.py](../../../tests/linear_operators/test_projection.py)
checks `P²x = Px` (and `P²X = PX` for BVs) for both
`complement=False` and `complement=True`. This catches subtle errors
in the `LowRankUpdatedLinearOperator` Woodbury path used by the
complement form.

### `real_flag` detection

[test_real_flag.py](../../../tests/linear_operators/test_real_flag.py)
is the dedicated test for the `check_if_real_valued` heuristic in
[linear_operator.py:148](../../../src/resolvent4py/linear_operators/linear_operator.py#L148).
It probes with a random vector and checks `imag(Lx)` norm. Tests
cover real/complex `Matrix`, real/complex `LowRank`, and
`ShiftAndScale` with real vs complex `α`.

## Common pitfalls

- `LowRankLinearOperator.S` (the `Σ` matrix) must be broadcast from
  rank 0 — every test does
  `S = comm.tompi4py().bcast(S, root=0)` after constructing it
  randomly. Otherwise each rank has a different `Σ` and the action
  is undefined.
- `MatrixExponential` tests use `square_stable_random_matrix` — for
  unstable `A`, `e^{At_f}` blows up and the comparison is
  meaningless.
- `Product` tests reverse-build the math operator: passing
  `linops=[L1, L2, L3, L4]`, `actions=[L1.solve, L2.apply,
  L3.apply_hermitian_transpose, L4.apply]` produces
  `L = L1⁻¹ L2 L3* L4` (i.e. left-to-right), matching the docstring
  in
  [product.py](../../../src/resolvent4py/linear_operators/product.py).
