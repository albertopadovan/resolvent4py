---
name: resolvent4py-tests-model-reduction
description: Single test for the balanced-truncation pipeline — compares quadrature-based Gramians + balanced projection + reduced-order assembly against scipy's continuous Lyapunov solver. Use when the user asks how balanced truncation accuracy is validated.
---

# `tests/model_reduction/`

Validates
[resolvent4py-model-reduction](../resolvent4py-model-reduction/SKILL.md).
Currently a single test:

## File

[test_balanced_truncation.py](../../../tests/model_reduction/test_balanced_truncation.py)

## What it does

1. **Build the system**: `(A, B, C)` with random `A` of size
   `5×5` and tall-skinny inputs/outputs.
2. **Quadrature**: piecewise Gauss–Legendre on `ω ∈ [0, ...]`. The
   first `30 · domega` band (where `domega = 0.05`) gets
   10-point quadrature per sub-interval; the next 50 sub-intervals
   (`domega *= 10`) get 5-point quadrature each. This refines near
   `ω = 0` where the resolvent peaks.
3. **`L_generator`**: closes over `A` and produces, for each `ω`,
   `(L, L.solve_mat, (L.destroy,))` where
   `L = (iωI − A)` factored by MUMPS.
4. **Pipeline**: `compute_gramian_factors` → `compute_balanced_projection(r=1)`
   → `assemble_reduced_order_tensors`.
5. **Reference**: scipy's `solve_continuous_lyapunov` for both
   `X` (reachability) and `Y` (observability), then
   `eig(X·Y)` for the Hankel singular values.
6. **Assertions**:
   - Dominant Hankel singular value error: `< 5%`.
   - Reduced operator `A_r` error: `< 5%`.

The 5% tolerance is loose because Gauss–Legendre on a finite
truncated interval cannot exactly reproduce the
`∫_{-∞}^∞ R(ω)BB*R(ω)*` integral.

## Patterns specific to this directory

### `functools.partial` for L_generators

The test uses `L_gen = partial(L_generator, A=Apetsc)` and replicates
it once per quadrature point: `[L_gen for _ in range(len(omegas))]`.
This is the canonical pattern for passing per-frequency generators
when every generator shares the same operator.

### Real vs complex handling

The test currently fixes `complex = False` at the top. The dead
code branch (`if complex:`) that mirrors quadrature points to
negative frequencies is documented but not exercised here — for
the complex case, see RSVD-based examples.

## Common pitfalls

- The piecewise quadrature **boundaries** matter: stitching from
  `intervals[idx]` (where `domega` changes) is sensitive to
  off-by-one errors in the slice. Don't rewrite the construction
  loop without re-checking that `omegas` and `wlgs` line up.
- Tolerances are *relative percent*, not absolute. The `100 *`
  factor in front of `np.linalg.norm(...)` is intentional.

## Future tests (`spectral_submanifold/`)

The sibling `tests/spectral_submanifold/` directory contains only an
empty `__init__.py`. SSM functionality is currently exercised
through the `examples/ssm/` and `examples/ssm_periodic/`
directories rather than pytest.
