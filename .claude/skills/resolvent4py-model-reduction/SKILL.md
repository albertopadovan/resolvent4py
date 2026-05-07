---
name: resolvent4py-model-reduction
description: Frequency-domain balanced truncation pipeline (quadrature Gramian factors, balancing projection via SVD of Y*X, reduced-order operator assembly). Use when the user asks about model reduction, balanced truncation, Gramians, or building reduced-order models from resolvent snapshots.
---

# `resolvent4py.model_reduction`

Frequency-domain balanced truncation as in
*Dergham et al., Phys. Fluids 2011*. The reachability and
observability Gramians are approximated by Gauss–Legendre quadrature
of the resolvent on the imaginary axis; an SVD of `Y*X` yields the
balancing projection.

## File

[balanced_truncation.py](../../../src/resolvent4py/model_reduction/balanced_truncation.py)
— exports `compute_gramian_factors`, `compute_balanced_projection`,
and `assemble_reduced_order_tensors`. `__init__.py` re-exports these
as `res4py.model_reduction.<name>`.

## The three-step pipeline

### 1. `compute_gramian_factors(L_generators, frequencies, weights, B, C)`

For each quadrature point `ω_j`, you supply a *generator*
`L_generators[j](ω_j)` that returns `(L, action, destroyers)` —
typically `L = MatrixLinearOperator(iωI − A, ksp)` with
`action = L.solve_mat`. The routine evaluates `R(ω_j)B` and
`R(ω_j)*C` on each quadrature point, scales by `√w_j`, and packs
the snapshots as the columns of `X` and `Y`.

When the underlying operator is real-valued (`L.get_real_flag() == True`)
and your quadrature points are non-negative, the routine **splits**
each snapshot into its real and imaginary parts and stores them as
separate columns. This halves the quadrature cost (positive
frequencies suffice). The `δ_j = √(1/2)` factor at `ω_j = 0`
prevents double-counting the zero frequency.

The `destroyers` tuple lets each generator clean up its own KSP/Mat
once that frequency is done — important because each generator
typically allocates a new MUMPS factorization.

### 2. `compute_balanced_projection(X, Y, r)`

Computes `Z = Y* X` and its SVD `U Σ V*`, then forms the balancing
modes:

```
Φ = X V Σ^(−1/2),  Ψ = Y U Σ^(−1/2)
```

The SVD is solved with SLEPc's `LAPACK` driver (i.e. fully dense),
so `Z` should be of size `(rB · nf, rC · nf)` — manageable. Each
column is rotated by `exp(−iθ)` where `θ` is chosen to minimize the
imaginary part — for real systems this keeps `Φ`, `Ψ` real even
though intermediate complex arithmetic is used.

### 3. `assemble_reduced_order_tensors(L, B, C, Phi, Psi)`

Returns `(A_r, B_r, C_r)` as numpy arrays of shape
`(r,r) / (r, nB) / (nC, r)`:

```
A_r = Ψ* L Φ,  B_r = Ψ* B,  C_r = Φ* C  (note the dot orientation)
```

These are formed via `BV.dot` / `LinOp.apply_mat`, so they're
distributed-aware but the result is gathered into numpy.

## Common pitfalls

- `compute_gramian_factors` accepts `L_generators` as a *list* of
  callables, one per frequency. If you want the same generator for
  all frequencies, build it with `functools.partial` and replicate:
  `[partial(L_gen, A=Apetsc) for _ in range(len(omegas))]` — see
  [test_balanced_truncation.py](../../../tests/model_reduction/test_balanced_truncation.py)
  for the canonical pattern.
- The `destroyers` tuple pattern is mandatory for memory hygiene at
  scale — without it the per-frequency MUMPS factorization leaks.
- `r` in `compute_balanced_projection` is silently clipped to
  `svd.getConverged()` — check the returned `S_` shape before
  trusting downstream sizes.

## Tests

See [resolvent4py-tests-model-reduction](../resolvent4py-tests-model-reduction/SKILL.md)
— it builds quadrature with a refined `domega` near zero, then
compares against `scipy.linalg.solve_continuous_lyapunov` for the
exact Hankel singular value and reduced operator.
