---
name: resolvent4py-linalg
description: Iterative linear-algebra routines (Arnoldi/eig, randomized SVD, time-stepping resolvent SVD) for resolvent4py. Use when the user asks about computing eigenvalues, SVDs, or resolvent modes of large-scale operators backed by PETSc/SLEPc.
---

# `resolvent4py.linalg`

Matrix-free, iterative linear-algebra kernels that consume any
[LinearOperator](../resolvent4py-linear-operators/SKILL.md) and
return SLEPc.BV / numpy outputs. Everything in this module assumes
the operator is large enough that you cannot form it as a dense
matrix.

## Files at a glance

| File | Public API | What it does |
|---|---|---|
| [eigendecomposition.py](../../../src/resolvent4py/linalg/eigendecomposition.py) | `arnoldi_iteration`, `eig`, `match_right_and_left_eigenvectors`, `check_eig_convergence` | Krylov eigensolver built on CGS2. |
| [randomized_svd.py](../../../src/resolvent4py/linalg/randomized_svd.py) | `randomized_svd`, `check_randomized_svd_convergence` | Halko-style randomized SVD with power iterations (Ribeiro 2020 variant). |
| [resolvent_analysis_time_stepping.py](../../../src/resolvent4py/linalg/resolvent_analysis_time_stepping.py) | `resolvent_analysis_rsvd_dt` | RSVD-dt: resolvent SVD via time-stepping for systems too large to factorize. |

`__init__.py` does `from .X import *`, so everything above is exposed
as `res4py.linalg.<name>`.

## Mental model

All three top-level routines have the same shape: you pass a
`LinearOperator L` plus a callable `action` (e.g. `L.apply`,
`L.solve`, `L.apply_hermitian_transpose_mat`, …) and the routine
drives the iteration through `action`. This is what makes the module
operator-agnostic — the same `eig` works on
`MatrixLinearOperator`, `LowRankUpdatedLinearOperator`,
`ProductLinearOperator`, etc.

### Arnoldi / `eig`

[eigendecomposition.py:24](../../../src/resolvent4py/linalg/eigendecomposition.py#L24)
implements classical Gram–Schmidt with re-orthogonalization (CGS2)
— same stability as MGS but only 2 `Allreduce`s per step instead of
`k`. To get eigenvalues near a target, pass `L.solve` and a
`process_evals` that maps shift-inverted eigenvalues back, e.g.

```python
D, V = res4py.linalg.eig(L, L.solve, krylov_dim, n_evals,
                         process_evals=lambda x: 1./x)
```

`match_right_and_left_eigenvectors` biorthogonalizes a separate
right/left run so that `W^* V = I` and `W^* L V = diag(λ)`.
`check_eig_convergence` returns the per-pair residual
`||L v - λ v||`.

### Randomized SVD

[randomized_svd.py:19](../../../src/resolvent4py/linalg/randomized_svd.py#L19)
requires `action ∈ {L.apply_mat, L.solve_mat}` and uses the
corresponding `*_hermitian_transpose_mat` for the adjoint sweep.
Real-valued operators have their random initial BV projected to
real entries; block-CC operators get `enforce_complex_conjugacy`
applied.

### RSVD-dt

[resolvent_analysis_time_stepping.py:29](../../../src/resolvent4py/linalg/resolvent_analysis_time_stepping.py#L29)
is for the regime where forming `(iωI − A)⁻¹` is impossible — it
drives the resolvent action via `solve_ivp` (in
[utils.time_stepping](../resolvent4py-utils/SKILL.md)). Returns one
`(U, Σ, V)` triple per resolved frequency. Inputs `B`, `C` default
to identity LinOps if `None`.

## Common pitfalls

- `eig` returns evals as a **diagonal numpy matrix**, not a 1D array.
  Call `np.diag(D)` before sorting/comparing.
- `randomized_svd` rejects any `action` that is not `apply_mat` /
  `solve_mat` — the `_mat` variants are required because the
  algorithm operates on BVs, not single vectors.
- For RSVD-dt, `n_periods` and `tol` together control transient
  decay; if convergence stalls increase `n_periods` before lowering
  `tol`.

## Tests

See [resolvent4py-tests](../resolvent4py-tests/SKILL.md)
for the parallel-aware reference checks.
