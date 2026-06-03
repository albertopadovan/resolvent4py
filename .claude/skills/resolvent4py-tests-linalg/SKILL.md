---
name: resolvent4py-tests-linalg
description: Tests for linalg routines — Arnoldi/eig with shift-invert, biorthogonalization, randomized SVD, and the time-stepping resolvent SVD (RSVD-dt). Use when the user asks about validating eigensolver results, expected tolerances, or the RSVD-dt test pattern.
---

# `tests/linalg/`

Validates [resolvent4py-linalg](../resolvent4py-linalg/SKILL.md)
against numpy / scipy reference computations.

## Test files

| File | Source under test | Cases |
|---|---|---|
| [test_eigendecomposition.py](../../../tests/linalg/test_eigendecomposition.py) | `linalg/eigendecomposition.py` | Shift-invert Arnoldi convergence vs scipy.linalg.eig; biorthogonalization (`W*V = I`, `W*AV = diag(λ)`) |
| [test_randomized_svd.py](../../../tests/linalg/test_randomized_svd.py) | `linalg/randomized_svd.py` | Singular values and triplet residuals vs scipy.linalg.svd |
| [test_resolvent_analysis_rsvd_dt.py](../../../tests/linalg/test_resolvent_analysis_rsvd_dt.py) | `linalg/resolvent_analysis_time_stepping.py` | Post-transient response (`compute_post_transient_solution`) and the full RSVD-dt SVD |
| [test_harmonic_resolvent_time_vs_freq.py](../../../tests/linalg/test_harmonic_resolvent_time_vs_freq.py) | `compute_post_transient_solution` ↔ algebraic harmonic resolvent | Cross-checks time-domain post-transient integration against the HB solve `(iΩ − A_HB)⁻¹ F̂` for a real T-periodic `A(t)`. Four tests: forward / adjoint × `B = C = I` / time-periodic `B(t), C(t)`. Tolerances `1e-4` on the well-resolved range (`n_compare = 10` interior modes when `n_omegas = n_pert = 24`). |

## Patterns specific to this directory

### Building a resolvent operator for shift-invert tests

The `_build_resolvent_operator(comm, A, omega)` helper builds
`L = (iωI − A)` with a MUMPS solver attached so that `L.solve` acts
as `R(iω) = (iωI − A)⁻¹`. The eigenvalue tests call
`res4py.linalg.eig(L, L.solve, krylov_dim, r,
process_evals=lambda x: 1j*omega - 1/x)` to recover eigenvalues of
`A` from the shift-inverted spectrum.

### Tolerance bands

- `eig` accuracy: `5e-1` *percent* on the dominant eigenvalues
  (Arnoldi is iterative; tighter than this needs more `krylov_dim`).
- `match_right_and_left_eigenvectors`: `1e-12` for `W*V − I`,
  `1e-6` for `W*AV − diag(λ)` (looser because two solver calls).
- `randomized_svd` singular values: `5e-1` percent.
- `randomized_svd` triplet residual: `1e-10`.
- RSVD-dt: `1e-2` relative error on the dominant singular value
  (time integration introduces RK truncation error).

### RSVD-dt parametrization

[test_resolvent_analysis_rsvd_dt.py](../../../tests/linalg/test_resolvent_analysis_rsvd_dt.py)
randomizes `omega`, `n_omegas`, `dt` per run and broadcasts from
rank 0 so all ranks agree. It tests both `real=True` (positive
freqs only) and `real=False` (full two-sided spectrum) and both
`L.apply` (forward) and `L.apply_hermitian_transpose` (adjoint)
post-transient responses.

## Common pitfalls

- `r = np.min([r, krylov_dim - 1])` — eig() can never return more
  modes than `krylov_dim`. Tests clamp to be safe with small
  fixture sizes.
- The convergence-radius tolerances above assume the small
  `(50, 50)` fixture matrix. If the fixture grows, tighten them.
