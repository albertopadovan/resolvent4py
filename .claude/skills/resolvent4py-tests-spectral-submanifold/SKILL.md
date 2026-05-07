---
name: resolvent4py-tests-spectral-submanifold
description: Empty test directory — SSM functionality is currently exercised through examples/ssm and examples/ssm_periodic rather than pytest. Use when the user asks why there are no SSM tests, or for guidance on how to add the first one.
---

# `tests/spectral_submanifold/`

This directory exists for symmetry with the source tree but currently
holds only an empty `__init__.py`. There are no automated tests yet
for [resolvent4py-spectral-submanifold](../resolvent4py-spectral-submanifold/SKILL.md).

## Where SSM is actually validated

The SSM solver (`SpectralSubmanifold` and the `DifferentialEquation`
ABC) is currently exercised through worked examples rather than unit
tests:

- [examples/ssm/](../../../examples/ssm/) — autonomous SSMs (toy model
  + KSE).
- [examples/ssm_periodic/](../../../examples/ssm_periodic/) — SSMs of
  time-periodic systems.

These examples set up a concrete `DifferentialEquation` subclass
(typically wrapping a discretized PDE), call `SpectralSubmanifold(...)
.solve(Phi, Psi, Lams)`, and then either compare `decode(s(t))`
against full-state simulation or check that `latent_space_dynamics`
agrees with reduced-order time stepping.

## Adding tests here

A first pytest unit test would check the multi-index combinatorics
in isolation:

- `compute_multiindices(m=2)` for `r=2` should return
  `[(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)]`.
- `_generate_quadratic_pairs((2,0), m=2)` enumerates
  `[((1,0), (1,0))]` (and similarly for higher orders).

These don't need PETSc and would catch regressions in the
combinatorial helpers before they corrupt downstream coefficient
solves.

For the full pipeline, follow the
[test_balanced_truncation.py](../../../tests/model_reduction/test_balanced_truncation.py)
template: build a small linear quadratic system, run
`solve(Phi, Psi, Lams)`, and compare `decode(s)` against
`p_0 + Σ_j p_j · prod(s ** j)` computed by hand.
