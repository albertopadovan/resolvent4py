---
name: resolvent4py-tests
description: Top-level orientation for the resolvent4py pytest suite — layout, conftest fixtures, pytest_utils helpers, MPI execution caveat. Use when the user asks how the test suite is organized, how to add a new test, or where the reference data comes from.
---

# `tests/`

Pytest test suite for resolvent4py. Mirrors the source tree
one-to-one: `src/resolvent4py/<x>/` is exercised by `tests/<x>/`.
Tests are designed to run under MPI (via `mpiexec -n N pytest ...`)
and silence stdout on non-root ranks so output isn't a mess.

## Layout

```
tests/
├── conftest.py              # Pytest fixtures (comm, sizes, random matrices)
├── pytest_utils.py          # Helpers: random data + numpy/PETSc error metrics
├── __init__.py              # Re-exports pytest_utils for relative imports
├── linalg/                  # → src/resolvent4py/linalg
├── linear_operators/        # → src/resolvent4py/linear_operators
├── model_reduction/         # → src/resolvent4py/model_reduction
├── spectral_submanifold/    # → src/resolvent4py/spectral_submanifold (empty)
└── utils/                   # → src/resolvent4py/utils
```

## Important convention: Stampede3 test execution

**Do not run `pytest` directly here** — `petsc4py` and `mpi4py` are
loaded through environment modules controlled by the user. Tests
must be launched through whatever wrapper script the user provides
(e.g. an `mpiexec`-wrapped `pytest`).

## `conftest.py` fixtures

- `comm` (session-scoped) — `PETSc.COMM_WORLD`.
- `rank_size` (session-scoped) — `(rank, size)` tuple.
- `square_matrix_size` — parametrized fixture, currently fixed at
  `(50, 50)` with `pytest.mark.local`.
- `rectangular_matrix_size` — `(50, 20)` with `pytest.mark.local`.
- `square_random_matrix`, `square_stable_random_matrix`,
  `rectangular_random_matrix` — generate `(Apetsc, Apython)` pairs
  via `pytest_utils`. The "stable" variant shifts eigenvalues to
  guarantee `Re(λ) < 0` (used by matrix-exponential and
  time-stepping tests).
- `test_output_dir` — tmp-path-derived directory for I/O round-trip
  tests.

The conftest also installs a `pytest_configure` hook that
unregisters the terminal reporter on non-root ranks so only rank 0
prints results.

## `pytest_utils.py` helpers

Everything tests need to compare a parallel resolvent4py call
against a sequential numpy/scipy reference:

| Helper | Returns |
|---|---|
| `generate_random_matrix(comm, (Nr, Nc), complex=True)` | `(Apetsc, Apython)` — distributed sparse + sequential dense, identical contents |
| `generate_stable_random_matrix(...)` | Same, with eigenvalues shifted to negative real part |
| `generate_random_bv(comm, (Nr, Nc), complex=True)` | `(X_BV, Xnumpy)` |
| `generate_random_vector(comm, N, complex=True)` | `(x_petsc, xnumpy)` |
| `compute_error_vector(comm, petsc_action, x, y, python_action, xpython)` | Relative `‖y − f(x)‖ / ‖f(x)‖` |
| `compute_error_vector_shell_operator(...)` | Same, for `mult(A, x, y)`-style PETSc shell ops |
| `compute_error_bv(...)` | Relative error between BV action and numpy reference |

These helpers are how nearly every test arrives at its assertion: a
single relative norm compared against a tolerance like `1e-8` to
`1e-14`.

## Test pattern (template)

The dominant pattern across the suite:

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

Variants:

- `*_on_vectors` vs `*_on_bvs` cover the `apply` vs `apply_mat`
  surface.
- `*_y_none` test that omitting the output buffer auto-allocates.
- `*_repeated_apply`, `*_varying_column_counts` exercise cached
  intermediate buffers (for `LowRankUpdated`, `Product`).

## What the subdirectories test

| Tests dir | Source under test | SKILL link |
|---|---|---|
| [linalg/](../../../tests/linalg/) | [linalg](../resolvent4py-linalg/SKILL.md) | [tests-linalg](../resolvent4py-tests-linalg/SKILL.md) |
| [linear_operators/](../../../tests/linear_operators/) | [linear_operators](../resolvent4py-linear-operators/SKILL.md) | [tests-linear-operators](../resolvent4py-tests-linear-operators/SKILL.md) |
| [model_reduction/](../../../tests/model_reduction/) | [model_reduction](../resolvent4py-model-reduction/SKILL.md) | [tests-model-reduction](../resolvent4py-tests-model-reduction/SKILL.md) |
| [spectral_submanifold/](../../../tests/spectral_submanifold/) | [spectral_submanifold](../resolvent4py-spectral-submanifold/SKILL.md) | [tests-spectral-submanifold](../resolvent4py-tests-spectral-submanifold/SKILL.md) |
| [utils/](../../../tests/utils/) | [utils](../resolvent4py-utils/SKILL.md) | [tests-utils](../resolvent4py-tests-utils/SKILL.md) |

## Common pitfalls

- Tests assume `complex=True` PETSc by default — `numpy_to_petsc`
  helpers cast through `complex128`. Building PETSc with real
  scalars will break most tests.
- `square_matrix_size` is parametrized but currently has a single
  value — adding more (e.g. for medium/large markers) requires
  matching `pytest.mark.<level>` declarations in `conftest.py`.
- The `pytest_utils.compute_error_*` helpers gather the result to
  rank 0 via `distributed_to_sequential_*`. The numpy reference is
  computed redundantly on every rank — that's intentional for
  per-rank assertions but means rank 0 does most of the verification work.
