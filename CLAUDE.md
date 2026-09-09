# resolvent4py

Parallel Python toolbox for analysis, model reduction and control of
high-dimensional linear (and weakly-nonlinear, quadratic) dynamical systems.
Built on `mpi4py` + `petsc4py` + `slepc4py`. Currently on v2.0.0.

## Environment — read this first

**Always activate the conda environment before any Python command:**

```bash
conda activate resolvent4py_env
```

This applies to `python`, `pytest`, `mpiexec`, and `sphinx-build` alike. The
base environment does not have a complex-scalar PETSc build and will fail in
confusing ways.

PETSc **must** be built with complex scalars (`PETSc.ScalarType == np.complex128`)
and MUMPS. `PETSC_DIR` / `PETSC_ARCH` / `SLEPC_DIR` should be unset so
petsc4py binds to the conda build — see the install section of the README.

## Running the tests

The suite is MPI-parallel. **Never run bare `pytest`** — many tests assert on
parallel layout and will fail or silently under-test on one rank:

```bash
mpiexec -n 2 pytest tests/
```

CI runs `mpirun -n 2 python3 -m pytest -v -m "local" tests/` in a
`dolfinx/dolfinx:stable` container. The `local` marker is only on a couple of
tests; the default (unmarked) run is the full suite.

Shared fixtures live in `tests/conftest.py` (`comm`, `square_random_matrix`,
`square_stable_random_matrix`, ...). Fixtures own the PETSc objects they
create and destroy them — don't destroy a fixture-provided matrix in a test.

## Architecture

`LinearOperator` (`src/resolvent4py/linear_operators/linear_operator.py`) is
the organizing abstraction. Every analysis routine in `linalg/` and
`model_reduction/` is written against that interface, so it works on any
operator without knowing how the operator is stored. When adding a capability,
ask whether it belongs on the interface or in a routine that consumes it.

| Package | Holds |
|---|---|
| `linear_operators/` | The ABC + concrete subclasses (matrix, low-rank, product, projection, propagator, Leray, NS, PETSc shell, ...) |
| `linalg/` | Arnoldi/eig, randomized SVD, RSVD-dt, harmonic resolvent |
| `model_reduction/` | Frequency-domain balanced truncation |
| `utils/` | BV/Vec/Mat helpers, MPI scatter/gather, PETSc I/O, KSP factories (MUMPS, GMRES+bjacobi), time stepping |

Optional methods (`apply_hermitian_transpose`, `solve`, ...) use
`@raise_not_implemented_error` rather than `@abstractmethod`, so a subclass
that can't support one simply doesn't override it.

## Conventions

**Keep the compute path in PETSc.** Anything with a distributed dimension `n`
stays a PETSc `Mat`/`Vec`/`BV` and is operated on in parallel. Never gather a
distributed object to rank 0 to run a serial algorithm on it.

scipy *is* allowed — and used in `src/` — but only on small dense matrices
that are replicated identically on every rank and whose size is a rank or
Krylov dimension, never `n`: the Arnoldi Hessenberg in
`linalg/eigendecomposition.py`, the r×r Woodbury inverse in
`linear_operators/low_rank_updated.py`. In `tests/`, scipy is the reference
oracle to compare against; that's expected.

**Ownership.** Operators destroy what they allocate; they do not destroy
objects handed to them by the caller. Test leaks usually trace back to
violating this.

**Style.** ruff, line length 79, `target-version = "py39"`. Python 3.9 must
keep working (needed for Stampede3), so no `match`, no `X | Y` type syntax at
runtime, no `dict[str, int]` annotations without `from __future__ import
annotations`.

## Docs

Sphinx sources in `docs/source/`, built via `docs/Makefile` (or
`docs/compile_html.sh`). Published to GitHub Pages from the `gh-pages` branch.

## Deeper references

`.claude/skills/` carries per-area detail (linear operators, linalg, model
reduction, SSM, utils, examples, tests). Those load on demand — reach for them
when a question goes past what's above.
