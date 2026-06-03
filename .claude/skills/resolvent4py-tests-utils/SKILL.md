---
name: resolvent4py-tests-utils
description: Tests for utils — BV ops, MPI scatter/gather, PETSc binary I/O, harmonic-balanced (de)serialization, KSP factories, matrix factories + COO/CSR + block extraction, random data, time stepping, vector ops + complex-conjugate enforcement. Use when the user asks about validating utility helpers or about edge cases like the bjacobi alignment test.
---

# `tests/utils/`

One test file per source file in
[resolvent4py-utils](../resolvent4py-utils/SKILL.md). This is the
largest test module by line count — utilities are the thinnest layer
above PETSc/SLEPc and need the broadest coverage.

## Test files

| File | Source | What's covered |
|---|---|---|
| [test_bv.py](../../../tests/utils/test_bv.py) | `bv.py` | `bv_add/conj/real/imag/slice/roll`; harmonic-balanced reshape (vec ↔ BV) round-trips |
| [test_comms.py](../../../tests/utils/test_comms.py) | `comms.py` | `compute_local_size` summation, dist⇄seq round-trips for vec/mat, scatter from root with/without explicit `locsize` |
| [test_io.py](../../../tests/utils/test_io.py) | `io.py` | Read/write round-trips for vec/mat/BV; harmonic-balanced matrix in 3 modes (block-diagonal, real-bflow Toeplitz, two-sided Toeplitz); error on too-few blocks |
| [test_ksp.py](../../../tests/utils/test_ksp.py) | `ksp.py` | MUMPS solve accuracy + `check_lu_factorization`; GMRES+bjacobi block-diagonal one-iteration property; tighter rtol → smaller residual |
| [test_matrix.py](../../../tests/utils/test_matrix.py) | `matrix.py` | Mat factories, hermitian-transpose, `mat_solve_hermitian_transpose`, harmonic resolvent generator, `extract_block_diagonal` |
| [test_miscellaneous.py](../../../tests/utils/test_miscellaneous.py) | `miscellaneous.py` | `get_mpi_type` for every supported numpy dtype; raises on unsupported |
| [test_random.py](../../../tests/utils/test_random.py) | `random.py` | Sizes, real/complex flag honored, square + rectangular |
| [test_time_stepping.py](../../../tests/utils/test_time_stepping.py) | `time_stepping.py` | RK3 forced/adjoint vs `scipy.integrate.solve_ivp` reference (both real and complex `A`); plus `compute_post_transient_solution(method='gmres')` ↔ `method='donothing'` cross-check on stable LTI and LTP, forward + adjoint |
| [test_vector.py](../../../tests/utils/test_vector.py) | `vector.py` | `vec_real/imag` (in-place + copy), `enforce_complex_conjugacy` round-trip, even-`nblocks` raises, harmonic-balanced vec→BV reshape |

## Notable test patterns

### Shared tmp-dir for I/O

[test_io.py](../../../tests/utils/test_io.py) uses a
`_shared_tmpdir(comm)` helper that creates `tempfile.mkdtemp()` on
rank 0 and broadcasts the path — required because PETSc binary I/O
is collective.

### Block-Jacobi alignment ([test_ksp.py](../../../tests/utils/test_ksp.py))

`test_gmres_bjacobi_block_diagonal_one_iter` builds a genuinely
block-diagonal matrix and asserts GMRES converges in *exactly* one
iteration when bjacobi(nblocks) is exact. To guarantee the bjacobi
sub-blocks line up with rank ownership, the test picks
`nblocks = 5` for serial runs and `nblocks = comm.size` otherwise
(one block per rank). If you change this rule, GMRES will silently
take more than one iteration.

### Harmonic-balanced I/O ([test_io.py](../../../tests/utils/test_io.py))

Three flavors:

1. `len(filenames_lst) == 1, real_bflow=True` ⇒ block-diagonal
   matrix (only `A_0` provided, all blocks = `A_0`).
2. `real_bflow=True` with `[A_0, A_1, ..., A_{nfb}]` ⇒ Toeplitz
   matrix with `A_{−k} = conj(A_k)` filled in automatically.
3. `real_bflow=False` with `[A_{−nfb}, ..., A_0, ..., A_{nfb}]` ⇒
   Toeplitz without conjugacy assumption.

Plus a 4th: `nfp < nfb` (more Fourier modes than the matrix can
hold) must raise `ValueError`.

### Round-trip tests

The harmonic-balanced reshape between stacked vectors and
`n × nblocks` BVs is checked in both directions and as a round-trip
(`vec → BV → vec` and `BV → vec → BV`). Tolerance `1e-12` because
the conversion goes through PETSc COO assembly with cross-rank
redistribution.

### Time stepping reference ([test_time_stepping.py](../../../tests/utils/test_time_stepping.py))

Two test groups in this file:

1. `test_time_stepping_forced` — builds a stable random `A` and a
   periodic forcing `f(t) = Σ F̂_k e^{ikωt}`, then compares
   `res4py.solve_ivp(...)` (RK3, `nsteps=10000`) against
   `scipy.integrate.solve_ivp` with `rtol=atol=1e-13` and the same
   forcing assembled in numpy. Tolerance: `1e-8`. Tested in 4
   combinations: `complex × adjoint`.

2. `test_post_transient_gmres_vs_donothing_LTI` /
   `_LTP` — runs `compute_post_transient_solution` with both
   `method='donothing'` (`tol=1e-10`, up to 500 periods) and
   `method='gmres'` (`gmres_rtol=1e-12`) on the same stable operator
   and forcing, gathers the two `Yhat` BVs, asserts relative
   agreement `< 1e-6`. LTI uses `pytest_utils.generate_stable_random_matrix`;
   LTP uses a local `_make_stable_periodic_op` (eigenvalue-shifted
   DC + `ε=0.1` AC). Both forward and adjoint are tested.

## Common pitfalls

- `test_vector.py::test_enforce_complex_conjugacy_even_blocks_raises`
  documents that `nblocks` must be **odd** — a frequent mistake
  when computing `nblocks = 2 * nfp` instead of `2 * nfp + 1`.
- `test_io.py` uses COO arrays with values cast to `complex128`
  even for real data — `read_vector` always returns complex
  scalars when PETSc is built with complex scalars, so reference
  comparisons must match dtype.
- `test_ksp.py::test_gmres_bjacobi_solver_custom_tolerances` —
  the assertion `residuals[1] <= residuals[0] + 1e-15` allows a
  small floating-point slack; do not tighten or it will fail
  intermittently.
