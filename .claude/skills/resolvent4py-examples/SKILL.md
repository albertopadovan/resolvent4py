---
name: resolvent4py-examples
description: Worked examples that demonstrate the full resolvent4py API end-to-end — CGL (eigendecomposition, RSVD, RSVD-dt, balanced truncation), toy_model (harmonic resolvent), autonomous SSMs (Hopf3D + KSE), and time-periodic SSMs (KSE, Rössler period-doubling — including a time-stepping shoot-and-solve path for SSM-style shifts). Use when the user wants a runnable starting point, asks "how do I set up X end-to-end", or needs the canonical recipe for a particular analysis.
---

# `examples/`

The `examples/` tree is the practical complement to the unit tests:
each subdirectory is a self-contained, runnable demonstration of one
analysis. Where the test suite assertions are tight reference
comparisons, the examples are the **starting templates** users
should copy from.

## Layout

```
examples/
├── cgl/                        # Linear Ginzburg–Landau, A is a single matrix
│   ├── cgl.py                  # Spatial discretization (4th-order central FD)
│   ├── generate_matrices.py    # Serial: write COO + B + C to data/
│   ├── demonstrate_eigendecomposition.py
│   ├── demonstrate_rsvd.py
│   ├── demonstrate_rsvd_dt.py
│   ├── demonstrate_balanced_truncation.py
│   ├── run_post_transient_approaches.py   # Comparison playground
│   └── README.rst
│
├── toy_model/                  # 3D Hopf, harmonic-balanced (T(t)=T(t+T))
│   ├── generate_matrices.py    # Serial: build A_k Fourier blocks → data/
│   ├── demonstrate_harmonic_resolvent.py
│   └── README.rst
│
├── ssm/                        # Autonomous spectral submanifolds
│   ├── toy_model/   (3D Hopf normal form — small, fast)
│   └── kse/         (Kuramoto–Sivashinsky, 64 sine modes)
│
└── ssm_periodic/               # SSMs of time-periodic systems
    ├── kse/        (KSE on a Hopf-born limit cycle, harmonic-balanced)
    └── rossler/    (Rössler period-doubling SSM, with both algebraic
                     and time-stepping/GMRES solve_linear_system paths)
```

## How to run them (canonical recipe)

Every CGL / toy_model example follows the same two-phase pattern,
documented in their `README.rst`:

```bash
# Phase 1: assemble the matrices ONCE, in serial
mpiexec -n 1 python -u generate_matrices.py

# Phase 2: run the analysis (any rank count)
mpiexec -n 2 python -u demonstrate_<analysis>.py
```

Outputs land in `data/` (matrices) and `results/` (figures). The
SSM examples are self-contained — `demonstrate_ssm.py` builds its
own `DifferentialEquation` subclass at the top of the script, so
there's no separate generation step.

The same `do not run python directly` rule from
[resolvent4py-tests](../resolvent4py-tests/SKILL.md) applies here:
on Stampede3 these go through whatever `mpiexec` wrapper the user
provides.

## Per-example summary

### `cgl/` — 1D linear PDE, single dense-COO matrix

Spatial operator from
[cgl.py](../../../examples/cgl/cgl.py): 4th-order central finite
differences for `∂_x` and `∂_x²` on `n=2000` nodes; the assembled
`A = -νD + γDD + μI` is real-coefficients but complex-valued
because `ν, γ ∈ ℂ`. `generate_matrices.py` writes `(rows.dat,
cols.dat, vals.dat)` triples plus `B.dat` and `C.dat`.

| Script | What it shows |
|---|---|
| [demonstrate_eigendecomposition.py](../../../examples/cgl/demonstrate_eigendecomposition.py) | Shift-invert Arnoldi at `s=0`; compares to the closed-form CGL eigenvalues from `CGL.compute_exact_eigenvalues`. |
| [demonstrate_rsvd.py](../../../examples/cgl/demonstrate_rsvd.py) | Frequency-domain `randomized_svd` on `R(iω)=(iωI−A)⁻¹` at `ω=0.648`; compared to `scipy.linalg.svd` of the dense resolvent. |
| [demonstrate_rsvd_dt.py](../../../examples/cgl/demonstrate_rsvd_dt.py) | Side-by-side `randomized_svd` vs `resolvent_analysis_rsvd_dt` at the same `ω`; expensive (~minutes), produces `singular_values_compare.png`. |
| [demonstrate_balanced_truncation.py](../../../examples/cgl/demonstrate_balanced_truncation.py) | `compute_gramian_factors` → `compute_balanced_projection(r=10)` over a uniform grid `ω ∈ [-30, 30]` with `Δω=0.324`. |
| [run_post_transient_approaches.py](../../../examples/cgl/run_post_transient_approaches.py) | Mixed showcase: builds `MatrixExponentialLinearOperator`, wraps it as `ShiftAndScale` then `PetscPythonLinearOperator.create_shell` so PETSc's GMRES can solve `(I − e^{AT})x = sol`. Demonstrates the matrix-free composition path. |

### `toy_model/` — harmonic resolvent on a 3D limit cycle

[generate_matrices.py](../../../examples/toy_model/generate_matrices.py)
integrates the nonlinear ODE to a limit cycle, then takes
`rfft` along time of both the orbit `Q(t)` and its Jacobian
`A(t)` to get Fourier coefficients `Â_k, q̂_k, dq̂_k`. These are
written as COO triples and as `Q_kk.dat`, `dQ_kk.dat` files.

[demonstrate_harmonic_resolvent.py](../../../examples/toy_model/demonstrate_harmonic_resolvent.py)
demonstrates four things in one script:

1. `read_harmonic_balanced_matrix` to assemble the block-Toeplitz
   `A` (and the mass matrix) from disk.
2. `assemble_harmonic_resolvent_generator(A, freqs, M)` →
   `T = -i diag(kω) M + A`. A small `1e-7·I` perturbation avoids
   a numerical singularity (the harmonic resolvent has a
   zero-eigenvalue along the phase-shift direction).
3. `ProjectionLinearOperator(Phi, Phi, complement=True)` builds the
   phase-direction-killing projectors `P_d`, `P_r`; then
   `ProductLinearOperator([P_r, T, P_d], [P_r.apply, T.solve,
   P_d.apply])` is `H_1 = P_r T⁻¹ P_d` — the canonical Padovan
   (2020) harmonic resolvent.
4. `eig` on `−PTP` (oblique projector) recovers the **Floquet
   exponents**.

### `ssm/toy_model/` — autonomous SSM on the 3D Hopf normal form

The `Hopf3D` class
([toy_model_differential_equation.py](../../../examples/ssm/toy_model/toy_model_differential_equation.py))
is the minimal `DifferentialEquation` subclass: 3-state system,
hand-coded `evaluate_quadratic_term`, MUMPS-based
`solve_linear_system`. Picks the two leading eigenpairs as master
modes, runs `SpectralSubmanifold(r=2, m=30, scaling=0.2).solve(...)`,
estimates the convergence radius, then integrates both an
on-manifold and an off-manifold initial condition through both the
ROM (`SSM.latent_space_dynamics` + `SSM.decode`) and the truth
ODE for comparison.

This is the **template to copy** when building your own
`DifferentialEquation` for a small ODE.

### `ssm/kse/` — autonomous SSM on Kuramoto–Sivashinsky

Same workflow as `ssm/toy_model/` but the differential equation is
the KSE on `[0, 2π]` with `n=64` sine modes
([kse_differential_equation.py](../../../examples/ssm/kse/kse_differential_equation.py)).
The quadratic term `B(q1, q2)` is computed pseudospectrally (FFT-
based dealiasing on a `4n` grid); when `c_star` is provided, the
system is *linearised* about an equilibrium and `B(c*, ·)` is
folded into the linear operator. Polynomial expansion order is
`m=32`, latent space `r=2`.

This is the template for **PDE-discretization SSMs**: anything
that needs FFT for the nonlinear evaluation and a non-trivial
spectral-Galerkin Jacobian.

### `ssm_periodic/rossler/` — SSM of the period-doubling Rössler orbit

The canonical *small* time-periodic SSM (3-state ODE, fast to iterate).
Two parallel pipelines, mirroring the `ssm/` subdirectories:

1. **Algebraic** (default, identical to `kse/` workflow):
   - [save_eigendecomp.py](../../../examples/ssm_periodic/rossler/save_eigendecomp.py) /
     [save_eigendecomp_2T.py](../../../examples/ssm_periodic/rossler/save_eigendecomp_2T.py) —
     compute the Floquet eigendecomposition (1T and 2T-doubled, respectively).
   - [save_ssm.py](../../../examples/ssm_periodic/rossler/save_ssm.py) /
     [save_ssm_2T.py](../../../examples/ssm_periodic/rossler/save_ssm_2T.py) —
     build the SSM and cache to npz.
   - `solve_linear_system` factors `s I − L_HB` with MUMPS — robust to
     any shift, but pays the full HB LU.

2. **Time-stepping with GMRES shoot-and-solve** (`use_time_stepping=True`
   on `RosslerPeriodic`):
   - Replaces the algebraic MUMPS solve with
     `compute_post_transient_solution(method='gmres', ...)` on
     `A(t) − s I` (built as a `ShiftAndScale` over a
     `TimePeriodicMatrixLinearOperator`).
   - Works for SSM-style shifts where `Re(s)` lands inside the Floquet
     spectrum and the iteration path (`method='donothing'`) would
     diverge. Costs scale with GMRES iterations (typically `O(N_state)`
     because the BVP is on the state-space, not the HB space).
   - Sanity checks:
     [demonstrate_shoot_and_solve.py](../../../examples/ssm_periodic/rossler/demonstrate_shoot_and_solve.py)
     drives the library `method='gmres'` path directly and compares to
     the algebraic baseline on stable + SSM-style shifts.
     [demonstrate_time_stepping_solve.py](../../../examples/ssm_periodic/rossler/demonstrate_time_stepping_solve.py)
     does the same comparison going through
     `eq.solve_linear_system` (the user-facing API).

[rossler_differential_equation.py](../../../examples/ssm_periodic/rossler/rossler_differential_equation.py)
hosts both paths; constructor flag `use_time_stepping=False` (default)
gives the algebraic version, `True` switches to GMRES. Additional
constructor kwargs `ts_dt` (default `T/200`), `gmres_rtol` (default
`1e-8`), `gmres_max_it` (default `200`), `ts_method` (default `"RK3"`),
`ts_verbose`.

**When time-stepping wins for SSM:** never on the 1T basis (master
mode is at `i·ω/2`, shifts are pure imaginary), but in the **2T basis**
(`save_ssm_2T.py`) the master Floquet exponent collapses to a real
value `≈ −3·10⁻³`, all SSM shifts `s_k = k·λ_master` are real
negative, and GMRES converges in ~3 iterations per monomial.

### `ssm_periodic/kse/` — SSM of a time-periodic KSE

The most ambitious example: KSE past its Hopf bifurcation
(`ν=0.058`) settles onto a stable limit cycle. The pipeline is:

1. [compute_periodic_orbit.py](../../../examples/ssm_periodic/kse/compute_periodic_orbit.py)
   — IMEX time-stepping to wash out transients, then Poincaré-section
   period detection refined by Newton shooting, saved to disk.
2. [demonstrate_ssm.py](../../../examples/ssm_periodic/kse/demonstrate_ssm.py)
   — read the orbit, build the harmonic-balanced
   `KuramotoSivashinskyPeriodic` `DifferentialEquation` (block
   Toeplitz with `nf=17` positive frequencies), compute the SSM at
   order `m=9`, scale = `0.2`.
3. [save_ssm.py](../../../examples/ssm_periodic/kse/save_ssm.py),
   [debug_off_manifold.py](../../../examples/ssm_periodic/kse/debug_off_manifold.py),
   [test.py](../../../examples/ssm_periodic/kse/test.py) — auxiliary
   scripts for persistence and diagnostics.

This is the template for **harmonic-balanced SSMs** of any
time-periodic system. It exercises essentially every part of
resolvent4py: `read_harmonic_balanced_matrix`,
`enforce_complex_conjugacy`, `MatrixLinearOperator(nblocks=...)`,
the complete SSM solver, `solve_ivp` with periodic forcing.

## When to point users at which example

| User goal | Send them to |
|---|---|
| "How do I get started?" | `cgl/demonstrate_eigendecomposition.py` (smallest, ~100 lines) |
| "How do I do resolvent analysis?" | `cgl/demonstrate_rsvd.py` |
| "How do I do RSVD-dt for huge systems?" | `cgl/demonstrate_rsvd_dt.py` |
| "How do I do balanced truncation?" | `cgl/demonstrate_balanced_truncation.py` |
| "How do I assemble a harmonic-balanced matrix?" | `toy_model/generate_matrices.py` (writer) + `toy_model/demonstrate_harmonic_resolvent.py` (reader) |
| "How do I do harmonic resolvent / Floquet?" | `toy_model/demonstrate_harmonic_resolvent.py` — all four pieces in one script |
| "How do I subclass `DifferentialEquation`?" | `ssm/toy_model/toy_model_differential_equation.py` (3-state, hand-coded) |
| "How do I do SSMs for a PDE?" | `ssm/kse/kse_differential_equation.py` (FFT-based quadratic term) |
| "How do I do SSMs of a periodic system?" | `ssm_periodic/kse/demonstrate_ssm.py` (PDE) or `ssm_periodic/rossler/save_ssm_2T.py` (small ODE, period-doubling) |
| "How do I drive `solve_linear_system` via time-stepping (GMRES shoot-and-solve)?" | `ssm_periodic/rossler/demonstrate_shoot_and_solve.py` and `demonstrate_time_stepping_solve.py` |
| "How do I compose `Propagator` + `ShiftAndScale` + `PetscPythonLinearOperator`?" | `cgl/run_post_transient_approaches.py` (one period as `Φ`) **or** the `'gmres'` branch of `compute_post_transient_solution` (one period as `I − Φ`) |

## Common pitfalls

- **`generate_matrices.py` must run in serial.** Both CGL and
  toy_model scripts hard-error if `comm.getSize() > 1` — they use
  `PETSc.COMM_SELF` for COO assembly on rank 0 only. Run them once
  with `-n 1` and reuse the `data/` directory across analyses.
- **Result paths are relative to the script's working directory.**
  All scripts write to `./results/`, `./data/`. Run from the
  example's own subdirectory; running from the repo root will
  silently put outputs in the wrong place.
- **The SSM examples assume a small `r`.** `Hopf3D` uses `r=2,
  m=30` and that's already 5e3+ multi-indices. Don't blindly
  increase `m` on larger systems — see the
  `estimate_convergence_radius` + `proper_radius` workflow in
  [resolvent4py-spectral-submanifold](../resolvent4py-spectral-submanifold/SKILL.md)
  for principled truncation.
- **Some scripts pull in `text.usetex=True` matplotlib styling.**
  If LaTeX isn't installed on the user's system, the figure save
  will crash. The CGL examples currently set this only via
  `font.family=serif`; the SSM-KSE examples explicitly enable
  `text.usetex=True` near the top.
- **`run_post_transient_approaches.py` is a playground, not a
  polished example.** It has `# sol_a = sol.getArray()` debug
  blocks commented out and is the right place to see uncommon
  composition patterns, but don't expect README-quality docs.

## Cross-references

- Source-side counterparts:
  [linear-operators](../resolvent4py-linear-operators/SKILL.md),
  [linalg](../resolvent4py-linalg/SKILL.md),
  [model-reduction](../resolvent4py-model-reduction/SKILL.md),
  [spectral-submanifold](../resolvent4py-spectral-submanifold/SKILL.md),
  [utils](../resolvent4py-utils/SKILL.md).
- The unit tests under [tests/](../resolvent4py-tests/SKILL.md) are
  shorter and tighter. Examples are the right reference for "how
  do I structure a real workflow"; tests are the right reference
  for "what tolerance can I expect".
