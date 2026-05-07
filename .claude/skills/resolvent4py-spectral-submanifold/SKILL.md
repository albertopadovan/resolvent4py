---
name: resolvent4py-spectral-submanifold
description: Spectral submanifold reduction for quadratic dynamical systems q'=Aq+B(q,q) — polynomial parametrization P(s) tangent to a chosen spectral subspace, with intrinsic dynamics ds/dt=Lambda*s+g(s). Use when the user asks about SSMs, invariant manifolds, latent-space dynamics, or polynomial expansions around an eigenspace.
---

# `resolvent4py.spectral_submanifold`

Compute spectral submanifolds (SSMs) for quadratic dynamical systems
`q̇ = Aq + B(q,q)`. An SSM is an invariant manifold tangent to an
`r`-dimensional spectral subspace of `A`; the SSM polynomial
`P(s) = Σ p_j s^j` and the intrinsic dynamics
`ṡ = Λs + g(s) = Σ g_j s^j` are computed order-by-order from the
homological equations.

## Files

| File | Class / API |
|---|---|
| [differential_equation.py](../../../src/resolvent4py/spectral_submanifold/differential_equation.py) | `DifferentialEquation` ABC — user supplies `evaluate_linear_term`, `evaluate_quadratic_term`, `solve_linear_system`. |
| [spectral_submanifold.py](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py) | `SpectralSubmanifold` — solver. Public methods: `solve`, `decode`, `encode`, `latent_space_dynamics`, `estimate_convergence_radius`. |

## How it works

### `DifferentialEquation` (you implement this)

The abstract base requires three methods:

- `evaluate_linear_term(q)` → `Aq`
- `evaluate_quadratic_term(q1, q2)` → `B(q1, q2)` (must be symmetric)
- `solve_linear_system(s, b)` → `(sI − A)⁻¹ b` for arbitrary complex
  shift `s`

Optionally override `evaluate_dynamics(t, q)` if your RHS has terms
beyond `Aq + B(q,q)`. `poly_deg` must be 2 (only quadratic
nonlinearities are currently supported — see the `__init__` guard in
[spectral_submanifold.py:75](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py#L75)).

### `SpectralSubmanifold.solve(Phi, Psi, Lams, scaling, verbose)`

Given right eigenvectors `Φ`, left eigenvectors `Ψ`, and eigenvalues
`Λ` of the chosen master subspace (precomputed via
[res4py.linalg.eig](../resolvent4py-linalg/SKILL.md) +
`match_right_and_left_eigenvectors`), this drives the SSM expansion
to order `m`:

1. Enumerate all multi-indices `j ∈ ℕ^r` with `|j| ≤ m`
   (`compute_multiindices`).
2. For each multi-index in increasing-order order:
   - assemble RHS contributions from quadratic interactions
     (`ssm_quad_rhs_idc`) and from the previously-known nonlinear
     dynamics terms (`ssm_nonlin_dynmc`),
   - project: `g_j = Ψ* rhs` (or zero if conjugacy to linear
     dynamics is requested),
   - solve `(λ_j I − A) p_j = rhs − Φ g_j` via
     `solve_linear_system`,
   - sanity-check `|Ψ* p_j| < 1e-6` (raises `ValueError` if not —
     usually means you need to adjust `scaling`).
3. Cache `ps`, `gs`, `Lams`, `V`, `W` on the object for
   `decode/encode/latent_space_dynamics` to use later.

`scaling` rescales eigenvectors as `V = Φ·s, W = Ψ/s`. This is the
knob to push around if convergence is poor or
`|Ψ* p_j|` exceeds tolerance.

### Other methods

- `decode(s)` → full-state `Σ_j p_j Π_k s_k^{j_k}`.
- `encode(q)` → latent-space coordinates `Ψ* q`.
- `latent_space_dynamics(t, s)` → `Λs + Σ_j g_j s^j`, suitable for
  `scipy.integrate.solve_ivp`.
- `estimate_convergence_radius()` — fits `log10(C_k)` vs `k` (where
  `C_k = Σ_{|j|=k} ||p_j||_1`) using the **Theil–Sen** estimator.
  This is robust to near-resonance outliers; ordinary least squares
  is skewed by a single near-resonant `p_j` blowing up. Skips the
  first 33% of orders (low-order terms aren't asymptotic).

## Multi-index combinatorics (helpers, internal)

The private helpers `_generate_constrained_compositions`,
`_generate_quadratic_pairs`, and `_generate_nonlinear_dynamics_pairs`
enumerate all `(i, l)` multi-index pairs for the polynomial
convolutions in the homological equations. They're cached on the
object as `ssm_multiindices`, `ssm_quad_rhs_idc`,
`ssm_nonlin_dynmc` at construction time so `solve` doesn't redo
combinatorics per call.

## Common pitfalls

- `ps[0]` is a zero vector (manifold passes through the origin); the
  next `r` entries `ps[1..r]` are the columns of `V`. Order-`k>1`
  terms start at index `r+1`. Don't index by integer if you can
  avoid it — go through `_get_multiindex_index` like
  [spectral_submanifold.py:376](../../../src/resolvent4py/spectral_submanifold/spectral_submanifold.py#L376).
- `_check_eigen_triplets` is a *warning*, not an error — it prints
  if biorthogonality / eigenvalue residuals exceed tolerance but
  proceeds. Watch the stdout when running `solve`.
- The intercept and slope from `estimate_convergence_radius` feed
  into `proper_radius()` in
  [utils/ssm.py](../../../src/resolvent4py/utils/ssm.py) to decide
  what fraction of `R_estimate` is safe for a given `manifold_tol`.

## Tests

[resolvent4py-tests-spectral-submanifold](../resolvent4py-tests-spectral-submanifold/SKILL.md)
exists but contains only an empty `__init__.py` at present —
SSM functionality is exercised through the
[examples/ssm/](../../../examples/ssm/) and
[examples/ssm_periodic/](../../../examples/ssm_periodic/)
use-cases instead.
