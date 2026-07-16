"""
Compute all three Floquet multipliers of the Rössler T-periodic base
flow by direct monodromy integration.

For a 3D state, the full spectrum is three multipliers μ_j = exp(Λ_j·T).
This is way cheaper than solving the (2T-HB, block-Toeplitz) eigenvalue
problem in ``save_eigendecomp_2T.py`` — that one uses SLEPc's shift-
invert Arnoldi and returns only the leading ``n_evals`` folded copies.
Here we just integrate the variational equation

    dΦ/dt = A(t) · Φ,   Φ(0) = I ∈ ℝ^{3×3},

    A(t) = M + 2·B(c_*(t), ·)

column-by-column from t = 0 to t = T_base, and diagonalise M = Φ(T_base).

Neutral direction: one multiplier is +1 by construction (orbit tangent
dc_*/dt is a solution of dv/dt = A(t) v).  Reports how close the
integrated eigenvalue is to unity as a self-check.

Output: ``data/floquet_multipliers.npz`` containing
    L    : (3,) complex Floquet exponents Λ_j = log(μ_j) / T_base
    mu   : (3,) complex Floquet multipliers over T_base
    M    : (3, 3) monodromy matrix
    T    : T_base

No PETSc / MPI — pure numpy + scipy.
"""
import os

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

from rossler_rhs import perturbation_linear_action


# ── Load T-periodic base flow ─────────────────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T_base = float(data["T"])
C = data["C"]                                  # (3, n_time_save)  1T orbit
n = int(data["n"])
n_time_save = C.shape[1]

print(f"Rossler:  c = {c},  T_base = {T_base:.6f},  n = {n}")
print(f"Loaded 1T orbit from data/periodic_orbit.npz  "
      f"({n_time_save} samples).")


# ── c_*(t) interpolant  ───────────────────────────────────────────────
# Closes cleanly: C[:, -1] == C[:, 0]  (periodic).
time_orbit = np.linspace(0.0, T_base, n_time_save, endpoint=True)
_c_interp = interp1d(time_orbit, C, axis=1, kind="cubic",
                     assume_sorted=True)


def c_star_at(t):
    return _c_interp(t % T_base)


# ── Variational RHS  dv/dt = A(t) v  ──────────────────────────────────
def var_rhs(t, v):
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c)


# ── Integrate M columns via variational equation ──────────────────────
# rtol/atol very tight — cheap here since n=3 and the ODE is linear.
rtol, atol = 1e-13, 1e-15
print(f"\nIntegrating variational equation column-by-column  "
      f"(RK45, rtol={rtol}, atol={atol}) ...")
Phi = np.zeros((n, n))
for j in range(n):
    e_j = np.zeros(n); e_j[j] = 1.0
    sol = sp.integrate.solve_ivp(
        var_rhs, [0.0, T_base], e_j,
        method="RK45", rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Column {j}: solver failed — {sol.message}")
    Phi[:, j] = sol.y[:, -1]
    print(f"  column {j+1}/{n}:  ‖Φ_j(T)‖ = {np.linalg.norm(Phi[:, j]):.4e}")


# ── Diagonalise the monodromy ─────────────────────────────────────────
mu, V = np.linalg.eig(Phi)
# Sort by |μ| descending (unstable / neutral / stable order)
order = np.argsort(-np.abs(mu))
mu = mu[order]
V = V[:, order]

# Floquet exponents (branch: principal log)
L = np.log(mu) / T_base

print(f"\nMonodromy Φ(T_base) = ")
np.set_printoptions(precision=6, suppress=False)
print(Phi)

print(f"\nFloquet multipliers over T_base "
      f"(sorted by |μ| descending):")
for j in range(n):
    print(f"  μ_{j+1} = {mu[j].real:+.6e}{mu[j].imag:+.6e}j    "
          f"|μ_{j+1}| = {abs(mu[j]):.6e}    "
          f"Λ_{j+1} = {L[j].real:+.6e}{L[j].imag:+.6e}j")


# ── Consistency check — one multiplier must be ≈ 1 (neutral) ─────────
i_neutral = int(np.argmin(np.abs(mu - 1.0)))
err_neutral = abs(mu[i_neutral] - 1.0)
print(f"\nNeutral-direction check:")
print(f"  Closest-to-unity multiplier:  μ_{i_neutral+1} = {mu[i_neutral]:+.4e}")
print(f"  |μ − 1|  = {err_neutral:.4e}   "
      f"(should be tiny; measures integration accuracy)")


# ── Also report signed T_base multipliers (parity-based branch fix) ──
# In the 2T-HB frame we resolved the branch via HB parity.  The direct
# monodromy calculation here gives the multipliers WITHOUT any branch
# ambiguity — μ over exactly one T_base period.
mag_products = float(np.abs(np.prod(mu)))
det_Phi = float(np.linalg.det(Phi))
tr_Phi = float(np.trace(Phi))
# Trace-determinant identity check:  |det Φ| = |Π μ|
print(f"\nSanity:")
print(f"  det Φ                 = {det_Phi:.6e}  (= Π μ_j)")
print(f"  |Π μ_j|               = {mag_products:.6e}")
print(f"  tr Φ                  = {tr_Phi:.6e}  (= Σ μ_j)")
print(f"  Σ Re μ_j              = {mu.real.sum():.6e}")


# ── Save ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
outpath = "data/floquet_multipliers.npz"
np.savez(outpath, L=L, mu=mu, M=Phi, T=T_base, c=c, n=n)
print(f"\nSaved → {outpath}")
