"""
Time-step the Kuramoto-Sivashinsky equation to a periodic orbit.

The KSE on [0, 2π] with sine-series ansatz:

    u_t = -u*u_x - u_xx - ν*u_xxxx

For sufficiently small ν (past the Hopf bifurcation of the nontrivial
equilibrium), the dynamics settle onto a stable limit cycle.  This script:

  1. Integrates from a random initial condition until transients decay.
  2. Detects the period T via Poincaré section crossings on a chosen mode.
  3. Refines the period estimate by averaging over several crossings.
  4. Saves the periodic orbit (one full period) to disk.
"""

import os
import numpy as np
from functools import partial
from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import imex_step

import matplotlib.pyplot as plt


# ── PDE / spatial discretization ────────────────────────────────────────────
nu = 0.058     # past the Hopf bifurcation → periodic orbit
n = 64          # number of Fourier sine modes
n_pts = 4 * n   # physical-space grid points for dealiasing

# ── Time stepping ───────────────────────────────────────────────────────────
dt = 1e-3                # time step
T_transient = 500.0      # integration time to wash out transients
T_detect = 200.0         # window for period detection after transient
save_every_detect = 1    # save every step during detection phase

# ── Periodicity detection ───────────────────────────────────────────────────
poincare_mode = 0        # index of the sine mode used for the Poincaré section
n_crossings_skip = 2     # skip the first few crossings (residual transient)
n_crossings_avg = 8      # average this many consecutive periods for T estimate
period_rtol = 1e-3       # relative tolerance: all averaged periods must agree


# %% Setup

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)


# %% Phase 1 – Transient integration

rng = np.random.default_rng(42)
c = 1e-3 * rng.standard_normal(n)

n_transient = int(T_transient / dt)
print(f"Phase 1: integrating {n_transient} steps (T={T_transient}) "
      f"to wash out transients ...")

for _ in range(n_transient):
    c = imex_step(c, lam, N_fn, dt)

print(f"  done.  |c| = {np.linalg.norm(c):.6e}")


# %% Phase 2 – Period detection via Poincaré section crossings
#
# We monitor c[poincare_mode] and record times when it crosses a reference
# level (its current value) with positive slope.  Consecutive crossing times
# give period estimates.

print(f"\nPhase 2: detecting period (monitoring mode {poincare_mode + 1}) ...")

# Use the current value of the monitored mode as the Poincaré section level
section_level = c[poincare_mode]

n_detect = int(T_detect / dt)
crossing_times: list[float] = []
prev_val = c[poincare_mode] - section_level
t_current = 0.0

for step in range(1, n_detect + 1):
    c = imex_step(c, lam, N_fn, dt)
    t_current = step * dt
    cur_val = c[poincare_mode] - section_level

    # Detect upward zero crossing
    if prev_val < 0.0 and cur_val >= 0.0:
        # Linear interpolation for sub-step accuracy
        frac = prev_val / (prev_val - cur_val)
        t_cross = (step - 1 + frac) * dt
        crossing_times.append(t_cross)

    prev_val = cur_val

n_required = n_crossings_skip + n_crossings_avg + 1
if len(crossing_times) < n_required:
    raise RuntimeError(
        f"Only found {len(crossing_times)} Poincaré crossings in the "
        f"detection window (need {n_required}).  "
        f"Try increasing T_detect or T_transient."
    )

# Compute period estimates from consecutive crossings (after skipping early ones)
crossing_times = np.array(crossing_times)
usable = crossing_times[n_crossings_skip:]
periods = np.diff(usable[: n_crossings_avg + 1])

T_mean = np.mean(periods)
T_spread = (np.max(periods) - np.min(periods)) / T_mean

print(f"  {len(crossing_times)} crossings detected")
print(f"  period estimates: {periods}")
print(f"  mean period T = {T_mean:.8f}")
print(f"  relative spread = {T_spread:.2e}")

if T_spread > period_rtol:
    print(f"  WARNING: spread {T_spread:.2e} > tolerance {period_rtol:.2e}. "
          f"The orbit may not be well converged.")


# %% Phase 3 – Record one full period

T_orbit = T_mean
n_orbit = int(np.round(T_orbit / dt))
dt_orbit = T_orbit / n_orbit  # adjusted dt to land exactly on T_orbit

print(f"\nPhase 3: recording one orbit (T = {T_orbit:.8f}, "
      f"{n_orbit} steps, dt_orbit = {dt_orbit:.6e}) ...")

# Continue from the last crossing so the saved orbit starts near the section
# Re-integrate one more period to reach a clean crossing point
# (we are currently dt past the last crossing; march forward ~one period)
n_approach = n_orbit
for _ in range(n_approach):
    c = imex_step(c, lam, N_fn, dt)

# Now record
t_orbit = np.linspace(0, T_orbit, n_orbit + 1)
C_orbit = np.zeros((n, n_orbit + 1))
C_orbit[:, 0] = c.copy()

for i in range(1, n_orbit + 1):
    c = imex_step(c, lam, N_fn, dt_orbit)
    C_orbit[:, i] = c

# Check periodicity: how close is the final state to the initial state?
closure_error = np.linalg.norm(C_orbit[:, -1] - C_orbit[:, 0])
closure_rel = closure_error / np.linalg.norm(C_orbit[:, 0])
print(f"  orbit closure error (absolute): {closure_error:.6e}")
print(f"  orbit closure error (relative): {closure_rel:.6e}")


# %% Save results

outdir = "data"
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(outdir, "periodic_orbit.npz")
np.savez(
    outfile,
    nu=nu,
    n=n,
    n_pts=n_pts,
    T=T_orbit,
    dt=dt_orbit,
    t=t_orbit,
    C=C_orbit,
)
print(f"\nSaved periodic orbit to {outfile}")
print(f"  C.shape = {C_orbit.shape}  (n_modes x n_snapshots)")
print(f"  T = {T_orbit:.8f}")


# %% Time-series plot

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
modes_to_plot = [1, 2, 3, 4]
for ax, mode in zip(axes, modes_to_plot):
    ax.plot(t_orbit, C_orbit[mode - 1, :], "k", lw=1)
    ax.set_ylabel(rf"$c_{{{mode}}}$")
axes[-1].set_xlabel(r"$t$")
axes[0].set_title(
    rf"Periodic orbit of KSE ($\nu = {nu}$, $T = {T_orbit:.4f}$)"
)
plt.tight_layout()
plt.show()


# %% Energy spectrum via temporal FFT

# Kinetic energy per mode: E_j(t) = 0.5 * c_j(t)^2
# Total energy: E(t) = sum_j E_j(t) = 0.5 * ||c||^2
E = 0.5 * np.sum(C_orbit ** 2, axis=0)

# FFT of E(t) over one period (exclude last point = duplicate of first)
E_periodic = E[:-1]
N_fft = len(E_periodic)
E_hat = np.fft.rfft(E_periodic) / N_fft
freqs = np.fft.rfftfreq(N_fft, d=dt_orbit)

# Amplitude spectrum (skip DC component for log scale)
amplitude = np.abs(E_hat)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(freqs[:20], amplitude[:20], "k", lw=1, marker='o')
ax.set_xlabel(r"Frequency $f$")
ax.set_ylabel(r"$|\hat{E}(f)|$")
ax.set_title(rf"Energy spectrum of periodic orbit ($\nu = {nu}$)")
ax.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.show()
