"""
Lock onto the T-periodic limit cycle of the Rössler system at
c = 5.3 (Padovan & Rowley 2022, Sec. IV).

Strategy:
  1. Integrate scipy.integrate.solve_ivp for a long transient to
     wash out the initial condition.
  2. Detect period via x = 0 upward Poincaré crossings (quadratic
     interpolation in time).
  3. Newton-shoot on the period using the variational Jacobian.
  4. Record one full period at uniform spacing for HB use.

Pure numpy / scipy — no PETSc, no MPI.
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from rossler_rhs import rossler_rhs


# ── Parameters ──────────────────────────────────────────────────────────────
c = 5.3  # bifurcation parameter (paper Sec. IV)

t_transient = 10000.0  # transient integration time
t_detect = 400.0  # window for Poincaré-crossing detection
rtol = 1e-12
atol = 1e-12

n_crossings_skip = 4  # skip residual transient
n_crossings_avg = 32  # average this many consecutive periods
period_rtol = 1e-6

newton_iters = 20
newton_tol = 1e-12

n_orbit = 512  # samples per period in the saved orbit


# %% Phase 1 – Transient integration to lock onto the attractor

rng = np.random.default_rng(0)
q0 = np.array([1.0, 0.0, 0.0]) + 1e-2 * rng.standard_normal(3)

print(f"Phase 1: integrating to t = {t_transient} to wash out transients ...")
sol = sp.integrate.solve_ivp(
    rossler_rhs,
    [0.0, t_transient],
    q0,
    args=(c,),
    method="DOP853",
    rtol=rtol,
    atol=atol,
)
q = sol.y[:, -1]
print(f"  done.  |q| = {np.linalg.norm(q):.6e}")


# %% Phase 2 – Period detection via x = 0 upward Poincaré crossings

print(f"\nPhase 2: detecting period (x = 0 upward crossing) ...")


def crossing_event(t, q, c_):
    return q[0]


crossing_event.terminal = False
crossing_event.direction = 1  # upward only

sol = sp.integrate.solve_ivp(
    rossler_rhs,
    [0.0, t_detect],
    q,
    args=(c,),
    method="DOP853",
    rtol=rtol,
    atol=atol,
    events=crossing_event,
    dense_output=True,
)
crossing_times = sol.t_events[0]
crossing_states = sol.y_events[0]

if len(crossing_times) < n_crossings_skip + n_crossings_avg + 1:
    raise RuntimeError(
        f"Only found {len(crossing_times)} crossings; "
        f"need at least {n_crossings_skip + n_crossings_avg + 1}."
    )

usable = crossing_times[n_crossings_skip:]
sub_periods = np.diff(usable[: n_crossings_avg + 1])
T_mean = float(np.mean(sub_periods))
T_spread = (sub_periods.max() - sub_periods.min()) / T_mean
print(f"  {len(crossing_times)} crossings found")
print(f"  mean period T = {T_mean:.10f}")
print(f"  relative spread = {T_spread:.2e}")

q_start = crossing_states[n_crossings_skip]
section_val = q_start[0]


# %% Phase 2.5 – Newton-shoot to refine the period

print(f"\nPhase 2.5: Newton-shoot refinement ...")


def integrate_to(q0_local, T_target):
    s = sp.integrate.solve_ivp(
        rossler_rhs,
        [0.0, T_target],
        q0_local,
        args=(c,),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    return s.y[:, -1]


T_newton = T_mean
for it in range(newton_iters):
    q_T = integrate_to(q_start, T_newton)
    r = q_T[0] - section_val
    dr_dT = rossler_rhs(T_newton, q_T, c)[0]
    if abs(dr_dT) < 1e-14:
        print(f"  iter {it}: derivative ~0 — stopping")
        break
    dT = -r / dr_dT
    T_newton = T_newton + dT
    print(
        f"  iter {it}: T = {T_newton:.12f},  |res| = {abs(r):.3e},  "
        f"dT = {dT:+.3e}"
    )
    if abs(r) < newton_tol:
        break


# %% Phase 3 – Sample one full period on a uniform grid

T_orbit = T_newton
print(
    f"\nPhase 3: sampling one orbit (T = {T_orbit:.10f}, "
    f"{n_orbit} samples) ..."
)

t_orbit = np.linspace(0.0, T_orbit, n_orbit + 1)
sol = sp.integrate.solve_ivp(
    rossler_rhs,
    [0.0, T_orbit],
    q_start,
    args=(c,),
    method="DOP853",
    rtol=rtol,
    atol=atol,
    t_eval=t_orbit,
)
C_orbit = sol.y  # shape (3, n_orbit + 1)

closure = np.linalg.norm(C_orbit[:, -1] - C_orbit[:, 0])
print(f"  orbit closure error: {closure:.3e}")


# %% Save

outdir = "data"
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(outdir, "periodic_orbit.npz")
np.savez(
    outfile,
    c=c,
    T=T_orbit,
    t=t_orbit,
    C=C_orbit,
)
print(f"\nSaved periodic orbit to {outfile}")
print(f"  C.shape = {C_orbit.shape}   T = {T_orbit:.8f}")


# %% Plots

# Time series
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
labels = ["x", "y", "z"]
for ax, comp, lab in zip(axes, C_orbit, labels):
    ax.plot(t_orbit, comp, "k", lw=1)
    ax.set_ylabel(lab)
axes[-1].set_xlabel("t")
axes[0].set_title(rf"Rössler periodic orbit (c={c}, T={T_orbit:.4f})")
plt.tight_layout()
plt.show()

# 3D phase portrait
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(C_orbit[0], C_orbit[1], C_orbit[2], "k", lw=1)
ax.scatter(*C_orbit.mean(axis=1), color="C3", s=40, label="temporal mean")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.tight_layout()
plt.show()

# Spectrum
C_per = C_orbit[:, :-1]
C_hat = np.fft.rfft(C_per, axis=1) / C_per.shape[1]
amp = np.max(np.abs(C_hat), axis=0)
n_harm = min(40, amp.size - 1)
k_idx = np.arange(1, n_harm + 1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(k_idx, amp[k_idx], "ko", mfc="none")
ax.set_xlabel("harmonic index k")
ax.set_ylabel(r"$\max_j |\hat{q}_j(k)|$")
ax.set_title(f"Rössler orbit amplitude spectrum (c={c}, T={T_orbit:.4f})")
ax.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.show()
