"""
Harmonic-balance Newton solver for the KSE **2T-periodic** orbit at
``nu_target`` (past the first period-doubling), with the analytic
block-Toeplitz Jacobian and a bordered orthogonal phase condition.

Mirrors ``compute_orbit_hb_newton.py`` but targets the period-doubled
attractor.  Differences:

  · Phase 1 forward-integrates *at ``nu_target``* (not just before the
    bifurcation), so the DNS settles onto the 2T attractor.  The
    initial-guess period is picked by closest-return in
    ``t ∈ [1.5·T_1T, 2.5·T_1T]`` — the 2T orbit's state at ``T_1T``
    is deliberately *different* from ``t = 0`` (that's what period-
    doubling means), only ``t = 2·T_1T`` returns.

  · Phase 2 runs HB Newton at fundamental frequency
    ``ω_2T = 2π/(2·T_phys)`` and higher ``nf_HB`` (2T orbit has richer
    Fourier content).

  · Output filename ``data/periodic_orbit_2T.npz``.

Prereqs: none — self-contained, only uses ``spatial_operators``,
``time_integrator``, and ``kse_differential_equation``.
"""

import os
from functools import partial

import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import imex_step_3 as imex_step
from kse_differential_equation import KuramotoSivashinskyPeriodic


# ── Parameters ─────────────────────────────────────────────────────────
# Keep in sync with the T-periodic script; only nf_HB grows.
nu_target = 1.0 / 33.3353          # past the first period-doubling
n = 16                              # # Fourier sine modes
n_pts = 4 * n                       # dealiased physical grid
nf_HB = 60                          # 2T-periodic harmonics — higher than
                                    # the 1T script (40) since the 2T
                                    # orbit has more spectral content.
n_time_save = 1024                  # samples in the saved orbit

# Phase-1 forward integration at nu_target → 2T attractor
T_1T_approx = 0.88                  # rough 1T period at this ν
T_guess_approx = 2.0 * T_1T_approx  # rough 2T period
dt = 1e-3
T_transient = 200.0

# Newton convergence
newton_tol = 1e-12
newton_max_iter = 60


# ── Phase 1 — forward integration at nu_target → 2T attractor ─────────
print(f"Phase 1: integrating KSE at 1/ν = {1/nu_target:.4f}  "
      f"for t = {T_transient}s  (dt = {dt}) to settle onto the 2T orbit ...")
lam = linear_eigenvalues(n, nu_target)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)
c = np.zeros(n); c[0] = 1.0
n_transient = int(T_transient / dt)
for step in range(n_transient):
    c = imex_step(c, lam, N_fn, dt)
    if step > 0 and step % 50000 == 0:
        print(f"  step {step:>8d}: t = {step*dt:>7.2f}, "
              f"‖c‖ = {np.linalg.norm(c):.4f}")

# Record ~3·T_2T so the closest-return search window sits safely inside.
n_record = int(3.0 * T_guess_approx / dt)
C_record = np.zeros((n, n_record + 1))
C_record[:, 0] = c
for k in range(1, n_record + 1):
    c = imex_step(c, lam, N_fn, dt)
    C_record[:, k] = c

# Find closest return to c(0) in the 2T-neighbourhood window.  For a
# genuine period-doubled orbit the state at T_1T is DIFFERENT from t=0;
# only t ≈ 2·T_1T (= T_2T) returns.
c0 = C_record[:, 0]
diff = np.linalg.norm(C_record - c0[:, None], axis=0)
k_lo = int(1.5 * T_1T_approx / dt)
k_hi = int(2.5 * T_1T_approx / dt)
k_period = k_lo + int(np.argmin(diff[k_lo:k_hi]))
T_guess = k_period * dt

# Sanity check: also report closure at t ≈ T_1T; if this were smaller
# than the 2T closure we'd actually be back on the 1T attractor and the
# HB-Newton below would silently converge to it.
k_1T = int(T_1T_approx / dt)
k_1T_win = list(range(int(0.8 * T_1T_approx / dt), int(1.2 * T_1T_approx / dt)))
k_1T_best = k_1T_win[int(np.argmin([diff[i] for i in k_1T_win]))]
print(
    f"  approximate 2T period T_guess = {T_guess:.6f}  (k = {k_period}),  "
    f"closure = {diff[k_period]:.4e}"
)
print(
    f"  (closest-return near T_1T = {T_1T_approx}:  "
    f"k = {k_1T_best},  T = {k_1T_best*dt:.6f},  "
    f"closure = {diff[k_1T_best]:.4e})"
)
if diff[k_1T_best] < 0.1 * diff[k_period]:
    print("  ⚠  1T-closure is MUCH tighter than 2T-closure.  The transient "
          "hasn't landed on a period-doubled orbit; either nu_target is "
          "wrong side of the bifurcation, or T_transient is too short.")

C_guess_open = C_record[:, :k_period]      # endpoint-excluded


# ── Phase 2 — HB-Newton at fundamental ω_2T = 2π / T_2T ───────────────
print(f"\nPhase 2: HB-Newton at 1/ν = {1/nu_target:.4f}  "
      f"(nf = {nf_HB},  T_2T-init = {T_guess:.6f}) ...")

n_harm = 2 * nf_HB + 1
n_t_eval = 4 * nf_HB + 1
N_C = n * n_harm

omega = 2.0 * np.pi / T_guess
C_init = resample(C_guess_open, 2 * nf_HB + 1, axis=1)
time_init = np.linspace(0.0, T_guess, 2 * nf_HB + 1, endpoint=False)
eq = KuramotoSivashinskyPeriodic(
    n=n, nu=nu_target, nf=nf_HB, nfb=0, c_star=C_init,
    time=time_init, n_pts=n_pts,
    periodic_diffeq=(omega * np.arange(nf_HB + 1), time_init, False),
)
L_diag = linear_eigenvalues(n, nu_target)
k_arr = np.arange(-nf_HB, nf_HB + 1)

Chat = eq.temporal_fft(C_init, nf_HB)


def _physical_states_on_eval_grid(Chat, omega):
    eq.omega = omega
    t_eval = np.linspace(0.0, 2.0 * np.pi / omega, n_t_eval, endpoint=False)
    return eq.temporal_ifft(Chat, t_eval).real, t_eval


def compute_residual(Chat, omega):
    """g_k = i k ω xhat_k − L xhat_k − N_k(ifft(xhat))."""
    x_t, _ = _physical_states_on_eval_grid(Chat, omega)
    N_t = np.zeros_like(x_t)
    for i in range(n_t_eval):
        N_t[:, i] = eq._evaluate_quadratic_term_numpy(x_t[:, i], x_t[:, i])
    Nhat_big = eq.temporal_fft(N_t, 2 * nf_HB)
    Nhat = Nhat_big[:, nf_HB : 3 * nf_HB + 1]
    return 1j * k_arr[None, :] * omega * Chat - L_diag[:, None] * Chat - Nhat


def compute_A_p(Chat, omega):
    """A_p = (FFT[A(t)])_p with A(t) = L + 2 B(·, x*(t))."""
    x_t, _ = _physical_states_on_eval_grid(Chat, omega)

    A_t = np.zeros((n_t_eval, n, n))
    for i in range(n_t_eval):
        A_ti = np.diag(L_diag.astype(float))
        x_ti = x_t[:, i]
        for j in range(n):
            e_j = np.zeros(n); e_j[j] = 1.0
            A_ti[:, j] += 2.0 * eq._evaluate_quadratic_term_numpy(e_j, x_ti)
        A_t[i] = A_ti

    A_fft = np.fft.fft(A_t, axis=0) / n_t_eval

    A_p = np.zeros((2 * nf_HB + 1, n, n), dtype=complex)
    A_p[nf_HB] = A_fft[0]
    for p in range(1, nf_HB + 1):
        A_p[nf_HB + p] = A_fft[p]
        A_p[nf_HB - p] = A_fft[n_t_eval - p]
    return A_p


def build_bordered_system(Chat, omega):
    """[ J     | J_ω ] [δxhat] = [-g]
       [ w^*   | 0   ] [δω   ]   [ 0]"""
    A_p = compute_A_p(Chat, omega)
    g = compute_residual(Chat, omega)

    M = np.zeros((N_C + 1, N_C + 1), dtype=complex)
    for ki in range(n_harm):
        k = ki - nf_HB
        for ji in range(n_harm):
            j = ji - nf_HB
            p = k - j
            if abs(p) <= nf_HB:
                M[ki*n:(ki+1)*n, ji*n:(ji+1)*n] = -A_p[nf_HB + p]
        M[ki*n:(ki+1)*n, ki*n:(ki+1)*n] += (1j * k * omega) * np.eye(n)

    M[:N_C, N_C] = (1j * k_arr[None, :] * Chat).flatten(order="F")
    w = (1j * k_arr[None, :] * omega * Chat).flatten(order="F")
    M[N_C, :N_C] = (w / np.linalg.norm(w)).conj()
    M[N_C, N_C] = 0.0

    rhs = np.zeros(N_C + 1, dtype=complex)
    rhs[:N_C] = -g.flatten(order="F")
    return M, rhs, g


print(f"  {'it':>3s}  {'‖g‖':>12s}  {'ω':>16s}  {'T_2T':>12s}")
for it in range(newton_max_iter):
    M, rhs, g = build_bordered_system(Chat, omega)
    g_norm = np.linalg.norm(g)
    T_curr = 2.0 * np.pi / omega
    print(f"  {it:>3d}  {g_norm:.6e}  {omega:.12f}  {T_curr:.10f}")
    if g_norm < newton_tol:
        print("  Converged.")
        break

    dx = np.linalg.solve(M, rhs)
    dChat = dx[:N_C].reshape((n, n_harm), order="F")
    domega = dx[N_C].real

    Chat = Chat + dChat
    omega = omega + domega

    Chat[:, nf_HB] = Chat[:, nf_HB].real
    for k in range(1, nf_HB + 1):
        avg = 0.5 * (Chat[:, nf_HB + k] + Chat[:, nf_HB - k].conj())
        Chat[:, nf_HB + k] = avg
        Chat[:, nf_HB - k] = avg.conj()
else:
    print(f"  Did not converge in {newton_max_iter} iterations.")


# ── Phase 3 — reconstruct on save grid and write to disk ──────────────
T_sol = 2.0 * np.pi / omega
print(f"\nNewton-refined  T_2T = {T_sol:.12f}")

eq.omega = omega
time_save = np.linspace(0.0, T_sol, n_time_save + 1, endpoint=True)
C_save = eq.temporal_ifft(Chat, time_save).real
closure_abs = float(np.linalg.norm(C_save[:, -1] - C_save[:, 0]))
closure_rel = closure_abs / float(np.linalg.norm(C_save[:, 0]))
print(f"closure (save grid) = {closure_abs:.4e}  (rel {closure_rel:.4e})")

os.makedirs("data", exist_ok=True)
np.savez(
    "data/periodic_orbit_2T.npz",
    C=C_save,
    T=T_sol,
    nu=nu_target,
    n=n,
    n_pts=n_pts,
    nf=nf_HB,
)
print("Saved refined 2T orbit → data/periodic_orbit_2T.npz")


# ── Phase 4 — DNS verification ────────────────────────────────────────
print("\nPhase 4: DNS verification ...")

A_p_check = compute_A_p(Chat, omega)
J_only = np.zeros((N_C, N_C), dtype=complex)
for ki in range(n_harm):
    k = ki - nf_HB
    for ji in range(n_harm):
        j = ji - nf_HB
        p = k - j
        if abs(p) <= nf_HB:
            J_only[ki*n:(ki+1)*n, ji*n:(ji+1)*n] = -A_p_check[nf_HB + p]
    J_only[ki*n:(ki+1)*n, ki*n:(ki+1)*n] += (1j * k * omega) * np.eye(n)

v_tangent = (1j * k_arr[None, :] * omega * Chat).flatten(order="F")
Jv = J_only @ v_tangent
ker_err = np.linalg.norm(Jv) / (np.linalg.norm(J_only) * np.linalg.norm(v_tangent))
print(f"  ‖J · (i k ω xhat)‖ / (‖J‖·‖v‖) = {ker_err:.4e}   "
      f"(≈ machine-zero ⇒ exact orbit + correct Jacobian)")


dt_dns = 5e-5
n_steps_dns = int(round(T_sol / dt_dns))
dt_dns = T_sol / n_steps_dns
lam_target = linear_eigenvalues(n, nu_target)
N_fn_target = partial(nonlinear_rhs, n_pts=n_pts)
c_dns = C_save[:, 0].copy()
for _ in range(n_steps_dns):
    c_dns = imex_step(c_dns, lam_target, N_fn_target, dt_dns)
err_abs = float(np.linalg.norm(c_dns - C_save[:, 0]))
err_rel = err_abs / float(np.linalg.norm(C_save[:, 0]))
print(f"  DNS integration over [0, T_2T={T_sol:.6f}] with dt = "
      f"{dt_dns:.2e}  ({n_steps_dns} steps)")
print(f"  ‖c_DNS(T_2T) - c(0)‖ = {err_abs:.4e}  (rel {err_rel:.4e})")


# Quick plot
fig, ax = plt.subplots(figsize=(8, 4))
for j in range(min(5, n)):
    ax.plot(time_save, C_save[j], lw=0.8, label=fr"$c_{{{j+1}}}$")
ax.set_xlabel("t"); ax.set_ylabel(r"$c_j(t)$")
ax.set_title(fr"HB-Newton refined 2T orbit  "
             fr"(1/$\nu$ = {1/nu_target:.4f}, $T_{{2T}}$ = {T_sol:.6f}, "
             fr"closure rel = {closure_rel:.2e})")
ax.legend(fontsize=8, ncol=5, loc="best"); ax.grid(alpha=0.3)
os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/orbit_hb_newton_2T.png", dpi=140, bbox_inches="tight")
print("Saved plot → results/orbit_hb_newton_2T.png")

os._exit(0)
