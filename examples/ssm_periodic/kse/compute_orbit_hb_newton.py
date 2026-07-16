"""
Harmonic-balance Newton solver for the KSE periodic orbit, with the
analytic block-Toeplitz Jacobian and a bordered orthogonal-update phase
condition.

Pipeline:

  Phase 1 — Forward-integrate the KSE at ``1/ν = 33.2701`` to settle onto
            the stable 1-cycle (just before the first period-doubling).
            Extract one period (≈ 0.88) as the Newton initial guess.

  Phase 2 — Solve

                g_k(xhat, ω) := i k ω xhat_k − L xhat_k − N_k(ifft(xhat)) = 0
                                                              k = −n_f … +n_f

            with Newton's method using the analytic Jacobian

                [D_xhat g]_{k,j} = i k ω δ_{kj} I − A_{k-j}
                [D_ω    g]_k     = i k xhat_k

            where ``A_p = (FFT[A(t)])_p`` with ``A(t) = L + 2·B(·, x*(t))``.
            For ``A(t)`` whose Fourier support is in ``[-n_f, +n_f]``, all
            ``A_p`` with ``|p| > n_f`` vanish, so the dense Jacobian is
            band-Toeplitz with bandwidth ``n_f``.

            Phase condition (bordered):

                w^* δxhat = 0,  w = (i k ω xhat) / ‖i k ω xhat‖

            i.e. Newton's update is orthogonal to the orbit-tangent
            direction in Fourier space, which kills the time-translation
            gauge.

            Temporal dealiasing: the residual evaluates ``N(x*(t))`` on a
            ``4·n_f + 1``-point grid so the Fourier coefficients
            ``N_k, k ∈ [-n_f, +n_f]`` are exact (no aliasing from N's
            support up to ``±2·n_f``).

  Output: ``data/periodic_orbit.npz``  containing  ``C, T, nu, n, n_pts, nf``.
"""

import os
import numpy as np
from functools import partial
from scipy.signal import resample
import matplotlib.pyplot as plt

from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import imex_step_3 as imex_step
from kse_differential_equation import KuramotoSivashinskyPeriodic


# ── Parameters ─────────────────────────────────────────────────────────────
nu_target = 1.0 / 33.3353          # ν used for the refined orbit

n = 16                            # # Fourier sine modes
n_pts = 4 * n                     # physical-space grid for dealiasing

nf_HB = 40                        # # T-periodic harmonics retained in Newton
n_time_save = 1024                # # time samples in the saved orbit

# Phase-1 initial-guess parameters (forward integration at nu_guess)
nu_guess = 1.0 / 33.2701          # near, but just before, the first doubling
T_guess_approx = 0.88             # rough period at this ν
dt = 1e-3
T_transient = 200.0

# Newton convergence
newton_tol = 1e-12
newton_max_iter = 50


# ────────────────────────────────────────────────────────────────────────────
# Phase 1 — forward integration → approximate periodic orbit
# ────────────────────────────────────────────────────────────────────────────
print(
    f"Phase 1: integrating KSE at 1/ν = {1/nu_guess:.4f}  "
    f"for t = {T_transient}s (dt = {dt}) to settle onto the 1-cycle ..."
)
lam_guess = linear_eigenvalues(n, nu_guess)
N_fn_guess = partial(nonlinear_rhs, n_pts=n_pts)
c = np.zeros(n); c[0] = 1.0
n_transient = int(T_transient / dt)
for step in range(n_transient):
    c = imex_step(c, lam_guess, N_fn_guess, dt)
    if step > 0 and step % 50000 == 0:
        print(f"  step {step:>8d}: t = {step*dt:>7.2f}, ‖c‖ = {np.linalg.norm(c):.4f}")

# Record one window and identify the period
n_record = int(2.0 * T_guess_approx / dt)
C_record = np.zeros((n, n_record + 1))
C_record[:, 0] = c
for k in range(1, n_record + 1):
    c = imex_step(c, lam_guess, N_fn_guess, dt)
    C_record[:, k] = c

c0 = C_record[:, 0]
diff = np.linalg.norm(C_record - c0[:, None], axis=0)
k_lo = int(0.9 * T_guess_approx / dt)
k_hi = int(1.1 * T_guess_approx / dt)
k_period = k_lo + int(np.argmin(diff[k_lo:k_hi]))
T_guess = k_period * dt
print(
    f"  approximate period T_guess = {T_guess:.6f}  (k = {k_period}),  "
    f"closure = {diff[k_period]:.4e}"
)
C_guess_open = C_record[:, : k_period]      # endpoint-excluded for FFT


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — HB-Newton with analytic Jacobian + bordered phase condition
# ────────────────────────────────────────────────────────────────────────────
print(f"\nPhase 2: HB-Newton at 1/ν = {1/nu_target:.4f}  (nf = {nf_HB}) ...")

n_harm = 2 * nf_HB + 1                       # # two-sided harmonics
n_t_eval = 4 * nf_HB + 1                     # dealiased grid for N(t)
N_C = n * n_harm                             # complex DOFs for xhat

# Initial guess  (set up `eq` once and just mutate its omega each iteration)
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

Chat = eq.temporal_fft(C_init, nf_HB)        # (n, 2*nf+1)


def _physical_states_on_eval_grid(Chat, omega):
    """Reconstruct x*(t) on the dealiased evaluation grid."""
    eq.omega = omega
    t_eval = np.linspace(0.0, 2.0 * np.pi / omega, n_t_eval, endpoint=False)
    return eq.temporal_ifft(Chat, t_eval).real, t_eval


def compute_residual(Chat, omega):
    """g_k = i k ω xhat_k − L xhat_k − N_k(ifft(xhat)), k = −nf … +nf."""
    x_t, _ = _physical_states_on_eval_grid(Chat, omega)
    N_t = np.zeros_like(x_t)
    for i in range(n_t_eval):
        N_t[:, i] = eq._evaluate_quadratic_term_numpy(x_t[:, i], x_t[:, i])
    Nhat_big = eq.temporal_fft(N_t, 2 * nf_HB)        # k ∈ [−2nf, +2nf]
    Nhat = Nhat_big[:, nf_HB : 3 * nf_HB + 1]         # k ∈ [−nf,  +nf]
    return 1j * k_arr[None, :] * omega * Chat - L_diag[:, None] * Chat - Nhat


def compute_A_p(Chat, omega):
    """A_p = (FFT[A(t)])_p with A(t) = L + 2 B(·, x*(t)), for p = −nf … +nf.

    A(t) is matrix-valued; its Fourier support equals x*(t)'s, namely
    [−nf, +nf].  Higher-|p| coefficients vanish so we only return that range.
    """
    x_t, _ = _physical_states_on_eval_grid(Chat, omega)

    A_t = np.zeros((n_t_eval, n, n))
    for i in range(n_t_eval):
        A_ti = np.diag(L_diag.astype(float))
        x_ti = x_t[:, i]
        for j in range(n):
            e_j = np.zeros(n); e_j[j] = 1.0
            A_ti[:, j] += 2.0 * eq._evaluate_quadratic_term_numpy(e_j, x_ti)
        A_t[i] = A_ti

    A_fft = np.fft.fft(A_t, axis=0) / n_t_eval        # (n_t_eval, n, n)

    A_p = np.zeros((2 * nf_HB + 1, n, n), dtype=complex)
    A_p[nf_HB] = A_fft[0]
    for p in range(1, nf_HB + 1):
        A_p[nf_HB + p] = A_fft[p]
        A_p[nf_HB - p] = A_fft[n_t_eval - p]
    return A_p


def build_bordered_system(Chat, omega):
    """Build the bordered Newton matrix and RHS:

        [ J      | J_ω ] [δxhat] = [-g]
        [ w^*    | 0   ] [δω   ]   [ 0]

    J is block-Toeplitz: J[k_block, j_block] = i k ω δ_kj I − A_{k-j}, where
    A_p includes L at p = 0.  J_ω[k_block] = i k xhat_k.  w is the
    unit-norm orbit-tangent direction in Fourier space.
    """
    A_p = compute_A_p(Chat, omega)
    g = compute_residual(Chat, omega)

    M = np.zeros((N_C + 1, N_C + 1), dtype=complex)
    for ki in range(n_harm):
        k = ki - nf_HB
        # Diagonal: + i k ω I   (the L part comes in via A_0 below)
        for ji in range(n_harm):
            j = ji - nf_HB
            p = k - j
            if abs(p) <= nf_HB:
                M[ki*n:(ki+1)*n, ji*n:(ji+1)*n] = -A_p[nf_HB + p]
        M[ki*n:(ki+1)*n, ki*n:(ki+1)*n] += (1j * k * omega) * np.eye(n)

    # ω column
    M[:N_C, N_C] = (1j * k_arr[None, :] * Chat).flatten(order="F")

    # Phase row:  w^*  with  w = (i k ω xhat) / ‖i k ω xhat‖
    w = (1j * k_arr[None, :] * omega * Chat).flatten(order="F")
    M[N_C, :N_C] = (w / np.linalg.norm(w)).conj()
    M[N_C, N_C] = 0.0

    rhs = np.zeros(N_C + 1, dtype=complex)
    rhs[:N_C] = -g.flatten(order="F")
    rhs[N_C] = 0.0

    return M, rhs, g


print(f"  {'it':>3s}  {'‖g‖':>12s}  {'ω':>16s}  {'T':>12s}")
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

    # Numerically scrub any accumulated asymmetry — conj symmetry should
    # already hold, but enforce it to be safe.
    Chat[:, nf_HB] = Chat[:, nf_HB].real
    for k in range(1, nf_HB + 1):
        avg = 0.5 * (Chat[:, nf_HB + k] + Chat[:, nf_HB - k].conj())
        Chat[:, nf_HB + k] = avg
        Chat[:, nf_HB - k] = avg.conj()
else:
    print(f"  Did not converge in {newton_max_iter} iterations.")


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — reconstruct on save grid and write to disk
# ────────────────────────────────────────────────────────────────────────────
T_sol = 2.0 * np.pi / omega
print(f"\nNewton-refined  T = {T_sol:.12f}")

eq.omega = omega
time_save = np.linspace(0.0, T_sol, n_time_save + 1, endpoint=True)
C_save = eq.temporal_ifft(Chat, time_save).real
closure_abs = float(np.linalg.norm(C_save[:, -1] - C_save[:, 0]))
closure_rel = closure_abs / float(np.linalg.norm(C_save[:, 0]))
print(f"closure (save grid)  = {closure_abs:.4e}  (rel {closure_rel:.4e})")

os.makedirs("data", exist_ok=True)
np.savez(
    "data/periodic_orbit.npz",
    C=C_save,
    T=T_sol,
    nu=nu_target,
    n=n,
    n_pts=n_pts,
    nf=nf_HB,
)
print("Saved refined orbit → data/periodic_orbit.npz")


# ────────────────────────────────────────────────────────────────────────────
# Phase 4 — DNS verification: integrate the orbit through one period with the
# full nonlinear IMEX-3 solver and measure how far ||c_DNS(T) - c(0)|| stays
# from the closure of the HB-Newton orbit.  This is the true test that
# `c(t)` actually satisfies dc/dt = f(c), since the HB system was solved in
# a truncated harmonic subspace and may leak at higher harmonics.
# ────────────────────────────────────────────────────────────────────────────
print("\nPhase 4: DNS verification ...")

# ── Check that (FFT of dc*/dt) = (i k ω xhat_k)_k  lies in null(J) ──
# For an exact orbit, time-translation symmetry implies J · (i k ω xhat) = 0
# in the harmonic-balanced linearisation.  This simultaneously verifies
# (a) the orbit is a true solution of the HB equations, and (b) our
# analytic Jacobian is the correct derivative of the residual.
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
print(
    f"  ‖J · (i k ω xhat)‖ / (‖J‖·‖v‖) = {ker_err:.4e}   "
    f"(machine-zero ≈ exact orbit + correct Jacobian)"
)


dt_dns = 5e-5
n_steps_dns = int(round(T_sol / dt_dns))
dt_dns = T_sol / n_steps_dns                  # exact stepping to t = T_sol
lam_target = linear_eigenvalues(n, nu_target)
N_fn_target = partial(nonlinear_rhs, n_pts=n_pts)
c_dns = C_save[:, 0].copy()
for _ in range(n_steps_dns):
    c_dns = imex_step(c_dns, lam_target, N_fn_target, dt_dns)
err_abs = float(np.linalg.norm(c_dns - C_save[:, 0]))
err_rel = err_abs / float(np.linalg.norm(C_save[:, 0]))
print(
    f"  DNS integration over [0, T={T_sol:.6f}] with "
    f"dt = {dt_dns:.2e}  ({n_steps_dns} steps)"
)
print(f"  ‖c_DNS(T) - c(0)‖     = {err_abs:.4e}  (rel {err_rel:.4e})")

os._exit(0)


# Quick plot
fig, ax = plt.subplots(figsize=(8, 4))
for j in range(min(5, n)):
    ax.plot(time_save, C_save[j], lw=0.8, label=f"$c_{{{j+1}}}$")
ax.set_xlabel("t"); ax.set_ylabel("c_j(t)")
ax.set_title(
    f"HB-Newton refined 1-cycle  (1/ν = {1/nu_target:.4f}, "
    f"T = {T_sol:.6f}, closure rel = {closure_rel:.2e})"
)
ax.legend(fontsize=8, ncol=5, loc="best"); ax.grid(alpha=0.3)
os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/orbit_hb_newton.png", dpi=140, bbox_inches="tight")
print("Saved plot → results/orbit_hb_newton.png")
