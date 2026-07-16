"""
Harmonic-balance Newton solver for the Rössler T-periodic limit cycle.

Pipeline (mirrors ``kse/compute_orbit_hb_newton.py``):

  Phase 1 — Always time-step the Rössler system at ``c = 5.3`` (well
            inside the simple 1-cycle regime), regardless of the
            target ``c``.  Extract one period via x = 0 upward
            Poincaré crossings + Newton-shoot in T.

  Phase 2 — HB Newton at the user-specified ``c_target`` with the
            analytic block-Toeplitz Jacobian and a bordered
            orthogonal phase condition.

                g_k(qhat, ω) := i k ω qhat_k − M qhat_k
                                − (FFT[B(q*, q*)])_k − b0_k = 0

            Jacobian:

                [D g / D qhat]_{k,j} = i k ω δ_{kj} I − A_{k-j}
                [D g / D ω   ]_k     = i k qhat_k

            where A_p = (FFT[A(t)])_p with A(t) = M + 2·B(q*(t), ·)
            and b0 is the constant Rössler drive (= (0,0,0.1)).

            Phase row: w^* δqhat = 0 with w = (i k ω qhat) /
            ‖i k ω qhat‖ (orbit-tangent direction in Fourier space).

  Phase 3 — Reconstruct on a uniform save grid and write to
            ``data/periodic_orbit.npz``.

  Phase 4 — DNS verification + tangent-in-null-space check.

Pure numpy / scipy — no PETSc, no MPI.

Output: ``data/periodic_orbit.npz`` containing
        ``C, T, c, n, nf``.
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc

from rossler_rhs import (
    rossler_rhs,
    linear_matrix,
    constant_drive,
    quadratic_bilinear,
)


# ── Parameters ─────────────────────────────────────────────────────────────
c_target = 5.6           # target bifurcation parameter (user choice)

# Phase-1 initial guess: always at c = 5.3 (stable 1-cycle).
c_guess = 5.3

# Phase 1 — transient + Poincaré + Newton-shoot
t_transient = 1000.0     # transient to wash out IC
t_detect = 400.0         # window for Poincaré-crossing detection
rtol_ivp = 1e-12
atol_ivp = 1e-12
n_crossings_skip = 4
n_crossings_avg = 32
shoot_iters = 20
shoot_tol = 1e-12

# Phase 2 — HB Newton
nf_HB = 30               # T-periodic harmonics retained
newton_tol = 1e-12
newton_max_iter = 50

# Phase 3 — save
n_time_save = 1024


# ────────────────────────────────────────────────────────────────────────────
# Phase 1 — time-step at c=5.3 to get an initial-guess orbit + period
# ────────────────────────────────────────────────────────────────────────────
print(f"Phase 1: time-stepping at c = {c_guess} for t = {t_transient} ...")
rng = np.random.default_rng(0)
q0 = np.array([1.0, 0.0, 0.0]) + 1e-2 * rng.standard_normal(3)
sol = sp.integrate.solve_ivp(
    rossler_rhs, [0.0, t_transient], q0, args=(c_guess,),
    method="DOP853", rtol=rtol_ivp, atol=atol_ivp,
)
q_settled = sol.y[:, -1]
print(f"  settled state |q| = {np.linalg.norm(q_settled):.4e}")

print(f"\nPhase 1b: detecting period (x = 0 upward crossings) ...")


def crossing_event(t, q, c_):
    return q[0]


crossing_event.terminal = False
crossing_event.direction = 1  # upward only

sol = sp.integrate.solve_ivp(
    rossler_rhs, [0.0, t_detect], q_settled, args=(c_guess,),
    method="DOP853", rtol=rtol_ivp, atol=atol_ivp,
    events=crossing_event,
)
crossing_times = sol.t_events[0]
crossing_states = sol.y_events[0]
if len(crossing_times) < n_crossings_skip + n_crossings_avg + 1:
    raise RuntimeError(
        f"Only {len(crossing_times)} crossings; "
        f"need {n_crossings_skip + n_crossings_avg + 1}."
    )
usable = crossing_times[n_crossings_skip:]
sub_periods = np.diff(usable[: n_crossings_avg + 1])
T_mean = float(np.mean(sub_periods))
T_spread = (sub_periods.max() - sub_periods.min()) / T_mean
print(f"  mean period T = {T_mean:.10f}, rel spread = {T_spread:.2e}")
q_start = crossing_states[n_crossings_skip]
section_val = q_start[0]


print(f"\nPhase 1c: Newton-shoot refinement (still at c = {c_guess}) ...")


def integrate_to(q0_local, T_target, c_param):
    s = sp.integrate.solve_ivp(
        rossler_rhs, [0.0, T_target], q0_local, args=(c_param,),
        method="DOP853", rtol=rtol_ivp, atol=atol_ivp,
    )
    return s.y[:, -1]


T_shoot = T_mean
for it in range(shoot_iters):
    q_T = integrate_to(q_start, T_shoot, c_guess)
    r_shoot = q_T[0] - section_val
    dr_dT = rossler_rhs(T_shoot, q_T, c_guess)[0]
    if abs(dr_dT) < 1e-14:
        print(f"  iter {it}: derivative ~0 — stopping")
        break
    dT = -r_shoot / dr_dT
    T_shoot += dT
    print(f"  iter {it}: T = {T_shoot:.12f}, |res| = {abs(r_shoot):.3e}, dT = {dT:+.3e}")
    if abs(r_shoot) < shoot_tol:
        break

T_guess = T_shoot
print(f"  → initial-guess T = {T_guess:.10f} (at c = {c_guess})")

# Sample one period for the HB initial guess.
n_init = 2 * nf_HB + 1
t_init = np.linspace(0.0, T_guess, n_init, endpoint=False)
sol = sp.integrate.solve_ivp(
    rossler_rhs, [0.0, T_guess], q_start, args=(c_guess,),
    method="DOP853", rtol=rtol_ivp, atol=atol_ivp, t_eval=t_init,
)
C_init = sol.y                                       # (3, n_init)


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — HB Newton at c_target with analytic Jacobian
# ────────────────────────────────────────────────────────────────────────────
print(
    f"\nPhase 2: HB-Newton at c = {c_target} (nf = {nf_HB}) ..."
)

n = 3                                                # Rössler state dim
n_harm = 2 * nf_HB + 1                               # two-sided harmonics
n_t_eval = 4 * nf_HB + 1                             # dealiased grid for N(t)
N_C = n * n_harm                                     # complex DOFs for qhat
k_arr = np.arange(-nf_HB, nf_HB + 1)

# Initial guess for qhat from the c=5.3 orbit.
def temporal_fft(X, nf_out):
    """Forward FFT along last axis → harmonics −nf_out..+nf_out."""
    n_time = X.shape[-1]
    Xhat_pos = np.fft.rfft(X, axis=-1) / n_time      # 0..n_time//2
    Xhat_pos = Xhat_pos[..., : nf_out + 1]
    Xhat_neg = np.conj(Xhat_pos[..., 1:][..., ::-1])
    return np.concatenate([Xhat_neg, Xhat_pos], axis=-1)


def temporal_ifft(Xhat, t_eval, omega):
    """Inverse FFT: sum_k Xhat_k exp(i k ω t_eval)."""
    n_h = Xhat.shape[-1]
    nf_loc = (n_h - 1) // 2
    k = np.arange(-nf_loc, nf_loc + 1)
    E = np.exp(1j * np.outer(t_eval, k * omega))
    return Xhat @ E.T


Qhat = temporal_fft(C_init, nf_HB)                   # (3, n_harm)
omega = 2.0 * np.pi / T_guess

M_mat = linear_matrix(c_target)
b0_vec = constant_drive()

# Pre-compute b0 in HB form: (b0)_k = b0 if k=0 else 0.
b0_hat = np.zeros((n, n_harm), dtype=complex)
b0_hat[:, nf_HB] = b0_vec.astype(complex)


def _physical_states_on_eval_grid(Qhat_in, omega_in):
    t_eval = np.linspace(0.0, 2.0 * np.pi / omega_in, n_t_eval, endpoint=False)
    return temporal_ifft(Qhat_in, t_eval, omega_in).real, t_eval


def compute_residual(Qhat_in, omega_in):
    """g_k = i k ω qhat_k − M qhat_k − (FFT[B(q*, q*)])_k − b0_k."""
    q_t, _ = _physical_states_on_eval_grid(Qhat_in, omega_in)
    # B(q*, q*) at each time
    Bqq_t = np.zeros_like(q_t)
    for i in range(n_t_eval):
        Bqq_t[:, i] = quadratic_bilinear(q_t[:, i], q_t[:, i])
    Bhat_big = temporal_fft(Bqq_t, 2 * nf_HB)
    Bhat = Bhat_big[:, nf_HB : 3 * nf_HB + 1]
    return (
        1j * k_arr[None, :] * omega_in * Qhat_in
        - M_mat @ Qhat_in
        - Bhat
        - b0_hat
    )


def compute_A_p(Qhat_in, omega_in):
    """A_p = (FFT[A(t)])_p with A(t) = M + 2·B(q*(t), ·)."""
    q_t, _ = _physical_states_on_eval_grid(Qhat_in, omega_in)
    A_t = np.zeros((n_t_eval, n, n))
    for i in range(n_t_eval):
        A_ti = M_mat.copy()
        q_ti = q_t[:, i]
        for j_col in range(n):
            e_j = np.zeros(n); e_j[j_col] = 1.0
            A_ti[:, j_col] += 2.0 * quadratic_bilinear(e_j, q_ti)
        A_t[i] = A_ti

    A_fft = np.fft.fft(A_t, axis=0) / n_t_eval        # (n_t_eval, n, n)
    A_p = np.zeros((2 * nf_HB + 1, n, n), dtype=complex)
    A_p[nf_HB] = A_fft[0]
    for p in range(1, nf_HB + 1):
        A_p[nf_HB + p] = A_fft[p]
        A_p[nf_HB - p] = A_fft[n_t_eval - p]
    return A_p


def build_bordered_system(Qhat_in, omega_in):
    """[ J | J_ω ] [δqhat; δω] = [-g; 0] with phase row w^* δqhat = 0."""
    A_p = compute_A_p(Qhat_in, omega_in)
    g = compute_residual(Qhat_in, omega_in)

    M_full = np.zeros((N_C + 1, N_C + 1), dtype=complex)
    for ki in range(n_harm):
        k = ki - nf_HB
        for ji in range(n_harm):
            j = ji - nf_HB
            p = k - j
            if abs(p) <= nf_HB:
                M_full[ki*n:(ki+1)*n, ji*n:(ji+1)*n] = -A_p[nf_HB + p]
        M_full[ki*n:(ki+1)*n, ki*n:(ki+1)*n] += (1j * k * omega_in) * np.eye(n)

    # ω column
    M_full[:N_C, N_C] = (1j * k_arr[None, :] * Qhat_in).flatten(order="F")

    # Phase row
    w = (1j * k_arr[None, :] * omega_in * Qhat_in).flatten(order="F")
    M_full[N_C, :N_C] = (w / np.linalg.norm(w)).conj()
    M_full[N_C, N_C] = 0.0

    rhs = np.zeros(N_C + 1, dtype=complex)
    rhs[:N_C] = -g.flatten(order="F")
    rhs[N_C] = 0.0
    return M_full, rhs, g


print(f"  {'it':>3s}  {'‖g‖':>12s}  {'ω':>16s}  {'T':>12s}")
for it in range(newton_max_iter):
    M_sys, rhs_sys, g_curr = build_bordered_system(Qhat, omega)
    g_norm = np.linalg.norm(g_curr)
    T_curr = 2.0 * np.pi / omega
    print(f"  {it:>3d}  {g_norm:.6e}  {omega:.12f}  {T_curr:.10f}")
    if g_norm < newton_tol:
        print("  Converged.")
        break
    dx = np.linalg.solve(M_sys, rhs_sys)
    dQhat = dx[:N_C].reshape((n, n_harm), order="F")
    domega = dx[N_C].real
    Qhat = Qhat + dQhat
    omega = omega + domega

    # Re-enforce conj-symmetry (defensive)
    Qhat[:, nf_HB] = Qhat[:, nf_HB].real
    for k in range(1, nf_HB + 1):
        avg = 0.5 * (Qhat[:, nf_HB + k] + Qhat[:, nf_HB - k].conj())
        Qhat[:, nf_HB + k] = avg
        Qhat[:, nf_HB - k] = avg.conj()
else:
    print(f"  Did not converge in {newton_max_iter} iters.")


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — reconstruct and save
# ────────────────────────────────────────────────────────────────────────────
T_sol = 2.0 * np.pi / omega
print(f"\nNewton-refined T = {T_sol:.12f}  (at c = {c_target})")

t_save = np.linspace(0.0, T_sol, n_time_save + 1, endpoint=True)
C_save = temporal_ifft(Qhat, t_save, omega).real     # (3, n_time_save+1)
closure_abs = float(np.linalg.norm(C_save[:, -1] - C_save[:, 0]))
closure_rel = closure_abs / float(np.linalg.norm(C_save[:, 0]))
print(f"  closure (save grid) = {closure_abs:.4e}  (rel {closure_rel:.4e})")

os.makedirs("data", exist_ok=True)
np.savez(
    "data/periodic_orbit.npz",
    C=C_save,                  # (3, n_time_save+1)
    t=t_save,                  # save grid
    T=T_sol,
    c=c_target,
    n=n,
    nf=nf_HB,
)
print("Saved → data/periodic_orbit.npz")


# ────────────────────────────────────────────────────────────────────────────
# Phase 4 — verifications
# ────────────────────────────────────────────────────────────────────────────
print("\nPhase 4: verifications ...")

# (a) DNS over one period with high-accuracy DOP853
sol_dns = sp.integrate.solve_ivp(
    rossler_rhs, [0.0, T_sol], C_save[:, 0], args=(c_target,),
    method="DOP853", rtol=1e-12, atol=1e-12,
)
err_abs = float(np.linalg.norm(sol_dns.y[:, -1] - C_save[:, 0]))
err_rel = err_abs / float(np.linalg.norm(C_save[:, 0]))
print(
    f"  DNS over [0, T={T_sol:.6f}]:  "
    f"‖c_DNS(T) − c(0)‖ = {err_abs:.4e}  (rel {err_rel:.4e})"
)

# (b) Tangent-in-null-space: J · (i k ω qhat) ≈ 0  iff Jacobian + orbit are exact
A_p_check = compute_A_p(Qhat, omega)
J_only = np.zeros((N_C, N_C), dtype=complex)
for ki in range(n_harm):
    k = ki - nf_HB
    for ji in range(n_harm):
        j = ji - nf_HB
        p = k - j
        if abs(p) <= nf_HB:
            J_only[ki*n:(ki+1)*n, ji*n:(ji+1)*n] = -A_p_check[nf_HB + p]
    J_only[ki*n:(ki+1)*n, ki*n:(ki+1)*n] += (1j * k * omega) * np.eye(n)
v_tangent = (1j * k_arr[None, :] * omega * Qhat).flatten(order="F")
Jv = J_only @ v_tangent
ker_err = np.linalg.norm(Jv) / (np.linalg.norm(J_only) * np.linalg.norm(v_tangent))
print(
    f"  ‖J · (i k ω qhat)‖ / (‖J‖·‖v‖) = {ker_err:.4e}   "
    f"(machine-zero ≈ exact orbit + correct Jacobian)"
)


# ────────────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# Time series
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
for ax, comp, lab in zip(axes, C_save, ["x", "y", "z"]):
    ax.plot(t_save, comp, "k", lw=1)
    ax.set_ylabel(lab)
axes[-1].set_xlabel("t")
axes[0].set_title(
    rf"Rössler HB-Newton orbit (c={c_target}, T={T_sol:.6f}, "
    rf"nf={nf_HB}, closure rel = {closure_rel:.2e})"
)
fig.tight_layout()
fig.savefig("results/orbit_hb_newton.png", dpi=140, bbox_inches="tight")
fig.savefig("results/orbit_hb_newton.pdf", bbox_inches="tight")
print("Saved plot → results/orbit_hb_newton.{png,pdf}")

# 3-D phase portrait
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(C_save[0], C_save[1], C_save[2], "k", lw=1)
ax.scatter(*C_save.mean(axis=1), color="C3", s=40, label="temporal mean")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend()
ax.set_title(f"Rössler orbit (c={c_target}, T={T_sol:.4f})")
fig.tight_layout()
fig.savefig("results/orbit_phase_portrait.png", dpi=140, bbox_inches="tight")
fig.savefig("results/orbit_phase_portrait.pdf", bbox_inches="tight")
print("Saved plot → results/orbit_phase_portrait.{png,pdf}")

# Spectrum
amp = np.max(np.abs(Qhat), axis=0)                   # (n_harm,)
k_show = np.arange(0, nf_HB + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(k_show, amp[nf_HB:], "ko", mfc="none")
ax.set_xlabel("harmonic index k")
ax.set_ylabel(r"$\max_j |\hat{q}_j(k)|$")
ax.set_title(f"Rössler orbit amplitude spectrum (c={c_target})")
ax.grid(True, which="both", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig("results/orbit_spectrum.png", dpi=140, bbox_inches="tight")
fig.savefig("results/orbit_spectrum.pdf", bbox_inches="tight")
print("Saved plot → results/orbit_spectrum.{png,pdf}")

plt.show()


PETSc.COMM_WORLD.Barrier()
os._exit(0)
