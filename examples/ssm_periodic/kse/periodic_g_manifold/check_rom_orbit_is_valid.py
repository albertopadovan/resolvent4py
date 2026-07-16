r"""
Test whether the ROM's saturated periodic orbit is a genuine (though
perhaps unstable) periodic orbit of the full KSE.

Pipeline:
    1. Long ROM simulation → saturate onto the ROM's periodic orbit.
    2. Reconstruct the full state  x(t) = c_*(t) + decode(t, s(t))  over
       the last T_HB window of the tail.
    3. Feed this as the initial guess to HB-Newton on the full KSE at
       fundamental period T_HB (= 2·T_phys).
    4. If Newton converges to a small residual with T ≈ T_HB, the ROM's
       orbit IS a valid KSE orbit at T_HB — likely the unstable partner
       of the stable 2T attractor (whose period T_2T is close to but
       NOT equal to T_HB).
    5. If Newton walks toward T_2T (the DNS-stable 2T orbit), the ROM's
       orbit was not a true KSE orbit — it was the SSM parameterisation
       reproducing the *shape* of the true attractor at the *wrong*
       period.

Reuses the HB-Newton machinery from compute_orbit_hb_newton_2T.py.
"""
# Bootstrap: this script lives in periodic_g_manifold/.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from functools import partial

import numpy as np
import scipy as sp
from scipy.signal import resample
import matplotlib.pyplot as plt
from petsc4py import PETSc

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROMPeriodicG
from spatial_operators import linear_eigenvalues, nonlinear_rhs
from time_integrator import imex_step_3 as imex_step
from kse_differential_equation import KuramotoSivashinskyPeriodic
from scipy.interpolate import interp1d


# ── Load periodic-g SSM cache + build ROM ─────────────────────────────
cache_path = "data/ssm_cache_periodic_g.npz"
cache = np.load(cache_path)
T_HB = float(cache["T"])
T_phys = float(cache["T_orbit_phys"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
nu = float(cache["nu"])
omega_HB = 2.0 * np.pi / T_HB
rho_domain = float(cache["rho_domain"])

rom = SpectralSubmanifoldROMPeriodicG(
    multiindices=cache["multiindices"],
    Lams=cache["Lams"],
    gs=cache["gs_periodic"],
    PS=cache["PS_hb"],
    W=cache["W_hb"],
    omega=omega_HB,
)
rom.use_g_spline = False

C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(t_ext, C_ext, axis=1, kind="cubic",
                          assume_sorted=True)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


# ── Phase 1 — Long ROM integration to saturation ──────────────────────
s0 = np.array([0.1 * rho_domain])          # small IC well inside R_g
n_periods_int = 200                          # long enough to saturate
t_end = n_periods_int * T_phys
n_t = 20000
t_eval = np.linspace(0.0, t_end, n_t)


def rom_rhs_real(t, s):
    return rom.latent_space_dynamics(t, s).real


print(f"Phase 1a: long ROM integration  0 → {n_periods_int} T_phys ...")
sol_rom = sp.integrate.solve_ivp(
    rom_rhs_real, [0.0, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-12,
    dense_output=True,
)
t_rom = sol_rom.t
S_rom = sol_rom.y[0]
print(f"  ROM done.  |s(0)| = {abs(S_rom[0]):.4f},  "
      f"|s(t_end)| = {abs(S_rom[-1]):.4f}   "
      f"(ρ_domain = {rho_domain:.4f})")


# ── Phase 1b — Long FOM integration (same on-manifold IC) ─────────────
lam_kse = linear_eigenvalues(n, nu)
N_fn_kse = partial(nonlinear_rhs, n_pts=n_pts)
x0_fom = c_star_at(0.0) + rom.decode(0.0, s0).real
print(f"\nPhase 1b: long FOM integration  0 → {n_periods_int} T_phys "
      f"(IMEX-3, dt = 1e-3) ...")
dt_fom = 1e-3
n_steps_fom = int(round(t_end / dt_fom))
dt_fom = t_end / n_steps_fom            # exact stepping to t = t_end
n_last = int(round(T_HB / dt_fom))       # steps in the tail T_HB window
Q_fom_tail = np.zeros((n, n_last + 1))   # last-T_HB tail buffer
q = x0_fom.copy()
step_first_saved = n_steps_fom - n_last
for step in range(n_steps_fom + 1):
    if step >= step_first_saved:
        Q_fom_tail[:, step - step_first_saved] = q
    if step < n_steps_fom:
        q = imex_step(q, lam_kse, N_fn_kse, dt_fom)
print(f"  FOM done.  ‖q(t_end)‖ = {np.linalg.norm(q):.4f}")
print(f"  Kept last T_HB window ({n_last + 1} samples at dt = {dt_fom:.2e}).")


# ── Phase 2a — ROM guess from decoded tail ────────────────────────────
nf_HB = 60                                # match compute_orbit_hb_newton_2T.py
n_time_guess = 2 * (2 * nf_HB + 1)       # oversampled for FFT
t_win_start = t_end - T_HB
t_win = np.linspace(t_win_start, t_end, n_time_guess, endpoint=False)

# ROM's s(t) on the window via dense output (interpolated to t_win)
s_win = sol_rom.sol(t_win)[0]

X_guess_rom = np.zeros((n, n_time_guess))
for i, t in enumerate(t_win):
    X_guess_rom[:, i] = c_star_at(t) + rom.decode(t, s_win[i:i+1]).real
print(f"\nPhase 2a: ROM Newton IC = decoded tail  "
      f"(t ∈ [{t_win_start:.3f}, {t_end:.3f}] = last T_HB).")
print(f"  ‖x_ROM(0) - x_ROM(T_HB)‖ = "
      f"{np.linalg.norm(X_guess_rom[:, -1] - X_guess_rom[:, 0]):.4e}  "
      f"(closure of the ROM tail)")


# ── Phase 2b — FOM guess from raw KSE tail ────────────────────────────
# Resample Q_fom_tail (endpoint-excluded) onto the same n_time_guess grid.
X_guess_fom = resample(Q_fom_tail[:, :n_last], n_time_guess, axis=1)
print(f"\nPhase 2b: FOM Newton IC = KSE tail  "
      f"(t ∈ [{t_win_start:.3f}, {t_end:.3f}] = last T_HB).")
print(f"  ‖x_FOM(0) - x_FOM(T_HB)‖ = "
      f"{np.linalg.norm(Q_fom_tail[:, -1] - Q_fom_tail[:, 0]):.4e}  "
      f"(closure of the FOM tail)")


# ── Phase 3 — HB-Newton setup (shared) ────────────────────────────────
n_harm = 2 * nf_HB + 1
n_t_eval = 4 * nf_HB + 1
N_C = n * n_harm
L_diag = linear_eigenvalues(n, nu)
k_arr = np.arange(-nf_HB, nf_HB + 1)

# One eq object — its temporal_fft/ifft and quadratic-term methods don't
# depend on the c_star/time slot, so we can reuse it across both Newton
# calls.  Use the ROM guess to seed it initially.
_C_seed = resample(X_guess_rom, n_harm, axis=1)
_t_seed = np.linspace(0.0, T_HB, n_harm, endpoint=False)
eq = KuramotoSivashinskyPeriodic(
    n=n, nu=nu, nf=nf_HB, nfb=0, c_star=_C_seed,
    time=_t_seed, n_pts=n_pts,
    periodic_diffeq=(2*np.pi/T_HB * np.arange(nf_HB + 1), _t_seed, False),
)


def _phys(Chat, omega):
    eq.omega = omega
    t_e = np.linspace(0.0, 2.0 * np.pi / omega, n_t_eval, endpoint=False)
    return eq.temporal_ifft(Chat, t_e).real, t_e


def residual(Chat, omega):
    x_t, _ = _phys(Chat, omega)
    N_t = np.zeros_like(x_t)
    for i in range(n_t_eval):
        N_t[:, i] = eq._evaluate_quadratic_term_numpy(x_t[:, i], x_t[:, i])
    Nhat_big = eq.temporal_fft(N_t, 2 * nf_HB)
    Nhat = Nhat_big[:, nf_HB : 3 * nf_HB + 1]
    return 1j * k_arr[None, :] * omega * Chat - L_diag[:, None] * Chat - Nhat


def A_p_fn(Chat, omega):
    x_t, _ = _phys(Chat, omega)
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


def bordered_system(Chat, omega):
    A_p = A_p_fn(Chat, omega)
    g = residual(Chat, omega)
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


newton_tol = 1e-12
newton_max_iter = 60


def run_newton(X_guess, tag):
    """HB-Newton starting from X_guess (n, n_time_guess) → (Chat, ω, hists)."""
    omega = 2.0 * np.pi / T_HB
    C_init = resample(X_guess, n_harm, axis=1)
    Chat = eq.temporal_fft(C_init, nf_HB)
    T_hist_l = [T_HB]
    g_hist_l = []
    print(f"\nHB-Newton [{tag}-seeded]  (T_init = T_HB = {T_HB:.6f}) ...")
    print(f"  {'it':>3s}  {'‖g‖':>12s}  {'ω':>16s}  {'T':>12s}  "
          f"{'ΔT/T_HB':>10s}")
    for it in range(newton_max_iter):
        M, rhs, g = bordered_system(Chat, omega)
        g_norm = float(np.linalg.norm(g))
        g_hist_l.append(g_norm)
        T_curr = 2.0 * np.pi / omega
        print(f"  {it:>3d}  {g_norm:.6e}  {omega:.12f}  {T_curr:.10f}  "
              f"{(T_curr - T_HB)/T_HB*1e2:+10.4f} %")
        if g_norm < newton_tol:
            print(f"  [{tag}] Converged.")
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
        T_hist_l.append(2.0 * np.pi / omega)
    else:
        print(f"  [{tag}] Did NOT converge in {newton_max_iter} iterations.")
    return Chat, omega, T_hist_l, g_hist_l


# ── Run Newton twice: ROM-seeded and FOM-seeded ───────────────────────
Chat_rom, omega_rom, T_hist_rom, g_hist_rom = run_newton(X_guess_rom, "ROM")
Chat_fom, omega_fom, T_hist_fom, g_hist_fom = run_newton(X_guess_fom, "FOM")

T_final_rom = 2.0 * np.pi / omega_rom
T_final_fom = 2.0 * np.pi / omega_fom
print(f"\nBoth Newtons finished.")
print(f"  T_ROM-seeded = {T_final_rom:.10f}   "
      f"(ΔT/T_HB = {(T_final_rom - T_HB)/T_HB*1e2:+.4f} %)")
print(f"  T_FOM-seeded = {T_final_fom:.10f}   "
      f"(ΔT/T_HB = {(T_final_fom - T_HB)/T_HB*1e2:+.4f} %)")
print(f"  |T_ROM − T_FOM|  = {abs(T_final_rom - T_final_fom):.4e}")

# For back-compat: expose a single T_final / g_hist (ROM path) so any
# downstream code that reads them still works.
T_final = T_final_rom
g_hist = g_hist_rom
T_hist = T_hist_rom
Chat = Chat_rom
omega = omega_rom


# ── Phase 4 — Reconstruct both refined orbits + compare ──────────────
n_time_save = 1024


def reconstruct(Chat_l, omega_l):
    eq.omega = omega_l
    T_l = 2.0 * np.pi / omega_l
    time_s = np.linspace(0.0, T_l, n_time_save + 1, endpoint=True)
    C_s = eq.temporal_ifft(Chat_l, time_s).real
    closure_abs = float(np.linalg.norm(C_s[:, -1] - C_s[:, 0]))
    closure_rel = closure_abs / float(np.linalg.norm(C_s[:, 0]))
    return C_s, T_l, time_s, closure_abs, closure_rel


C_save_rom, _, time_save_rom, closure_abs_rom, closure_rel_rom = \
    reconstruct(Chat_rom, omega_rom)
C_save_fom, _, time_save_fom, closure_abs_fom, closure_rel_fom = \
    reconstruct(Chat_fom, omega_fom)

print(f"\nOrbit closures:")
print(f"  ROM-seeded:  ‖C(T) − C(0)‖ = {closure_abs_rom:.4e}  "
      f"(rel {closure_rel_rom:.4e})")
print(f"  FOM-seeded:  ‖C(T) − C(0)‖ = {closure_abs_fom:.4e}  "
      f"(rel {closure_rel_fom:.4e})")

# Point-wise distance between the two refined orbits (as functions of
# phase in [0, 1)).  Interpolate FOM to the ROM time grid so this is a
# same-phase-fraction comparison, not a same-t comparison — since the
# periods can differ slightly, same-t doesn't line up.
phase = np.linspace(0.0, 1.0, n_time_save + 1)
C_rom_phase = C_save_rom             # already on phase grid via time_save_rom
# FOM sampled at fractional phases of its own period
C_fom_at_rom_phase = np.zeros_like(C_rom_phase)
for j in range(n):
    C_fom_at_rom_phase[j] = np.interp(
        phase * (2 * np.pi / omega_fom),      # times in FOM's period
        time_save_fom, C_save_fom[j],
    )
dist_orbits = np.linalg.norm(C_rom_phase - C_fom_at_rom_phase, axis=0)
print(f"\n(a) Phase-aligned distance between ROM-seeded and FOM-seeded orbits:")
print(f"  mean = {dist_orbits.mean():.4e},  max = {dist_orbits.max():.4e}")

orbit_2T_path = "data/periodic_orbit_2T.npz"
C_2T_dns = None
if os.path.exists(orbit_2T_path):
    orb2T = np.load(orbit_2T_path)
    T_2T_dns = float(orb2T["T"])
    C_2T_dns = orb2T["C"]
    print(f"\nDNS-stable 2T orbit (compute_orbit_hb_newton_2T.py):  "
          f"T_2T_dns = {T_2T_dns:.10f}")
    for label, T_l in [("ROM-seeded", T_final_rom),
                       ("FOM-seeded", T_final_fom)]:
        dT_vs_HB = abs(T_l - T_HB)
        dT_vs_2T = abs(T_l - T_2T_dns)
        verdict = ("near T_HB" if dT_vs_HB < 0.5 * dT_vs_2T
                   else "near T_2T_dns" if dT_vs_2T < 0.5 * dT_vs_HB
                   else "AMBIGUOUS")
        print(f"  {label}:  T = {T_l:.10f},  "
              f"|ΔT_HB| = {dT_vs_HB:.4e},  "
              f"|ΔT_2T| = {dT_vs_2T:.4e}  →  {verdict}")

os.makedirs("data", exist_ok=True)
np.savez(
    "data/periodic_orbit_2T_from_rom.npz",
    C=C_save_rom, T=T_final_rom, nu=nu, n=n, n_pts=n_pts, nf=nf_HB,
    T_HB_ref=T_HB, T_init=T_HB,
)
np.savez(
    "data/periodic_orbit_2T_from_fom.npz",
    C=C_save_fom, T=T_final_fom, nu=nu, n=n, n_pts=n_pts, nf=nf_HB,
    T_HB_ref=T_HB, T_init=T_HB,
)
print(f"\nSaved  ROM-seeded → data/periodic_orbit_2T_from_rom.npz")
print(f"       FOM-seeded → data/periodic_orbit_2T_from_fom.npz")


# ── Phase 5 — Plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Newton convergence
ax = axes[0]
ax.semilogy(range(len(g_hist_rom)), g_hist_rom, "o-", color="tab:red",
            lw=1.4, label="ROM-seeded")
ax.semilogy(range(len(g_hist_fom)), g_hist_fom, "s-", color="tab:green",
            lw=1.4, label="FOM-seeded")
ax.set_xlabel("Newton iteration", fontsize=12)
ax.set_ylabel(r"$\|g\|$", fontsize=13)
ax.set_title("Newton residual convergence", fontsize=12)
ax.grid(True, which="both", ls=":", lw=0.5)
ax.legend(fontsize=10, loc="best")

# (a_1, a_2)
ax = axes[1]
ax.plot(C_save_rom[0], C_save_rom[1], "-", color="tab:red", lw=2.2,
        alpha=0.9,
        label=fr"ROM-seeded  ($T = {T_final_rom:.4f}$)")
ax.plot(C_save_fom[0], C_save_fom[1], "--", color="tab:green", lw=1.8,
        alpha=0.9,
        label=fr"FOM-seeded  ($T = {T_final_fom:.4f}$)")
if C_2T_dns is not None:
    ax.plot(C_2T_dns[0], C_2T_dns[1], ":", color="tab:blue", lw=1.4,
            alpha=0.85,
            label=fr"DNS-stable 2T  ($T = {T_2T_dns:.4f}$)")
t_1T = np.linspace(0.0, T_phys, 400, endpoint=False)
C_1T = np.array([c_star_at(t) for t in t_1T]).T
ax.plot(C_1T[0], C_1T[1], "-", color="0.55", lw=1.2, alpha=0.7,
        label=r"1T base flow")
ax.set_xlabel(r"$a_1$", fontsize=14)
ax.set_ylabel(r"$a_2$", fontsize=14)
ax.set_aspect("equal", adjustable="datalim")
ax.set_title(r"$(a_1, a_2)$: ROM-seeded vs FOM-seeded",
             fontsize=12)
ax.grid(alpha=0.25)
ax.legend(fontsize=10, loc="best")

fig.tight_layout()
os.makedirs("results", exist_ok=True)
outpath = "results/check_rom_orbit_is_valid.png"
fig.savefig(outpath, dpi=140, bbox_inches="tight")
fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved -> {outpath}")

plt.show()

PETSc.COMM_WORLD.Barrier()
os._exit(0)
