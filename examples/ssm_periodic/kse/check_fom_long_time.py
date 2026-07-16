"""
Long-time on-manifold FOM: does it settle on the ROM's saturated
2T-periodic orbit (decode(t, s*), period T_HB = 2·T_phys), or on the
HB Newton 2T orbit c_2T(t) (period T_2T ≠ T_HB)?

Start FOM exactly on the SSM's saturated state:
    x_0 = c_*(0) + decode(0, s*)                (on-manifold)
integrate to t = 100 T_phys with RK45, and compare the tail against
both candidate attractors via phase-optimal distance.  Also fit the
FOM's observed period from a Poincaré-section-style zero-crossing
count of one mode.
"""
import os
from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir, C_FOM, C_ROM,
)
from spatial_operators import linear_eigenvalues, nonlinear_rhs


setup_matplotlib()
ctx = build_context()
rom = ctx["rom"]
c_star_at = ctx["c_star_at"]
s_star = ctx["s_star"]
T_HB = ctx["T_HB"]
T_phys = ctx["T_phys"]
n = ctx["n"]
n_pts = ctx["n_pts"]
nu = ctx["nu"]
Lam0 = complex(ctx["cache"]["Lams"][0])

print(f"|Λ|  = {abs(Lam0.real):.4e},  T_phys = {T_phys:.4f},  "
      f"T_HB = {T_HB:.6f} (= 2·T_phys)")
print(f"s* = {s_star:.4f}  (SSM saturated latent)")


lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)


# ── HB-Newton 2T orbit reference ──────────────────────────────────────
orb = np.load("data/periodic_orbit_2T.npz")
C_2T = orb["C"]
T_2T = float(orb["T"])
n_time_2T = C_2T.shape[1]
t_2T = np.linspace(0.0, T_2T, n_time_2T, endpoint=False)
print(f"T_2T = {T_2T:.6f}   ({T_2T / T_phys:.5f} · T_phys,   "
      f"mismatch vs T_HB = {(T_2T - T_HB) / T_HB * 1e6:.1f} ppm)")


# ── IC exactly on the SSM's saturated state ───────────────────────────
s0 = np.array([s_star], dtype=complex)
delta_ssm_sat_at_0 = rom.decode(0.0, s0).real
x0 = c_star_at(0.0) + delta_ssm_sat_at_0
print(f"IC: x_0 = c_*(0) + decode(0, s*),  "
      f"||decode(0, s*)|| = {np.linalg.norm(delta_ssm_sat_at_0):.4f}")


# ── Integrate FOM to 200 T_phys ───────────────────────────────────────
t_end = 200.0 * T_phys
n_t = 40000
rtol, atol = 1e-9, 1e-11        # loosened from 1e-11 for the long horizon
t_eval = np.linspace(0.0, t_end, n_t)


def _kse_rhs(_t, q):
    return lam * q + N_fn(q)


print(f"\nFOM integration  0 → {t_end / T_phys:.1f} T_phys "
      f"(RK45, rtol={rtol}, atol={atol}) ...")
sol = sp.integrate.solve_ivp(
    _kse_rhs, [0.0, t_end], x0,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
)
X_fom = sol.y                                    # (n, n_t)
print(f"  nfev = {sol.nfev}")


# ── ROM saturated: decode(t, s*) at every t_eval ──────────────────────
X_rom_sat = np.zeros((n, n_t))
for k in range(n_t):
    X_rom_sat[:, k] = c_star_at(t_eval[k]) + rom.decode(t_eval[k], s0).real


# ── Cyclic 2T reference at every t_eval ──────────────────────────────
def c_2T_at(times):
    t_ext = np.append(t_2T, T_2T)
    out = np.zeros((n, len(times)))
    tm = np.mod(times, T_2T)
    for j in range(n):
        y_ext = np.append(C_2T[j], C_2T[j, 0])
        out[j] = np.interp(tm, t_ext, y_ext)
    return out


X_c2T = c_2T_at(t_eval)


# ── Distances ─────────────────────────────────────────────────────────
# (a) FOM vs ROM saturated  — at MATCHING times (both known at t_eval).
#     If the true attractor's period is T_HB (i.e., matches decode(·, s*)),
#     this distance decays to zero.
# (b) FOM vs c_2T           — phase-optimal min over τ.
#     If the true attractor's period is T_2T, this decays to zero.
def phase_optimal_dist(X_a, X_b_family, n_tau=200, T_period=None):
    """||X_a[t_k] − X_b(τ_k)||, min over τ ∈ [0, T_period)."""
    tau_grid = np.linspace(0.0, T_period, n_tau, endpoint=False)
    C_grid = X_b_family(tau_grid)
    dists = np.zeros(X_a.shape[1])
    for k in range(X_a.shape[1]):
        diff = X_a[:, k:k+1] - C_grid
        dists[k] = float(np.linalg.norm(diff, axis=0).min())
    return dists


def rom_sat_at(times):
    out = np.zeros((n, len(times)))
    for j, t in enumerate(times):
        out[:, j] = c_star_at(t) + rom.decode(t, s0).real
    return out


ref_norm_rom = float(np.linalg.norm(rom_sat_at(np.linspace(0.0, T_HB, 200)),
                                     axis=0).mean())
ref_norm_c2T = float(np.linalg.norm(c_2T_at(np.linspace(0.0, T_2T, 200)),
                                     axis=0).mean())

# (a) FOM vs ROM sat — SAME-TIME comparison (no phase optimisation) since
#     if they share T_HB they should stay in lockstep.  Also do phase-
#     optimal to see if a fixed phase shift closes the gap.
d_same_time_rom = np.linalg.norm(X_fom - X_rom_sat, axis=0)
d_phase_opt_rom = phase_optimal_dist(X_fom, rom_sat_at,
                                      n_tau=200, T_period=T_HB)

# (b) FOM vs c_2T — phase-optimal since T_2T ≠ T_HB means same-time is
#     always going to look drift-y.
d_phase_opt_c2T = phase_optimal_dist(X_fom, c_2T_at,
                                      n_tau=200, T_period=T_2T)


# ── Coarse trace ──────────────────────────────────────────────────────
print("\nlog₁₀( d / ||orbit||_mean )  vs  t/T_phys :")
print(f"  {'t/T':>6s}   "
      f"{'|FOM−ROM_sat|_t':>16s}  "
      f"{'|FOM−ROM_sat|_τ*':>16s}  "
      f"{'|FOM−c_2T|_τ*':>15s}")
sample_ts = np.linspace(0.0, t_end / T_phys, 21)
for tk in sample_ts:
    idx = int(np.clip(round(tk / (t_end / T_phys) * (n_t - 1)), 0, n_t - 1))
    def L(x, ref):
        return np.log10(max(x / ref, 1e-30))
    print(f"  {tk:>6.2f}   "
          f"{L(d_same_time_rom[idx], ref_norm_rom):>16.3f}  "
          f"{L(d_phase_opt_rom[idx], ref_norm_rom):>16.3f}  "
          f"{L(d_phase_opt_c2T[idx], ref_norm_c2T):>15.3f}")


# ── Late-window summary ───────────────────────────────────────────────
mask_late = (t_eval / T_phys >= 180.0)
print(f"\nLate window t/T ∈ [180, 200]:")
print(f"  ||FOM − ROM_sat|| / ||ROM_sat||_mean  "
      f"same-time :  mean = {d_same_time_rom[mask_late].mean() / ref_norm_rom:.4e}, "
      f"max = {d_same_time_rom[mask_late].max() / ref_norm_rom:.4e}")
print(f"  ||FOM − ROM_sat|| / ||ROM_sat||_mean  "
      f"phase-opt :  mean = {d_phase_opt_rom[mask_late].mean() / ref_norm_rom:.4e}, "
      f"max = {d_phase_opt_rom[mask_late].max() / ref_norm_rom:.4e}")
print(f"  ||FOM − c_2T   || / ||c_2T   ||_mean  "
      f"phase-opt :  mean = {d_phase_opt_c2T[mask_late].mean() / ref_norm_c2T:.4e}, "
      f"max = {d_phase_opt_c2T[mask_late].max() / ref_norm_c2T:.4e}")


# ── FOM period from zero-crossing count ──────────────────────────────
# Count upcrossings of a_1(t) − <a_1> = 0 in the late window and infer
# an effective period.
a1 = X_fom[0, mask_late] - X_fom[0, mask_late].mean()
tm = t_eval[mask_late]
zc = np.where((a1[:-1] < 0) & (a1[1:] >= 0))[0]
if len(zc) >= 2:
    T_fom_eff = np.mean(np.diff(tm[zc]))
    print(f"\nFOM effective period from a_1 upcrossings:  "
          f"T_FOM ≈ {T_fom_eff:.6f}")
    print(f"    T_FOM / T_HB  = {T_fom_eff / T_HB:.6f}")
    print(f"    T_FOM / T_2T  = {T_fom_eff / T_2T:.6f}")
    print(f"    (ideal 1.0 for whichever period the FOM has locked to)")


# ── Plots ─────────────────────────────────────────────────────────────
outdir = ensure_outdir()
C_C2T = "0.5"
C_SAT = "tab:orange"

# Panel 1: log distances vs t (both metrics)
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.semilogy(t_eval / T_phys, d_same_time_rom / ref_norm_rom,
            "-", color=C_ROM, lw=1.2, label=r"$\|x_{\rm FOM}(t) - x_{\rm SSM}(t)\|$ same-$t$")
ax.semilogy(t_eval / T_phys, d_phase_opt_rom / ref_norm_rom,
            "--", color=C_ROM, lw=1.2, alpha=0.6,
            label=r"same $\downarrow$ phase-opt over $T_{HB}$")
ax.semilogy(t_eval / T_phys, d_phase_opt_c2T / ref_norm_c2T,
            "-", color=C_C2T, lw=1.6,
            label=r"$\|x_{\rm FOM}(t) - c_{2T}(\tau^*)\|$ phase-opt over $T_{2T}$")
ax.set_xlabel(r"$t / T_{\rm phys}$", fontsize=14)
ax.set_ylabel(r"$d(t) / \|\text{ref}\|_{\rm mean}$", fontsize=14)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10, loc="best")
ax.set_title("Long-time FOM vs the two candidate attractors  "
             r"(SSM saturation vs HB $c_{2T}$)", fontsize=12)
fig.tight_layout()
outpath1 = os.path.join(outdir, "check_fom_long_time_distances")
fig.savefig(outpath1 + ".png", bbox_inches="tight", pad_inches=0.3, dpi=140)
fig.savefig(outpath1 + ".pdf", bbox_inches="tight", pad_inches=0.3)
print(f"\nSaved -> {outpath1}.png/.pdf")


# Panel 2: phase-space (a_1, a_2) in late window with both references
mask_last = (t_eval / T_phys >= 190.0)     # last 10 T_phys
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ix, iy) in zip(axes, [(0, 1), (2, 3)]):
    ax.plot(np.append(C_2T[ix], C_2T[ix, 0]),
            np.append(C_2T[iy], C_2T[iy, 0]),
            "-", color=C_C2T, lw=3.0, alpha=0.7, label=r"$c_{2T}$ (HB)")
    # Decode(t, s*) sampled over T_HB
    tt = np.linspace(0.0, T_HB, 300)
    Xsat = np.zeros((n, 300))
    for j, t in enumerate(tt):
        Xsat[:, j] = c_star_at(t) + rom.decode(t, s0).real
    ax.plot(np.append(Xsat[ix], Xsat[ix, 0]),
            np.append(Xsat[iy], Xsat[iy, 0]),
            "-", color=C_SAT, lw=2.5, alpha=0.9, label=r"decode$(t, s_*)$")
    ax.plot(X_fom[ix, mask_last], X_fom[iy, mask_last],
            "-", color=C_FOM, lw=1.0, alpha=0.9,
            label=r"FOM tail ($t/T_{\rm phys} \in [190, 200]$)")
    ax.set_xlabel(rf"$a_{{{ix+1}}}$", fontsize=14)
    ax.set_ylabel(rf"$a_{{{iy+1}}}$", fontsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=10, loc="best")
fig.suptitle(r"Long-time FOM in phase space vs two candidate 2T attractors",
             fontsize=13)
fig.tight_layout()
outpath2 = os.path.join(outdir, "check_fom_long_time_phase")
fig.savefig(outpath2 + ".png", bbox_inches="tight", pad_inches=0.3, dpi=140)
fig.savefig(outpath2 + ".pdf", bbox_inches="tight", pad_inches=0.3)
print(f"Saved -> {outpath2}.png/.pdf")


print("\nVerdict guide:")
print("  · If  |FOM − ROM_sat|  goes to ~0 and  |FOM − c_2T|  stays ~0.01+,")
print("     the FOM has locked onto the SSM's saturated 2T orbit at T_HB.")
print("     The HB Newton c_2T disagreed because HB Newton itself did not")
print("     converge cleanly to the true attractor.")
print("  · If  |FOM − c_2T|  goes to ~0 and  |FOM − ROM_sat|  stays ~0.01+,")
print("     the FOM has locked onto c_2T (period T_2T), and the SSM's")
print("     saturated orbit is an approximate — but not exact — parameterisation.")
print("  · If  both  stay small,  T_2T ≈ T_HB to numerical precision and")
print("     there is no meaningful distinction.")
print("  · If  neither goes to zero,  the FOM's true attractor is neither.")

plt.show(block=True)

PETSc.COMM_WORLD.Barrier()
os._exit(0)
