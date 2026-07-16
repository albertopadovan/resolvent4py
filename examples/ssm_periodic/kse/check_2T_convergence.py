"""
Check whether the FOM and SSM ROM have reached the 2T-periodic orbit
by t/T_phys ≈ 12.  Uses the same on- and off-manifold ICs as
fig4_timeseries.py, integrates to t = 15 T_phys, and compares the tail
against the HB Newton 2T orbit stored in ``data/periodic_orbit_2T.npz``.

Two figures (one per IC):
    check_2T_conv_on   IC exactly on the SSM.
    check_2T_conv_off  IC with the transverse kick used in fig4 (0.1 in
                       modes 1..4); ROM re-encodes via w(0)ᵀ Δ.

Each figure shows, over t/T_phys ∈ [12, 15]:
    · time series a_j(t) for j = 1..4, FOM (blue), ROM (red dashed),
      and c_2T periodically extended (grey) for reference,
    · (a_1, a_2) and (a_3, a_4) phase portraits with the 2T orbit as a
      closed curve.

If the tail curves overlap the reference orbit in every panel, the
system has landed on the 2T attractor.
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

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)
Lam0 = complex(ctx["cache"]["Lams"][0])   # master Floquet exponent at s=0


# ── 2T HB reference orbit ─────────────────────────────────────────────
orb = np.load("data/periodic_orbit_2T.npz")
C_2T = orb["C"]                              # (n, n_time_2T)
T_2T = float(orb["T"])                       # ≈ 2 T_phys but not exact
n_time_2T = C_2T.shape[1]
t_2T = np.linspace(0.0, T_2T, n_time_2T, endpoint=False)
print(f"2T HB orbit:  T_2T = {T_2T:.6f}  "
      f"(= {T_2T / T_phys:.5f} · T_phys,  vs T_HB = "
      f"{T_HB:.6f} = 2·T_phys),  {n_time_2T} samples")


# ── Integration parameters ────────────────────────────────────────────
# Both ROM and FOM use RK45 with matched tolerances so any observed
# rate mismatch is physics, not integrator.
t_end = 15.0 * T_phys
n_t = 6000
rtol, atol = 1e-11, 1e-12
t_eval = np.linspace(0.0, t_end, n_t)


def _kse_rhs(_t, q):
    return lam * q + N_fn(q)


def _integrate_rom(s0):
    S = sp.integrate.solve_ivp(
        rom.latent_space_dynamics, [0.0, t_end], s0,
        method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    ).y
    X = np.zeros((n, n_t))
    for i in range(n_t):
        X[:, i] = c_star_at(t_eval[i]) + rom.decode(t_eval[i], S[:, i]).real
    return X


def _integrate_fom(x0):
    return sp.integrate.solve_ivp(
        _kse_rhs, [0.0, t_end], x0,
        method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    ).y


# ── ICs (same conventions as fig4_timeseries.py) ──────────────────────
s0_on = 0.1 * np.array([s_star], dtype=complex)
delta_ssm_0 = rom.decode(0.0, s0_on).real
x0_on = c_star_at(0.0) + delta_ssm_0

transverse = np.zeros(n)
transverse[0:4] = 1.0
transverse = transverse / np.linalg.norm(transverse)
x0_off = x0_on + 0.1 * transverse
s0_off = rom.encode(0.0, x0_off - c_star_at(0.0))
print(f"On-manifold  s0 = {s0_on[0]:+.4f}")
print(f"Off-manifold s0 = {s0_off[0]:+.6f}  (encode of x0_off − c_*(0))")


# ── Integrate everything ──────────────────────────────────────────────
print("\nROM integration (on) ...");  X_rom_on  = _integrate_rom(s0_on)
print("ROM integration (off) ..."); X_rom_off = _integrate_rom(s0_off)
print("FOM integration (on) ...");  X_fom_on  = _integrate_fom(x0_on)
print("FOM integration (off) ..."); X_fom_off = _integrate_fom(x0_off)


# ── Along-SSM diagnostic: |s(t) − s*| ─────────────────────────────────
# The right question for "is the ROM saturating faster than the FOM?"
# is not about distance to c_2T (which is contaminated by the
# T_2T ≠ T_HB period mismatch — trace floor sits at ~10^-2 regardless of
# convergence).  Instead compare the along-SSM latent:
#   ROM: s_ROM(t) = encode(t, X_ROM(t) − c_*(t))
#   FOM: s_FOM(t) = encode(t, X_FOM(t) − c_*(t))
# For real Λ and r=1 the FOM projection is real; take Re() to strip
# double-precision imag noise.  |s − s*| is cleanly exponential
# (growth at Re(Λ) then decay at g'(s*)).
def encode_series(X):
    s_trace = np.zeros(n_t, dtype=complex)
    for k in range(n_t):
        s_trace[k] = rom.encode(t_eval[k], X[:, k] - c_star_at(t_eval[k]))[0]
    return s_trace.real


print("\nEncoding FOM/ROM state onto s(t) via w(t)ᵀ(x − c_*) ...")
s_rom_on_t  = encode_series(X_rom_on)
s_rom_off_t = encode_series(X_rom_off)
s_fom_on_t  = encode_series(X_fom_on)
s_fom_off_t = encode_series(X_fom_off)
s_star_real = float(np.real(s_star))
d_s_rom_on  = np.abs(s_rom_on_t  - s_star_real)
d_s_rom_off = np.abs(s_rom_off_t - s_star_real)
d_s_fom_on  = np.abs(s_fom_on_t  - s_star_real)
d_s_fom_off = np.abs(s_fom_off_t - s_star_real)


print(f"\nlog₁₀ |s(t) − s*|  vs  t/T_phys   "
      f"(s* = {s_star_real:.4f}):")
print(f"  {'t/T':>5s}  {'FOM_on':>8s}  {'ROM_on':>8s}  "
      f"{'FOM_off':>8s}  {'ROM_off':>8s}")
for tk in np.linspace(0.0, t_end / T_phys, 16):
    idx = int(np.clip(round(tk * T_phys / t_end * (n_t - 1)), 0, n_t - 1))
    def _safelog(x):
        return np.log10(max(x, 1e-30))
    print(f"  {tk:>5.2f}  "
          f"{_safelog(d_s_fom_on[idx]):>8.3f}  "
          f"{_safelog(d_s_rom_on[idx]):>8.3f}  "
          f"{_safelog(d_s_fom_off[idx]):>8.3f}  "
          f"{_safelog(d_s_rom_off[idx]):>8.3f}")


def fit_s_rate(d_s, tag, win):
    m = (t_eval / T_phys >= win[0]) & (t_eval / T_phys <= win[1])
    if m.sum() < 5:
        print(f"  [{tag}] window empty");  return None
    y = d_s[m]
    floor = max(d_s.min() * 3.0, 1e-30)
    m2 = y > floor
    if m2.sum() < 5:
        print(f"  [{tag}] at floor");  return None
    slope, _ = np.polyfit(t_eval[m][m2], np.log(y[m2]), 1)
    print(f"  [{tag}] b = {slope:+.4e}   "
          f"({'growth' if slope > 0 else 'decay'} time = "
          f"{abs(1.0/slope):.3f})")
    return slope


# Pick these AFTER eyeballing the trace above.  Sensible defaults:
# growth = early [1, 6], saturation = latest bracket before floor.
fit_growth_s = (1.0, 6.0)
fit_sat_s    = (10.0, 13.0)

print(f"\n|s − s*| GROWTH-phase fit, t/T ∈ {fit_growth_s}:")
fit_s_rate(d_s_fom_on,  "FOM on ", fit_growth_s)
fit_s_rate(d_s_rom_on,  "ROM on ", fit_growth_s)
fit_s_rate(d_s_fom_off, "FOM off", fit_growth_s)
fit_s_rate(d_s_rom_off, "ROM off", fit_growth_s)

print(f"\n|s − s*| SATURATION-phase fit, t/T ∈ {fit_sat_s}:")
fit_s_rate(d_s_fom_on,  "FOM on ", fit_sat_s)
fit_s_rate(d_s_rom_on,  "ROM on ", fit_sat_s)
fit_s_rate(d_s_fom_off, "FOM off", fit_sat_s)
fit_s_rate(d_s_rom_off, "ROM off", fit_sat_s)


# ── Late-window mask ──────────────────────────────────────────────────
t_lo, t_hi = 12.0, 15.0
mask = (t_eval / T_phys >= t_lo) & (t_eval / T_phys <= t_hi)
n_mask = int(mask.sum())
print(f"\nLate window t/T ∈ [{t_lo}, {t_hi}]: {n_mask} samples")


# ── 2T orbit periodically extended over the late window ──────────────
# Interpolate C_2T onto t_eval[mask] via (t mod T_2T)
def c_2T_at(times):
    """Cyclic interpolation of the HB 2T orbit onto arbitrary times."""
    t_mod = np.mod(times, T_2T)
    out = np.zeros((n, len(times)))
    # Loop over modes; use np.interp with periodic wrap by appending
    t_ext = np.append(t_2T, T_2T)
    for j in range(n):
        y_ext = np.append(C_2T[j], C_2T[j, 0])
        out[j] = np.interp(t_mod, t_ext, y_ext)
    return out


C_2T_late = c_2T_at(t_eval[mask])            # (n, n_mask) — reference over the tail


# ── Render ─────────────────────────────────────────────────────────────
outdir = ensure_outdir()
C_REF = "0.4"


def _render(X_fom, X_rom, tag_ic, title_ic):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.4],
                          hspace=0.35, wspace=0.25)

    # Row 1-2: time series a_j(t) for j = 1, 2 (top) and j = 3, 4 (row 2)
    for row, (jL, jR) in enumerate([(0, 1), (2, 3)]):
        for col, j in enumerate([jL, jR]):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t_eval[mask] / T_phys, C_2T_late[j], "-",
                    color=C_REF, lw=3.0, alpha=0.6, label=r"$c_{2T}$ (HB)")
            ax.plot(t_eval[mask] / T_phys, X_fom[j, mask], "-",
                    color=C_FOM, lw=1.4, label="FOM")
            ax.plot(t_eval[mask] / T_phys, X_rom[j, mask], "--",
                    color=C_ROM, lw=1.4, label="ROM")
            ax.set_ylabel(rf"$a_{{{j+1}}}(t)$", fontsize=15)
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel(r"$t / T_{\rm phys}$", fontsize=13)
            if row == 0 and col == 0:
                ax.legend(fontsize=10, ncol=3, loc="upper right")

    # Row 3: phase portraits (a_1, a_2) and (a_3, a_4)
    for col, (ix, iy) in enumerate([(0, 1), (2, 3)]):
        ax = fig.add_subplot(gs[2, col])
        # 2T orbit as closed curve (full period, not just late window)
        ax.plot(np.append(C_2T[ix], C_2T[ix, 0]),
                np.append(C_2T[iy], C_2T[iy, 0]),
                "-", color=C_REF, lw=3.0, alpha=0.6,
                label=r"$c_{2T}$ (HB)")
        ax.plot(X_fom[ix, mask], X_fom[iy, mask], "-",
                color=C_FOM, lw=1.4, label="FOM tail")
        ax.plot(X_rom[ix, mask], X_rom[iy, mask], "--",
                color=C_ROM, lw=1.4, label="ROM tail")
        ax.set_xlabel(rf"$a_{{{ix+1}}}$", fontsize=14)
        ax.set_ylabel(rf"$a_{{{iy+1}}}$", fontsize=14)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        if col == 0:
            ax.legend(fontsize=10, loc="best")

    fig.suptitle(
        rf"2T-orbit convergence check  ({title_ic}),  "
        rf"$t/T_{{\rm phys}} \in [{t_lo}, {t_hi}]$",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(outdir, f"check_2T_conv_{tag_ic}")
    fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.3, dpi=140)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.3)
    print(f"Saved -> {outpath}.png/.pdf")
    return fig


_render(X_fom_on, X_rom_on, tag_ic="on", title_ic="on-manifold IC")
_render(X_fom_off, X_rom_off, tag_ic="off", title_ic="off-manifold IC")


# ── Quantitative distance to the 2T orbit ─────────────────────────────
# Phase-align each snapshot: min over τ of ||x(t) − c_2T(τ)||.  Computed
# over the FULL trajectory, then late-window statistics are reported for
# the "have we arrived?" question and the full trace is used for the
# rate fit below.
n_tau = 400
tau_grid = np.linspace(0.0, T_2T, n_tau, endpoint=False)
C_grid_2T = c_2T_at(tau_grid)                # (n, n_tau)
ref_norm = float(np.linalg.norm(C_grid_2T, axis=0).mean())


def distance_to_2T(X, tag):
    dists = np.zeros(n_t)
    for k in range(n_t):
        diff = X[:, k:k+1] - C_grid_2T
        dists[k] = float(np.linalg.norm(diff, axis=0).min())
    dl = dists[mask]
    print(f"  [{tag}] mean/max ||x − c_2T|| / mean||c_2T|| (t/T ∈ "
          f"[{t_lo:.0f},{t_hi:.0f}]) = "
          f"{dl.mean()/ref_norm:.4e} / {dl.max()/ref_norm:.4e}")
    return dists


print("\nDistance to 2T orbit (phase-optimal):")
d_fom_on  = distance_to_2T(X_fom_on,  "FOM on ")
d_rom_on  = distance_to_2T(X_rom_on,  "ROM on ")
d_fom_off = distance_to_2T(X_fom_off, "FOM off")
d_rom_off = distance_to_2T(X_rom_off, "ROM off")


# ── Trace log(dist) vs t at a coarse grid so the fit window is picked
# from real data, not from a guess. ─────────────────────────────────────
print("\nlog₁₀(||x − c_2T(τ*)||/||c_2T||)  vs  t/T_phys  (coarse trace):")
print(f"  {'t/T':>6s}  {'FOM_on':>8s}  {'ROM_on':>8s}  "
      f"{'FOM_off':>8s}  {'ROM_off':>8s}")
for tk in np.linspace(0.0, t_end / T_phys, 16):
    idx = int(np.clip(round(tk * T_phys / t_end * (n_t - 1)), 0, n_t - 1))
    print(f"  {tk:>6.2f}  "
          f"{np.log10(d_fom_on[idx]/ref_norm):>8.3f}  "
          f"{np.log10(d_rom_on[idx]/ref_norm):>8.3f}  "
          f"{np.log10(d_fom_off[idx]/ref_norm):>8.3f}  "
          f"{np.log10(d_rom_off[idx]/ref_norm):>8.3f}")


# ── Empirical decay-rate fits over the FULL trajectory ────────────────
# Fit log||x − c_2T(τ*)|| = a + b·t on a window bracketing the visible
# log-linear regime.  Adjust `fit_window` after eyeballing the trace
# above.  b > 0 during growth toward s*, b < 0 during decay to the fixed
# point; comparison to g'(s*) checks whether the ROM's saturation is
# what actually gets seen.
def fit_rate(dists, tag, fit_window):
    m = (t_eval / T_phys >= fit_window[0]) & (t_eval / T_phys <= fit_window[1])
    if m.sum() < 5:
        print(f"  [{tag}] fit window empty, skipping")
        return None
    y = dists[m]
    y_floor = max(dists.min() * 3.0, 1e-30)
    m2 = y > y_floor
    if m2.sum() < 5:
        print(f"  [{tag}] all points at noise floor, skipping")
        return None
    slope, _ = np.polyfit(t_eval[m][m2], np.log(y[m2]), 1)
    print(f"  [{tag}] rate b = {slope:+.4e}   "
          f"({'growth' if slope > 0 else 'decay'} time = "
          f"{abs(1.0 / slope):.3f})")
    return slope


# Two windows: growth (linear master-mode growth at rate Λ ≈ 0.21) and
# saturation (approach to s* at rate g'(s*) ≈ −2.35).  The exact bounds
# should be adjusted after glancing at the trace table above.
fit_growth = (2.0, 8.0)
fit_decay = (10.0, 13.0)

print(f"\nGROWTH-phase rate fit, t/T_phys ∈ {fit_growth}:")
b_fom_on_g  = fit_rate(d_fom_on,  "FOM on ", fit_growth)
b_rom_on_g  = fit_rate(d_rom_on,  "ROM on ", fit_growth)
b_fom_off_g = fit_rate(d_fom_off, "FOM off", fit_growth)
b_rom_off_g = fit_rate(d_rom_off, "ROM off", fit_growth)

print(f"\nSATURATION-phase rate fit, t/T_phys ∈ {fit_decay}:")
b_fom_on  = fit_rate(d_fom_on,  "FOM on ", fit_decay)
b_rom_on  = fit_rate(d_rom_on,  "ROM on ", fit_decay)
b_fom_off = fit_rate(d_fom_off, "FOM off", fit_decay)
b_rom_off = fit_rate(d_rom_off, "ROM off", fit_decay)


# ── Theoretical ROM rate: g'(s*) from the SSM polynomial ──────────────
# g(s) = Σ_j g_j s^j (r=1); linearise at s* → ds/dt ≈ g'(s*)·(s−s*).
# So (s − s*) ~ exp(g'(s*)·t) and ||decode(t, s) − decode(t, s*)|| decays
# at the same rate (Jacobian factor is bounded).  For a scalar latent
# with real Λ this is real and negative.
multiindices = ctx["cache"]["multiindices"]
gs_flat = ctx["cache"]["gs"][:, 0]
max_deg = int(multiindices[:, 0].max())
g_poly = np.zeros(max_deg + 1, dtype=complex)
for k in range(len(multiindices)):
    g_poly[int(multiindices[k, 0])] += complex(gs_flat[k])
# g'(s) = Σ_{j≥1} j·g_j·s^{j−1}
g_prime_at = sum(j * g_poly[j] * s_star ** (j - 1)
                 for j in range(1, max_deg + 1))
print(f"\nROM rate at s*:  g'(s*) = {g_prime_at.real:+.4e}"
      f"{g_prime_at.imag:+.4e}j")
print(f"                                       1/e-time = "
      f"{-1.0 / g_prime_at.real if g_prime_at.real < 0 else float('inf'):.3f}")


# ── Verdict ───────────────────────────────────────────────────────────
print("\nExpected linear-regime rates:")
print(f"  Growth  (small |s|):   Re(Λ) = {Lam0.real:+.4e}  ← T-orbit master mode")
print(f"  Decay   (s → s*):    g'(s*) = {g_prime_at.real:+.4e}  ← ROM saturation")
print("\nInterpretation:")
print("  · b_rom_on in the GROWTH window should track Re(Λ).  If it doesn't,")
print("    the fit window sits partly in the saturation regime — move it.")
print("  · b_rom_on in the SATURATION window should track g'(s*).")
print("  · b_fom - b_rom on either window:")
print("      ≈ 0    → FOM and ROM decay along the same direction — no")
print("               extra slow Floquet mode of c_2T bottlenecks the FOM.")
print("      > 0    → FOM slower than ROM (less negative), meaning there")
print("               IS a slow-decay Floquet direction of c_2T outside")
print("               the SSM that dominates the tail.")

plt.show(block=True)

PETSc.COMM_WORLD.Barrier()
os._exit(0)
