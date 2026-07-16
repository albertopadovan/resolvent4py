r"""
Study 1 (on-manifold):  compare the ROM's autonomous latent  s(t)  with
the FOM's along-SSM projection  s_FOM(t) = w(t)ᵀ (q(t) − c_*(t)).

Setup:
    IC on SSM:   x_0 = c_*(0) + decode(0, s_0),   s_0 = 0.5
    ROM:         ds/dt = g(s),                    s(0) = s_0.
    FOM:         Rössler(t, q),                   q(0) = x_0.

If the FOM stays on the SSM (which it should, given ε_transverse = 0
at t = 0 and the SSM being an invariant manifold), then

    s_FOM(t) = w(t)ᵀ (q(t) − c_*(t))    =    s_ROM(t)   for all t.

Any drift is a signature of either
    (a) the polynomial SSM parameterisation P(t, s) not being exactly
        invariant at finite polynomial order,  OR
    (b) the encode/decode pair not being a true left-inverse away from
        the small-s linearisation.
"""
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import (
    setup_matplotlib, build_context, ensure_outdir,
    C_FOM, C_ROM,
)
from rossler_rhs import rossler_rhs


setup_matplotlib()
ctx = build_context()
rom = ctx["rom"]
c_star_at = ctx["c_star_at"]
s_star = ctx["s_star"]
T_HB = ctx["T_HB"]
T_base = ctx["T_base"]
n = ctx["n"]
c = ctx["c"]


# ── Parameters ────────────────────────────────────────────────────────
s0 = np.array([0.5 + 0.0j], dtype=complex)
n_periods = 300
t_end = n_periods * T_base
n_t = 20000
rtol, atol = 1e-11, 1e-13
t_eval = np.linspace(0.0, t_end, n_t)

Lam0 = complex(ctx["cache"]["Lams"][0])
print(f"On-manifold study:  c = {c},  T_base = {T_base:.4f},  "
      f"T_HB = {T_HB:.4f}")
print(f"  s0 = {s0[0].real:+.4f}   (s* = {s_star:.4f},  "
      f"rho_domain = {ctx['rho_domain']:.4f})")
print(f"  Re(Λ) = {Lam0.real:+.4e}   Im(Λ) = {Lam0.imag:+.4e}")
print(f"  t_end = {n_periods} · T_base = {t_end:.4f}")


# ── ROM: autonomous s(t) ─────────────────────────────────────────────
print("\nIntegrating ROM (autonomous ds/dt = g(s)) ...")
sol_rom = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [0.0, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
)
s_rom = sol_rom.y[0]
t_rom = sol_rom.t
n_t_rom = len(t_rom)
if n_t_rom < n_t:
    print(f"  ⚠ ROM terminated early at t/T_base = "
          f"{t_rom[-1]/T_base:.3f}  (last |s| = {abs(s_rom[-1]):.3f})")


# ── FOM: full Rössler, IC exactly on the SSM ─────────────────────────
v0 = rom.decode(0.0, s0).real
x0 = c_star_at(0.0) + v0
print(f"  ||decode(0, s0)|| = {np.linalg.norm(v0):.4f}")


def _rhs(t, q):
    return rossler_rhs(t, q, c)


print("Integrating FOM (Rössler) ...")
sol_fom = sp.integrate.solve_ivp(
    _rhs, [0.0, t_end], x0,
    method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    dense_output=True,
)
q_fom = sol_fom.y                        # (n, n_t_fom)
t_fom = sol_fom.t
print(f"  FOM nfev = {sol_fom.nfev}")


# ── s_FOM(t) = w(t)ᵀ (q(t) − c_*(t)) ─────────────────────────────────
print("\nProjecting FOM onto the SSM latent  s_FOM(t) = "
      "w(t)ᵀ (q(t) − c_*(t)) ...")
s_fom = np.zeros(len(t_fom), dtype=complex)
for i, t in enumerate(t_fom):
    delta_q = q_fom[:, i] - c_star_at(t)
    s_fom[i] = rom.encode(t, delta_q)[0]
s_fom = s_fom.real


# ── Plot ─────────────────────────────────────────────────────────────
outdir = ensure_outdir()
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6.5))

# Top: s_ROM(t) and s_FOM(t) overlaid
ax = axes[0]
ax.plot(t_rom / T_base, s_rom.real, "-", color=C_ROM, lw=1.6,
        label=r"$s_{\mathrm{ROM}}(t)$  (autonomous $ds/dt = g(s)$)")
ax.plot(t_fom / T_base, s_fom, "--", color=C_FOM, lw=1.4, alpha=0.9,
        label=r"$s_{\mathrm{FOM}}(t) = w(t)^\top (q(t) - c_*(t))$")
ax.axhline(0.0, color="0.7", lw=0.5, ls=":")
ax.set_ylabel(r"$s(t)$", fontsize=18, labelpad=6)
ax.tick_params(axis="both", which="major", labelsize=13)
ax.legend(loc="best", fontsize=13)

# Bottom: log |s_ROM − s_FOM| — how well the SSM projection tracks
# the ROM's autonomous solution.  If < ~1e-10, they agree to numerical
# precision (FOM stays on the SSM).  If it drifts up in time,
# polynomial invariance error is accumulating.
n_common = min(len(t_rom), len(t_fom))
diff = np.abs(s_rom.real[:n_common] - s_fom[:n_common])
ax = axes[1]
ax.semilogy(t_rom[:n_common] / T_base, np.maximum(diff, 1e-30),
            "-", color="tab:purple", lw=1.4,
            label=r"$|s_{\mathrm{ROM}}(t) - s_{\mathrm{FOM}}(t)|$")
ax.set_xlabel(r"$t / T_{\mathrm{base}}$", fontsize=18, labelpad=6)
ax.set_ylabel(r"$|s_{\mathrm{ROM}} - s_{\mathrm{FOM}}|$",
              fontsize=16, labelpad=6)
ax.tick_params(axis="both", which="major", labelsize=13)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
ax.legend(loc="best", fontsize=13)

fig.suptitle(
    fr"Rössler on-manifold:  autonomous $s_{{\mathrm{{ROM}}}}(t)$  "
    fr"vs  FOM-projected $s_{{\mathrm{{FOM}}}}(t)$"
    fr"   ($c = {c}$,   $s_0 = {s0[0].real}$)",
    fontsize=13,
)
fig.tight_layout()

outpath = os.path.join(outdir, "study_on_manifold_s")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
print(f"\nSaved -> {outpath}.png/.pdf")


# ── FFT of s_FOM(t) over a T_HB-periodic tail window ─────────────────
# Window is EXACTLY one T_HB = 2·T_base, taken well after transients.
# With a 1·T_HB window the DFT bins land exactly at k · (1/T_HB) for
# integer k, so a genuinely T_HB-periodic tail shows up as sharp lines
# at integer k with no leakage.  Any energy at fractional k would
# indicate a period > T_HB (drift, sub-harmonics, or chaos).
n_win_T_HB = 1                                     # exactly 1 · T_HB
n_fft = 32768                                      # DFT length — fine grid
# Anchor the window at the END of the trajectory so we're on the
# saturated tail (as close to the 2T attractor as this run gets).
t_stop = float(t_fom[-1])
t_start = t_stop - n_win_T_HB * T_HB
if t_start < t_fom[0]:
    raise RuntimeError(
        f"Tail window [{t_start:.3f}, {t_stop:.3f}] extends before "
        f"available FOM data (t_start_fom = {t_fom[0]:.3f}).  Increase "
        f"n_periods or reduce n_win_T_HB."
    )
t_fft = np.linspace(t_start, t_stop, n_fft, endpoint=False)

# Sample q(t) directly from the FOM's dense-output interpolant, then
# encode.  This gives real high-frequency content up to whatever the
# RK45 step size supported — not the ~150-samples/T_HB of the raw
# t_eval grid.
q_win = sol_fom.sol(t_fft)                         # (n, n_fft)
s_fom_win = np.zeros(n_fft)
for i in range(n_fft):
    s_fom_win[i] = rom.encode(t_fft[i],
                              q_win[:, i] - c_star_at(t_fft[i]))[0].real

# Hann window not applied on purpose:  we want the exact bin structure.
# With integer periods in the window, leakage is minimal.
S = np.fft.rfft(s_fom_win) / n_fft
freqs = np.fft.rfftfreq(n_fft, d=(t_stop - t_start) / n_fft)   # (n_fft/2+1,)
k_axis = freqs * T_HB                              # harmonic index in T_HB frame
E = np.abs(S) ** 2

print(f"\nFFT window:  t ∈ [{t_start:.3f}, {t_stop:.3f}]  "
      f"= [{t_start/T_base:.1f}, {t_stop/T_base:.1f}] · T_base"
      f"   (1 × T_HB)")
print(f"  n_fft = {n_fft},  bin width Δf = "
      f"{freqs[1]:.4e}  (= {freqs[1] * T_HB:.4f} · 1/T_HB)")

# Print the top 8 peaks by energy
top = np.argsort(E)[::-1][:8]
top = top[np.argsort(k_axis[top])]
print(f"\nTop-8 spectral peaks:")
print(f"  {'k = f·T_HB':>12s}  {'f·T_base':>10s}  {'|S_k|²':>14s}")
for idx in top:
    print(f"  {k_axis[idx]:>12.4f}  {freqs[idx]*T_base:>10.4f}  "
          f"{E[idx]:>14.4e}")


fig2, ax = plt.subplots(figsize=(11, 4.5))
kmax_show = min(50, int(k_axis[-1]))
mask_show = k_axis <= kmax_show
E_floor = 1e-30
markerline, stemlines, baseline = ax.stem(
    k_axis[mask_show], np.maximum(E[mask_show], E_floor),
    linefmt="-", markerfmt="o", basefmt=" ",
)
plt.setp(markerline, markersize=5, markerfacecolor=C_FOM,
         markeredgecolor=C_FOM,
         label=r"$|\hat{s}_{\mathrm{FOM}}(k)|^2$")
plt.setp(stemlines, linewidth=1.2, color=C_FOM)
# Guides at integer k (T_HB harmonics) and even k (T_base harmonics)
for k in range(1, kmax_show + 1):
    ax.axvline(k, color="0.6", lw=0.5, ls=":", alpha=0.5, zorder=0)
for k in range(2, kmax_show + 1, 2):
    ax.axvline(k, color="tab:blue", lw=0.6, ls="--", alpha=0.4, zorder=0)
ax.set_yscale("log")
ax.set_xlabel(r"$k = f \cdot T_{HB}$   "
              r"(integer $k$ $\to$ $T_{HB}$-periodic; "
              r"even $k$ $\to$ $T_{\mathrm{base}}$-periodic)",
              fontsize=13, labelpad=6)
ax.set_ylabel(r"$|\hat{s}_{\mathrm{FOM}}(k)|^2$",
              fontsize=15, labelpad=6)
ax.set_xlim(-0.5, kmax_show + 0.5)
ax.set_ylim(1e-12, max(1.0, 10.0 * float(E[mask_show].max())))
ax.set_title(
    fr"Spectrum of  $s_{{\mathrm{{FOM}}}}(t) = w(t)^\top (q(t) - c_*(t))$   "
    fr"(1 $\times T_{{HB}}$ window,   "
    fr"$t/T_{{\mathrm{{base}}}} \in [{t_start/T_base:.0f}, "
    fr"{t_stop/T_base:.0f}]$)",
    fontsize=12,
)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
ax.legend(loc="best", fontsize=12)
fig2.tight_layout()

outpath_fft = os.path.join(outdir, "study_on_manifold_s_fft")
fig2.savefig(outpath_fft + ".png", bbox_inches="tight", pad_inches=0.4)
fig2.savefig(outpath_fft + ".pdf", bbox_inches="tight", pad_inches=0.4)
print(f"Saved -> {outpath_fft}.png/.pdf")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
