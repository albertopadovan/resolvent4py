"""
Figure 4: fluctuations around the T-periodic Rössler base flow

    delta_a_j(t) = x_j(t) - c_star_j(t),   j = x, y, z

comparing full-order Rössler (FOM) against the SSM ROM prediction
delta_a_j_rom(t) = decode(t, s(t))_j.

Two figures:
    fig4_ts_on_manifold   IC exactly on the SSM.
    fig4_ts_off_manifold  IC with a decent transverse kick in state
                          space; the ROM re-projects via encode.
"""

import os
from functools import partial

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


# ── Integration parameters ───────────────────────────────────────────────
# Long enough for the off-manifold transverse transient to decay while
# still showing a few clean 2T oscillations on the tail.  Both ROM and
# FOM use RK45 with the same tolerances so any observed rate mismatch
# reflects physics (Floquet spectrum of c_2T vs the SSM master mode),
# not integrator differences.
t_end = 20.0 * T_base
n_t = 4000
rtol, atol = 1e-11, 1e-12
t_eval = np.linspace(0.0, t_end, n_t)


def _rossler_rhs(t, q):
    return rossler_rhs(t, q, c)


def _integrate_rom(s0):
    S = sp.integrate.solve_ivp(
        rom.latent_space_dynamics, [0.0, t_end], s0,
        method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    ).y
    dX = np.zeros((n, n_t))
    for i in range(n_t):
        dX[:, i] = rom.decode(t_eval[i], S[:, i]).real
    return dX


def _integrate_fom_fluctuations(x0_full):
    X_out = sp.integrate.solve_ivp(
        _rossler_rhs, [0.0, t_end], x0_full,
        method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    ).y
    dX = np.zeros_like(X_out)
    for i in range(n_t):
        dX[:, i] = X_out[:, i] - c_star_at(t_eval[i])
    return dX


# ── Initial conditions ──────────────────────────────────────────────────
# On-manifold IC:  start at a SMALL fixed latent amplitude s0 ≈ 0.5 so
# that the encoder is in the "approximately left-inverse of P(t, s)"
# regime.  At small |s|, decode(t, s) ≈ s · v(t) + O(s²), and
# encode(t, v · s) ≈ s to O(s²) — polynomial corrections are negligible.
# This lets us watch the master-mode's linear growth and cleanly
# compare the FOM's SSM projection against the ROM's autonomous s(t).
s0 = np.array([0.5 + 0.0j], dtype=complex)
delta_ssm_0 = rom.decode(0.0, s0).real
x0_on = c_star_at(0.0) + delta_ssm_0
Lam0 = complex(ctx["cache"]["Lams"][0])
print(f"On-manifold IC   s0 = {s0[0].real:.4f}   "
      f"(s* = {s_star:.4f},  rho_domain = {ctx['rho_domain']:.4f})")
print(f"                 Re(Lam) = {Lam0.real:+.4e}")
print(f"                 ||decode(0, s0)|| = {np.linalg.norm(delta_ssm_0):.4f}")

# Off-manifold IC:  add a transverse kick with magnitude ~ ||decode|| so
# it's visible in every panel but small enough that
# encode(0, x0_off − c_*(0)) still lands at ≈ s0 + O(eps),  well below
# any nonlinear-encode saturation.  The kick is first projected via
# rom.neutral_project(0, ·) to strip its component along the orbit
# tangent (the phase-shift / neutral Floquet mode).  Otherwise the FOM
# picks up a slow linear phase drift along c_*(t) that the SSM can't
# see, contaminating the transverse-decay signal we're after.
transverse = np.ones(n) / np.sqrt(n)
# transverse = rom.neutral_project(0.0, transverse).real
transverse /= np.linalg.norm(transverse)
eps = 0.5 * float(np.linalg.norm(delta_ssm_0))
x0_off = x0_on + eps * transverse
print(f"Off-manifold IC:  ||kick|| = {eps:.4f}   "
      f"(after neutral-projection, then renormalised to unit vector)")

# The off-manifold ROM must re-project via encode = w(0)ᵀ Δ; using the
# on-manifold s0 would silently ignore the transverse kick's along-
# manifold component.
s0_off = rom.encode(0.0, x0_off - c_star_at(0.0)).real
print(s0_off)
print(f"                  encode(0, Δ_off) = {s0_off[0]:+.6f}   "
      f"(vs on-manifold s0 = {s0[0]:+.6f})")
if abs(s0_off[0]) > 1.0:
    print(f"                  ⚠ |encoded s0| > 1 — encoder may not be a "
          f"good left-inverse.  Reduce eps or s0.")


# ── Run everything ──────────────────────────────────────────────────────
print("\nROM integration (on-manifold IC) ...")
dX_rom_on = _integrate_rom(s0)

print("ROM integration (off-manifold IC, s0 = w(0)ᵀ Δ_off) ...")
dX_rom_off = _integrate_rom(s0_off)

print("FOM integration (on-manifold IC) ...")
dX_fom_on = _integrate_fom_fluctuations(x0_on)

print("FOM integration (off-manifold IC) ...")
dX_fom_off = _integrate_fom_fluctuations(x0_off)


# ── Render ──────────────────────────────────────────────────────────────
outdir = ensure_outdir()

mode_labels = ["x", "y", "z"]


def _plot_3x1(dX_fom, dX_rom, tag):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 7.0))
    for j in range(3):
        ax = axes[j]
        ax.plot(t_eval / T_base, dX_fom[j], color=C_FOM, lw=1.6,
                label="FOM (truth)")
        ax.plot(t_eval / T_base, dX_rom[j], color=C_ROM, lw=1.6, ls="--",
                label="ROM (SSM)")
        ax.set_ylabel(
            rf"$\delta {mode_labels[j]}(t)$",
            fontsize=20, labelpad=6,
        )
        ax.tick_params(axis="both", which="major", labelsize=15)
    axes[-1].set_xlabel(r"$t / T$", fontsize=22, labelpad=6)
    axes[0].legend(loc="upper right", fontsize=15, ncol=2)
    fig.tight_layout()

    outpath = os.path.join(outdir, f"fig4_ts_{tag}_manifold")
    fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
    print(f"Saved -> {outpath}.png/.pdf")
    return fig


_plot_3x1(dX_fom_on, dX_rom_on, tag="on")
_plot_3x1(dX_fom_off, dX_rom_off, tag="off")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
