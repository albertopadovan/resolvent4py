"""
Figure 4: fluctuations around the T-periodic base flow

    delta_a_j(t) = x_j(t) - c_star_j(t)

for j = 1..4, comparing full-order KSE (FOM) against the SSM ROM
prediction  delta_a_j_rom(t) = decode(t, s(t))_j.

Two figures:
    fig4_ts_on_manifold   IC exactly on the SSM.
    fig4_ts_off_manifold  IC with a decent transverse kick in modes
                          j = 1..4.
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


# ── Integration parameters ───────────────────────────────────────────────
# Long enough for the off-manifold transverse transient to decay while
# still showing a few clean 2T oscillations on the tail.  Both ROM and
# FOM use RK45 with the same tolerances so any observed rate mismatch
# reflects physics (Floquet spectrum of c_2T vs the SSM master mode),
# not integrator differences.
t_end = 20.0 * T_phys
n_t = 4000
rtol, atol = 1e-11, 1e-12
t_eval = np.linspace(0.0, t_end, n_t)


def _kse_rhs(_t, q):
    return lam * q + N_fn(q)


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
        _kse_rhs, [0.0, t_end], x0_full,
        method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    ).y
    dX = np.zeros_like(X_out)
    for i in range(n_t):
        dX[:, i] = X_out[:, i] - c_star_at(t_eval[i])
    return dX


# ── Initial conditions ──────────────────────────────────────────────────
# Sit on the polynomial fixed point s* of g(s)=0 — the ROM's latent
# dynamics has ds/dt = 0 there, so decode(t, s*) is exactly the 2T-orbit
# perturbation from c_star(t).  Starting at s0 = 0.5 s* instead would
# leave the ROM at rate ~ Re(Lam) which is tiny near the period-doubling
# bifurcation and produces essentially constant, tiny oscillations over
# any reasonable integration window.
s0 = 0.1 * np.array([s_star], dtype=complex)
delta_ssm_0 = rom.decode(0.0, s0).real
x0_on = c_star_at(0.0) + delta_ssm_0
Lam0 = complex(ctx["cache"]["Lams"][0])
print(f"On-manifold IC   s0 = s* = {s_star:.4f}   "
      f"(rho_domain = {ctx['rho_domain']:.4f})")
print(f"                 Re(Lam) = {Lam0.real:+.4e}   "
      f"(bifurcation rate — small ⇒ we start at the fixed point)")
print(f"                 ||decode(0, s*)|| = {np.linalg.norm(delta_ssm_0):.4f}")

# Off-manifold: unit-norm equal-weight kick in modes 1..4, scaled to
# match the SSM perturbation amplitude so the transverse transient is
# clearly visible in every panel.
transverse = np.zeros(n)
transverse[0:4] = 1.0
transverse = transverse / np.linalg.norm(transverse)
eps = 1.0 * float(np.linalg.norm(delta_ssm_0))
x0_off = x0_on + 0.1 *transverse
print(f"Off-manifold IC:  ||kick|| = {eps:.4f}   "
      f"(equal-weight sum of modes j = 1..4)")

# The off-manifold ROM must re-project via encode = w(0)ᵀ Δ; using the
# on-manifold s0 would silently ignore the transverse kick's along-
# manifold component.
s0_off = rom.encode(0.0, x0_off - c_star_at(0.0))
print(f"                  encode(0, Δ_off) = {s0_off[0]:+.6f}   "
      f"(vs on-manifold s0 = {s0[0]:+.6f})")


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


def _plot_4x1(dX_fom, dX_rom, tag):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9.5, 8.5))
    for j in range(4):
        ax = axes[j]
        ax.plot(t_eval / T_phys, dX_fom[j], color=C_FOM, lw=1.6,
                label="FOM (truth)")
        ax.plot(t_eval / T_phys, dX_rom[j], color=C_ROM, lw=1.6, ls="--",
                label="ROM (SSM)")
        ax.set_ylabel(
            rf"$a_{{{j+1}}} - \bar a_{{{j+1}}}(t)$",
            fontsize=20, labelpad=6,
        )
        ax.tick_params(axis="both", which="major", labelsize=15)
    axes[-1].set_xlabel(r"$t / T$", fontsize=22, labelpad=6)
    axes[0].legend(loc="upper right", fontsize=15, ncol=2)
    fig.tight_layout()

    outpath = os.path.join(outdir, f"fig4_ts_{tag}_manifold")
    fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.3)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.3)
    print(f"Saved -> {outpath}.png/.pdf")
    return fig


_plot_4x1(dX_fom_on, dX_rom_on, tag="on")
_plot_4x1(dX_fom_off, dX_rom_off, tag="off")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
