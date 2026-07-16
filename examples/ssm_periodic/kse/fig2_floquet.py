"""
Figure 2: Floquet multipliers of the KSE T-periodic base flow at 1/nu
near period-doubling.

Loads the Floquet exponents (`L`) already stored in
``data/eigendecomp_cache.npz`` by ``save_eigendecomp.py``, converts to
Floquet multipliers, and plots them against the unit circle.

The exponents are in the 2T-HB frame (period T_HB = 2·T_phys), i.e.
what SLEPc returned after the orbit-tangent neutral subspace was
deflated.  So we plot the physical T_phys multipliers

    mu_T = exp(Lambda · T_phys)

which places the period-doubling mode close to -1 on the negative real
axis (Lambda ≈ i·pi/T_phys for that mode in the 2T-HB frame).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import setup_matplotlib, build_context, ensure_outdir, C_2T


setup_matplotlib()
ctx = build_context()
T_phys = ctx["T_phys"]

eigcache = np.load("data/eigendecomp_cache.npz")
L = eigcache["L"]                                  # (3,) complex Λ_2T
Phi = eigcache["Phi"]                              # (n_harm·n, 3) complex
nf = int(eigcache["nf"])
n = int(eigcache["n"])
n_harm = 2 * nf + 1
assert Phi.shape[0] == n_harm * n, (
    f"Phi row count {Phi.shape[0]} != n_harm·n = {n_harm * n}"
)

print(f"Loaded {L.size} Floquet exponents from data/eigendecomp_cache.npz")
for kk, lam in enumerate(L):
    print(f"  Lambda_{kk+1} = {lam.real:+.4e}{lam.imag:+.4e}j")

# ── Branch fix via HB harmonic content ────────────────────────────────────
# In the 2T-HB frame each row of Phi is one k-harmonic with
#     k ∈ [-nf, ..., +nf]   (ordering from resolvent4py/spectral_submanifold/
#                            spectral_submanifold_rom.py:131 and
#                            kse_differential_equation.py:134)
# The 2T-HB Floquet exponent Λ_2T is defined only mod iω_HB = iπ/T_phys,
# so from Λ_2T alone the T_phys multiplier is ambiguous by a sign.
# The eigenvector resolves the ambiguity:
#     ψ(t) = Σ_k φ_k · exp(i k ω_HB t)
#   ⇒ ψ(T_phys) = Σ_k φ_k · exp(i k π) = Σ_k φ_k · (-1)^k
# A T_phys-periodic mode is supported on EVEN k only  (ψ(T_phys) =  ψ(0))
#   ⇒ μ_T = +exp(Λ·T_phys)
# A period-doubled mode is supported on ODD k only    (ψ(T_phys) = -ψ(0))
#   ⇒ μ_T = -exp(Λ·T_phys)
# Numerical eigenvectors will have residual energy on the "wrong-parity"
# harmonics; pick the branch with the DOMINANT parity.
k_arr = np.arange(-nf, nf + 1)
even_mask = (k_arr % 2 == 0)
odd_mask = ~even_mask

mu_raw = np.exp(L * T_phys)
mu = np.zeros_like(mu_raw)
branch = []
for j in range(len(L)):
    phi = Phi[:, j].reshape(n_harm, n)
    energy_per_k = np.sum(np.abs(phi) ** 2, axis=1)     # (n_harm,)
    e_even = float(np.sum(energy_per_k[even_mask]))
    e_odd = float(np.sum(energy_per_k[odd_mask]))
    ratio = e_odd / max(e_even + e_odd, 1e-30)
    if e_odd > e_even:
        mu[j] = -mu_raw[j]
        branch.append(("odd", ratio))
    else:
        mu[j] = +mu_raw[j]
        branch.append(("even", ratio))

print("\nFloquet multipliers (over T_phys, branch resolved from HB parity):")
for kk, (m, (parity, ratio)) in enumerate(zip(mu, branch)):
    print(
        f"  mu_{kk+1} = {m.real:+.4f}{m.imag:+.4f}j     "
        f"|mu| = {abs(m):.4f}   "
        f"({parity}-parity dominant, odd/total = {ratio:.3f})"
    )


# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 6.0))

# Unit circle
theta = np.linspace(0.0, 2.0 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color="0.4", lw=1.2)

# Axes lines
ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
ax.axvline(0.0, color="0.85", lw=0.6, zorder=0)

# Multipliers
ax.scatter(
    mu.real, mu.imag, s=140, color=C_2T,
    edgecolor="black", linewidth=0.7, zorder=5,
)

# Circle any unstable multiplier (|μ| > 1) in blue
unstable = np.abs(mu) > 1.0
if np.any(unstable):
    ax.scatter(
        mu.real[unstable], mu.imag[unstable],
        s=520, facecolors="none", edgecolors="#2166ac",
        linewidths=2.0, zorder=6,
    )

ax.set_xlabel(r"$\Re(\mu)$", fontsize=22, labelpad=8)
ax.set_ylabel(r"$\Im(\mu)$", fontsize=22, labelpad=8)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(False)

lim = 1.35 * max(1.05, float(np.max(np.abs(mu))))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

ax.text(
    0.03, 0.97, "Floquet multipliers",
    transform=ax.transAxes, ha="left", va="top", fontsize=17,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white", edgecolor="0.6", linewidth=0.6,
    ),
)

fig.tight_layout()


# ── Save ─────────────────────────────────────────────────────────────────
outdir = ensure_outdir()
outpath = os.path.join(outdir, "fig2_floquet")
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.3)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.3)
print(f"\nSaved -> {outpath}.png/.pdf")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
