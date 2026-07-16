"""
Figure 2: Floquet multipliers of the Rössler T-periodic base flow at
c near period-doubling.

Loads the Floquet exponents (`L`) already stored in
``data/eigendecomp_cache_2T.npz`` by ``save_eigendecomp_2T.py``, converts
to Floquet multipliers, and plots them against the unit circle.

The exponents are in the 2T-HB frame (period T_HB = 2·T_base), i.e.
what SLEPc returned after the orbit-tangent neutral subspace was
deflated.  So we plot the physical T_base multipliers

    mu_T = exp(Lambda · T_base)

which places the period-doubling mode close to -1 on the negative real
axis (Lambda ≈ i·pi/T_base for that mode in the 2T-HB frame).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from _fig_common import setup_matplotlib, build_context, ensure_outdir, C_2T


setup_matplotlib()
ctx = build_context()
T_base = ctx["T_base"]

# Prefer the direct-monodromy multipliers (all 3, no branch ambiguity)
# from save_floquet_multipliers.py; fall back to the SLEPc 2T-HB
# eigendecomposition if the direct file isn't there yet.
mult_path = "data/floquet_multipliers.npz"
if os.path.exists(mult_path):
    print(f"Using direct monodromy multipliers from {mult_path}")
    fq = np.load(mult_path)
    mu = fq["mu"]                     # (3,) — all 3 T_base multipliers
    L = fq["L"]
    print(f"Loaded {mu.size} Floquet multipliers.")
    for j, m in enumerate(mu):
        print(f"  μ_{j+1} = {m.real:+.6f}{m.imag:+.6f}j     "
              f"|μ| = {abs(m):.6f}    "
              f"Λ = {L[j].real:+.4e}{L[j].imag:+.4e}j")
else:
    print(f"WARNING: {mult_path} not found — falling back to "
          f"SLEPc 2T-HB parity-resolved multipliers.  "
          f"Run save_floquet_multipliers.py for the direct 3-multiplier version.")
    eigcache = np.load("data/eigendecomp_cache_2T.npz")
    L = eigcache["L"]
    Phi = eigcache["Phi"]
    nf = int(eigcache["nf"])
    n = int(eigcache["n"])
    n_harm = 2 * nf + 1
    assert Phi.shape[0] == n_harm * n, (
        f"Phi row count {Phi.shape[0]} != n_harm·n = {n_harm * n}"
    )
    print(f"Loaded {L.size} Floquet exponents from data/eigendecomp_cache_2T.npz")
    for kk, lam in enumerate(L):
        print(f"  Lambda_{kk+1} = {lam.real:+.4e}{lam.imag:+.4e}j")


# ── Branch fix (only needed for the HB fallback path) ────────────────
# The direct-monodromy path already has unambiguous T_base multipliers.
# For the HB fallback, resolve the ± sign via harmonic parity: even-k
# support → T_base-periodic → +exp(Λ·T_base); odd-k support → period-
# doubled → −exp(Λ·T_base).
if not os.path.exists(mult_path):
    k_arr = np.arange(-nf, nf + 1)
    even_mask = (k_arr % 2 == 0)
    odd_mask = ~even_mask
    mu_raw = np.exp(L * T_base)
    mu = np.zeros_like(mu_raw)
    branch = []
    for j in range(len(L)):
        phi = Phi[:, j].reshape(n_harm, n)
        energy_per_k = np.sum(np.abs(phi) ** 2, axis=1)
        e_even = float(np.sum(energy_per_k[even_mask]))
        e_odd = float(np.sum(energy_per_k[odd_mask]))
        ratio = e_odd / max(e_even + e_odd, 1e-30)
        if e_odd > e_even:
            mu[j] = -mu_raw[j]
            branch.append(("odd", ratio))
        else:
            mu[j] = +mu_raw[j]
            branch.append(("even", ratio))
    print("\nFloquet multipliers (over T_base, branch resolved from HB parity):")
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

# Filled markers = physical Floquet multipliers.  For the HB fallback
# path we also render the ± counterpart branch (hollow markers) to
# expose the sign ambiguity; the direct-monodromy path has no
# ambiguity so we skip it.
if not os.path.exists(mult_path):
    mu_other = -mu
    ax.scatter(
        mu_other.real, mu_other.imag, s=100,
        facecolors="none", edgecolors="0.4", linewidths=1.0, zorder=4,
        label=r"$\mp \exp(\Lambda \cdot T_{\rm base})$  (other branch)",
    )
ax.scatter(
    mu.real, mu.imag, s=80, color=C_2T,
    edgecolor="black", linewidth=0.7, zorder=5,
)

# Circle any GENUINELY unstable multiplier (|μ| > 1) in blue.  Exclude
# anything within 1e-3 of μ = +1 so the neutral direction (which sits
# there by construction, sometimes slightly outside the unit circle by
# integration noise) isn't highlighted.
is_neutral = np.abs(mu - 1.0) < 1e-3
unstable = (np.abs(mu) > 1.0) & (~is_neutral)
if np.any(unstable):
    ax.scatter(
        mu.real[unstable], mu.imag[unstable],
        s=520, facecolors="none", edgecolors="#2166ac",
        linewidths=2.0, zorder=6,
    )

ax.legend(loc="lower left", fontsize=11, framealpha=0.9)

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
fig.savefig(outpath + ".png", bbox_inches="tight", pad_inches=0.4)
fig.savefig(outpath + ".pdf", bbox_inches="tight", pad_inches=0.4)
print(f"\nSaved -> {outpath}.png/.pdf")

plt.show(block=True)


PETSc.COMM_WORLD.Barrier()
os._exit(0)
