"""
Reproduce figure 1(c) of Figueras & de la Llave (arXiv:1605.01085) using
the chart-1 SSM ROM.

The paper plots the projection onto the (a_1, a_2) Fourier coordinates
of a periodic orbit just past the first period-doubling.  Panels:

  b)  1/ν = 33.2701   1T orbit (figure-8, single loop)
  c)  1/ν = 33.3353   2T orbit (double-traversed figure-8)
  d)  1/ν = 33.3569   4T orbit

The chart-1 SSM at this 1/ν has a stable polynomial fixed point
``s*`` (root of ``g(s) = 0``) at the 2T orbit branch.  Reconstructing
the orbit is just

    x(t) = c_star(t) + decode(t, s*)              t ∈ [0, T)

where ``c_star(t)`` is the cached 1T orbit (HB period = 2·T_phys) and
``decode`` is the SSM polynomial map.  No time-stepping required.

  • a_1 = x[0]
  • a_2 = x[1]

The chart-1 cache should be built at the closest available 1/ν to the
paper's panel (the user's save_ssm.py currently uses 1/ν = 33.31, very
close to panel (c) at 33.3353; the qualitative figure-8 shape is robust
in this regime).

Output: ``results/fig1c_repro_latent.png``
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM


# ── Load chart-1 SSM cache ─────────────────────────────────────────────────
cache = np.load("data/ssm_cache.npz")
PS_hb = cache["PS_hb"]
W_hb = cache["W_hb"]
multiindices = cache["multiindices"]
Lams = cache["Lams"]
gs = cache["gs"]
conj_to_linear = bool(cache["conj_to_linear_dynamics"])

nu = float(cache["nu"])
n = int(cache["n"])
T = float(cache["T"])                  # HB period = 2·T_phys
T_phys = T / 2.0
rho_domain = float(cache["rho_domain"])
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
omega = 2.0 * np.pi / T

print(
    f"Loaded chart-1 SSM cache:\n"
    f"  1/ν = {1/nu:.4f}   (paper panel c uses 1/ν = 33.3353)\n"
    f"  T (HB period = 2·T_phys) = {T:.4f},   T_phys = {T_phys:.4f}\n"
    f"  Λ_1 = {complex(Lams[0]).real:+.4e}\n"
    f"  ρ_1 = {rho_domain:.4f}"
)


# ── ROM (decoder only; we won't integrate the latent ODE) ────────────────
rom = SpectralSubmanifoldROM(
    multiindices=multiindices, Lams=Lams, gs=gs, PS=PS_hb, W=W_hb,
    conj_to_linear_dynamics=conj_to_linear, omega=omega,
)


# ── c_star_at(t) — periodic interpolant of the 1T orbit ─────────────────
time_ext = np.append(time_orbit, T)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)
def c_star_at(t):
    return c_star_interp(t % T)


# ── Find the 2T-orbit polynomial fixed point s* of g(s) = 0 ─────────────
# For r=1 the cubic-pitchfork normal form gives  s* = ±√(−Λ/g_3)  to leading
# order; the full polynomial roots correct that with higher orders.
g_coeffs_low_to_high = np.array(
    [complex(gs[j_idx, 0]) for j_idx in range(len(multiindices))]
)
roots = np.roots(g_coeffs_low_to_high[::-1])
# Keep real-valued non-trivial roots inside the convergence radius.
real_roots = [
    z.real for z in roots
    if abs(z.imag) < 1e-3 * max(abs(z.real), 1.0)
       and abs(z) > 0.1 * rho_domain
       and abs(z) < rho_domain * 1.10            # tolerate slight over-shoot
]
real_roots = sorted(real_roots, key=lambda x: -abs(x))   # largest |·| first
if not real_roots:
    raise RuntimeError(
        "No non-trivial real fixed point of g(s)=0 found inside ρ — "
        "the SSM may not be in the period-doubled regime, or ρ is too tight."
    )
s_star_pos = +abs(real_roots[0])
s_star_neg = -abs(real_roots[0])
print(f"\n2T-orbit fixed points of g(s) = 0:  s* = ±{abs(s_star_pos):.4f}")


# ── Reconstruct the 2T orbit on one fundamental period T = 2·T_phys ─────
n_samples = 2000
t_orbit = np.linspace(0.0, T, n_samples, endpoint=False)

# Branch (+)
X_2T_pos = np.zeros((n, n_samples))
for i, t in enumerate(t_orbit):
    s = np.array([s_star_pos], dtype=complex)
    X_2T_pos[:, i] = c_star_at(t) + rom.decode(t, s).real

# Branch (−)  (Z₂ symmetry partner)
X_2T_neg = np.zeros((n, n_samples))
for i, t in enumerate(t_orbit):
    s = np.array([s_star_neg], dtype=complex)
    X_2T_neg[:, i] = c_star_at(t) + rom.decode(t, s).real


# ── Also reconstruct the 1T orbit (= c_star itself, no SSM contribution) ─
X_1T = np.zeros((n, n_samples))
for i, t in enumerate(t_orbit):
    X_1T[:, i] = c_star_at(t)


# ── Plot a_1 (= x[0]) vs a_2 (= x[1]) ────────────────────────────────────
os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

# Left: just the 2T orbit (both branches), mirroring paper's red on white.
ax = axes[0]
ax.plot(
    X_2T_pos[0], X_2T_pos[1],
    color="#C42021", lw=1.0, label=rf"2T orbit  (s$^*$ = +{s_star_pos:.3f})",
)
ax.plot(
    X_2T_neg[0], X_2T_neg[1],
    color="#C42021", lw=1.0, ls="--", alpha=0.7,
    label=rf"2T orbit  (s$^*$ = {s_star_neg:.3f})",
)
ax.set_xlim(-4, 4); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$a_1$"); ax.set_ylabel(r"$a_2$")
ax.set_title(
    rf"Latent-space reconstruction  (1/$\nu$ = {1/nu:.4f}, "
    rf"$s^* = \pm{abs(s_star_pos):.3f}$)"
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# Right: 1T orbit (= c_star) on top of the 2T for context.
ax = axes[1]
ax.plot(
    X_1T[0], X_1T[1],
    color="0.5", lw=1.0, ls="-",
    label=rf"1T orbit  ($s = 0$)",
)
ax.plot(
    X_2T_pos[0], X_2T_pos[1],
    color="#C42021", lw=1.0,
    label=rf"2T orbit  ($s^* = +{s_star_pos:.3f}$)",
)
ax.plot(
    X_2T_neg[0], X_2T_neg[1],
    color="#C42021", lw=1.0, ls="--", alpha=0.7,
    label=rf"2T orbit  ($s^* = {s_star_neg:.3f}$)",
)
ax.set_xlim(-4, 4); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$a_1$")
ax.set_title("1T orbit overlaid for context")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(
    r"Reproduction of Figueras–de la Llave (2016) Figure 1(c) "
    r"using chart-1 SSM ROM",
    fontsize=12,
)
fig.tight_layout()

outpath = "results/fig1c_repro_latent"
fig.savefig(outpath + ".png", dpi=300, bbox_inches="tight")
fig.savefig(outpath + ".pdf", bbox_inches="tight")
print(f"\nSaved → {outpath}.png/.pdf")
plt.show()
