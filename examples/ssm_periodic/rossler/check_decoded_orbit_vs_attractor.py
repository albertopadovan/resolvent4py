r"""
Compare the SSM's decoded 2T-periodic orbit against the Rössler FOM's
actual attractor at the target ``c``.

Two objects on the plot:

  1. ``x_ROM(t) = c_*(t) + decode(t, s*(t))``,   t ∈ [0, T_HB)
     where ``s*(t)`` is the ROM's periodic orbit.  For the periodic-g
     cache we look for it via HB-Newton; if that fails (typically
     because |s*| > R_g) we fall back to the constant ``s_dc*`` (the
     real root of the DC polynomial).  For the constant-g cache ``s*``
     IS a scalar and decode is periodic in t through PS(t) alone.

  2. FOM attractor tail from a long time-integration of the Rössler
     system at the same c.  A 100·T_base transient is discarded and
     the last 4·T_base are plotted.

Auto-dispatches on cache variant (`data/ssm_cache_2T.npz` /
`data/ssm_cache_2T_periodic_g.npz`).
"""
import os
import sys

import numpy as np
import scipy as sp
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from petsc4py import PETSc
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import (
    SpectralSubmanifoldROM,
    SpectralSubmanifoldROMPeriodicG,
)
from rossler_rhs import rossler_rhs


# ── Which cache ───────────────────────────────────────────────────────
cache_path = sys.argv[1] if len(sys.argv) > 1 \
    else "data/ssm_cache_2T_periodic_g.npz"
if not os.path.exists(cache_path):
    raise FileNotFoundError(cache_path)

cache = np.load(cache_path)
multiindices = cache["multiindices"]
degs = multiindices[:, 0].astype(int)
max_deg = int(degs.max())
T_HB = float(cache["T"])                       # = 2·T_base
T_base = float(cache["T_base"])
c = float(cache["c"])
n = int(cache["n"])
omega = 2.0 * np.pi / T_HB


# ── Build ROM (dispatch) ──────────────────────────────────────────────
if "gs_periodic" in cache.files:
    variant = "periodic-g"
    gs_arr = cache["gs_periodic"][:, :, 0]     # (n_terms, n_harm_g)
    n_harm_g = gs_arr.shape[1]
    nf_g = (n_harm_g - 1) // 2
    rom = SpectralSubmanifoldROMPeriodicG(
        multiindices=cache["multiindices"],
        Lams=cache["Lams"],
        gs=cache["gs_periodic"],
        PS=cache["PS_hb"],
        W=cache["W_hb"],
        omega=omega,
    )
    rom.use_g_spline = False
elif "gs" in cache.files:
    variant = "constant-g"
    gs_arr = cache["gs"][:, 0][:, None]        # (n_terms, 1)
    n_harm_g, nf_g = 1, 0
    rom = SpectralSubmanifoldROM(
        multiindices=cache["multiindices"],
        Lams=cache["Lams"],
        gs=cache["gs"],
        PS=cache["PS_hb"],
        W=cache["W_hb"],
        conj_to_linear_dynamics=bool(cache["conj_to_linear_dynamics"]),
        omega=omega,
        v_neutral=(cache["V_neut_hb"][:, :, 0]
                   if "V_neut_hb" in cache.files else None),
        w_neutral=(cache["W_neut_hb"][:, :, 0]
                   if "W_neut_hb" in cache.files else None),
    )
else:
    raise KeyError(cache_path)

print(f"[{variant}] {cache_path}")
print(f"  c = {c},  T_base = {T_base:.4f},  T_HB = 2·T_base = {T_HB:.4f}")


# ── c_*(t) interpolant ───────────────────────────────────────────────
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(
    t_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


# ── Determine s*(t) ──────────────────────────────────────────────────
# DC-polynomial root as a fallback / initial guess.
g_dc = np.zeros(max_deg + 1, dtype=complex)
for k_term, d in enumerate(degs):
    g_dc[d] += complex(gs_arr[k_term, nf_g] if variant == "periodic-g"
                       else gs_arr[k_term, 0])
roots = np.roots(g_dc[::-1])
real_roots = sorted(
    [z.real for z in roots
     if abs(z.imag) < 1e-6 * max(abs(z.real), 1.0)],
    key=lambda x: abs(x),
)
nontrivial = [r for r in real_roots if abs(r) > 1e-8]
s_dc_star = nontrivial[0] if nontrivial else 1.0
print(f"  s_dc* (DC-polynomial root)  = {s_dc_star:+.4f}")

# Try HB-Newton for periodic-g (skip for constant-g — s* is exact scalar).
s_hb_star_full = None
if variant == "periodic-g":
    nf_s = max(nf_g, 20)
    n_time_hb = 4 * (nf_g + max_deg * nf_s + 1)
    n_time_hb = max(n_time_hb, 4 * (2 * nf_s + 1))
    t_grid_hb = np.linspace(0.0, T_HB, n_time_hb, endpoint=False)
    k_idx_s = np.arange(-nf_s, nf_s + 1)
    weights_s = np.exp(1j * np.outer(t_grid_hb, k_idx_s * omega))
    k_idx_g = np.arange(-nf_g, nf_g + 1)
    weights_g = np.exp(1j * np.outer(t_grid_hb, k_idx_g * omega))
    g_j_of_t = (weights_g @ gs_arr.T).real

    def unpack(x):
        s_hb = np.zeros(2 * nf_s + 1, dtype=complex)
        s_hb[nf_s] = x[0]
        for k in range(1, nf_s + 1):
            s_hb[nf_s + k] = x[2 * k - 1] + 1j * x[2 * k]
            s_hb[nf_s - k] = np.conj(s_hb[nf_s + k])
        return s_hb

    def residual(x):
        s_hb = unpack(x)
        s_of_t = (weights_s @ s_hb).real
        G = np.zeros(n_time_hb)
        for k_term, d in enumerate(degs):
            G += g_j_of_t[:, k_term] * (s_of_t ** d)
        G_full = np.fft.fftshift(np.fft.fft(G)) / n_time_hb
        freqs = np.fft.fftshift(np.fft.fftfreq(n_time_hb, d=T_HB / n_time_hb))
        k_arr = np.rint(freqs * T_HB).astype(int)
        G_hb = np.zeros(2 * nf_s + 1, dtype=complex)
        for i, kk in enumerate(k_arr):
            if -nf_s <= kk <= nf_s:
                G_hb[nf_s + kk] = G_full[i]
        lhs = 1j * k_idx_s * omega * s_hb
        resid_c = lhs - G_hb
        y = np.zeros(2 * nf_s + 1)
        y[0] = float(resid_c[nf_s].real)
        for k in range(1, nf_s + 1):
            y[2 * k - 1] = float(resid_c[nf_s + k].real)
            y[2 * k] = float(resid_c[nf_s + k].imag)
        return y

    x0 = np.zeros(2 * nf_s + 1)
    x0[0] = s_dc_star
    sol_hb = root(residual, x0, method="hybr", tol=1e-12,
                  options={"maxfev": 500 * (2 * nf_s + 1)})
    r0 = np.linalg.norm(residual(x0))
    rf = np.linalg.norm(sol_hb.fun)
    if sol_hb.success and rf < 1e-6 * max(1.0, r0):
        s_hb_star_full = unpack(sol_hb.x)
        print(f"  HB-Newton converged:  ||F|| {r0:.3e} → {rf:.3e}")
    else:
        print(f"  HB-Newton FAILED  ({sol_hb.message.strip()})  "
              f"||F|| {r0:.3e} → {rf:.3e}")
        print(f"  → decoding the constant DC fallback s(t) = {s_dc_star:+.4f}")


# ── decode(t, s*(t)) over one T_HB ───────────────────────────────────
n_dec = 800
t_dec = np.linspace(0.0, T_HB, n_dec, endpoint=False)


def s_star_at(t):
    """Reconstruct s*(t) from the HB result (if present) or use the DC."""
    if s_hb_star_full is None:
        return s_dc_star
    weights = np.exp(1j * k_idx_s * omega * t)
    return float(np.real(np.dot(weights, s_hb_star_full)))


X_ROM = np.zeros((n, n_dec))
for i, tt in enumerate(t_dec):
    s_val = s_star_at(tt)
    dec = rom.decode(tt, np.array([s_val], dtype=complex)).real
    X_ROM[:, i] = c_star_at(tt) + dec

s_min = min(s_star_at(t) for t in t_dec)
s_max = max(s_star_at(t) for t in t_dec)
print(f"  s*(t) over T_HB:  min = {s_min:+.4f},  max = {s_max:+.4f},  "
      f"peak-to-peak {s_max - s_min:.4f}")


# ── Long DNS of the Rössler system to get the actual attractor ───────
t_transient = 100.0 * T_base
t_hold = 10.0 * T_base
t_end_dns = t_transient + t_hold
n_dns = 20000
t_eval_dns = np.linspace(0.0, t_end_dns, n_dns)

x0 = np.array([1.0, 1.0, 1.0])
print(f"\nLong DNS of Rössler at c = {c},  0 → {t_end_dns:.2f} = "
      f"{t_transient/T_base:.0f} + {t_hold/T_base:.0f} T_base ...")
sol_dns = solve_ivp(
    rossler_rhs, [0.0, t_end_dns], x0, args=(c,),
    method="RK45", t_eval=t_eval_dns, rtol=1e-10, atol=1e-12,
)
X_DNS = sol_dns.y
mask_tail = sol_dns.t >= t_transient
X_DNS_tail = X_DNS[:, mask_tail]
t_DNS_tail = sol_dns.t[mask_tail]
print(f"  nfev = {sol_dns.nfev},  tail = {mask_tail.sum()} samples "
      f"over t ∈ [{t_transient:.2f}, {t_end_dns:.2f}]")


# ── Plots ─────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# 3D phase portrait
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection="3d")
# 1T base orbit (grey dashed) for context
t_one = np.linspace(0.0, T_base, 300)
C_ref = _c_star_interp(t_one % T_HB)
ax.plot(C_ref[0], C_ref[1], C_ref[2], "--", color="0.6", lw=1.2,
        alpha=0.8, label="1T base orbit")
ax.plot(X_DNS_tail[0], X_DNS_tail[1], X_DNS_tail[2], "-",
        color="black", lw=1.0, alpha=0.6,
        label=f"FOM attractor  (tail {t_hold/T_base:.0f} T_base)")
ax.plot(np.append(X_ROM[0], X_ROM[0, 0]),
        np.append(X_ROM[1], X_ROM[1, 0]),
        np.append(X_ROM[2], X_ROM[2, 0]),
        "-", color="tab:red", lw=1.8,
        label=r"$c_*(t) + \mathrm{decode}(t, s^*(t))$")
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_zlabel(r"$z$", fontsize=12)
ax.legend(fontsize=10, loc="best")
ax.set_title(fr"Rössler at $c={c}$  ({variant} SSM)", fontsize=12)
fig.tight_layout()
outpath3d = f"results/decoded_orbit_vs_attractor_3D_{variant.replace('-','_')}"
fig.savefig(outpath3d + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath3d + ".pdf", bbox_inches="tight")
print(f"\nSaved -> {outpath3d}.png/.pdf")


# 3-panel time series
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
labels = [r"$x(t)$", r"$y(t)$", r"$z(t)$"]
# Wrap the ROM orbit modulo T_HB into the DNS window for direct overlay.
# Since it's periodic with period T_HB, tile it enough times to cover t_hold.
n_periods_show = int(np.ceil(t_hold / T_HB)) + 1
t_rom_tiled = np.concatenate([
    (t_transient + k * T_HB) + t_dec for k in range(n_periods_show)
])
X_ROM_tiled = np.concatenate([X_ROM for _ in range(n_periods_show)], axis=1)
mask_show = (t_rom_tiled >= t_transient) & (t_rom_tiled <= t_end_dns)

for j in range(3):
    ax = axes[j]
    ax.plot((t_DNS_tail - t_transient) / T_base, X_DNS_tail[j], "-",
            color="black", lw=1.0, alpha=0.7,
            label="FOM attractor")
    ax.plot((t_rom_tiled[mask_show] - t_transient) / T_base,
            X_ROM_tiled[j, mask_show], "--", color="tab:red", lw=1.4,
            alpha=0.85, label=r"decode$(t, s^*(t)) + c_*(t)$")
    ax.set_ylabel(labels[j], fontsize=14)
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(loc="best", fontsize=11)
axes[-1].set_xlabel(r"$(t - t_{\rm transient}) / T_{\rm base}$", fontsize=13)
fig.suptitle(fr"Rössler at $c={c}$: SSM's decoded 2T orbit "
             fr"vs FOM attractor tail  ({variant})", fontsize=12)
fig.tight_layout()
outpath_ts = f"results/decoded_orbit_vs_attractor_ts_{variant.replace('-','_')}"
fig.savefig(outpath_ts + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath_ts + ".pdf", bbox_inches="tight")
print(f"Saved -> {outpath_ts}.png/.pdf")

plt.show()


PETSc.COMM_WORLD.Barrier()
os._exit(0)
