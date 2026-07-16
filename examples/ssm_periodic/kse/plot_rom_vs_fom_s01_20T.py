r"""
On-manifold ROM vs FOM perturbation comparison from s0 = 0.1,
integrated for 20 T_phys, constant-g SSM cache.

IC on-manifold:  x_0 = c_*(0) + decode(0, s_0).
Plots  δ_j(t) = a_j(t) − c*_j(t)  for j = 1, 2 (two subpanels),
ROM (red) vs FOM (black dashed).
"""
from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM
from spatial_operators import linear_eigenvalues, nonlinear_rhs


# ── Load SSM cache + build ROM directly ────────────────────────────────
cache = np.load("data/ssm_cache.npz")
T_HB = float(cache["T"])                    # = 2·T_phys
T_phys = float(cache["T_orbit_phys"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
nu = float(cache["nu"])
omega = 2.0 * np.pi / T_HB
rho_domain = float(cache["rho_domain"])
Lam0 = complex(cache["Lams"][0])

v_neutral = cache["V_neut_hb"][:, :, 0] if "V_neut_hb" in cache.files else None
w_neutral = cache["W_neut_hb"][:, :, 0] if "W_neut_hb" in cache.files else None

rom = SpectralSubmanifoldROM(
    multiindices=cache["multiindices"],
    Lams=cache["Lams"],
    gs=cache["gs"],
    PS=cache["PS_hb"],
    W=cache["W_hb"],
    conj_to_linear_dynamics=bool(cache["conj_to_linear_dynamics"]),
    omega=omega,
    v_neutral=v_neutral,
    w_neutral=w_neutral,
)


# c_*(t) via cubic-spline of the cached time-series
C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(t_ext, C_ext, axis=1, kind="cubic",
                          assume_sorted=True)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

s0 = np.array([1.0 + 0.0j], dtype=complex)
tau = 2.0 * T_phys / 4.0                        # initial time = 2T/5
t_end = tau + 60.0 * T_phys
print(f"|Λ| = {abs(Lam0.real):.4e}, T_phys = {T_phys:.4f}, "
      f"rho_domain = {rho_domain:.3f}")
print(f"tau (init t) = {tau:.4f} = 2T/5,  s0 = {s0[0].real},  "
      f"t_end = {t_end:.4f} = tau + 20 T_phys")


# ── Diagnostic: find real roots s* of g(s) = 0 (r = 1) ─────────────────
gs_arr = cache["gs"]
multiindices = cache["multiindices"]
g_coeffs = np.zeros(int(multiindices[:, 0].max()) + 1, dtype=complex)
for j in range(len(multiindices)):
    g_coeffs[int(multiindices[j, 0])] = complex(gs_arr[j, 0])
roots = np.roots(g_coeffs[::-1])                          # high-to-low
real_roots = sorted(
    [z.real for z in roots
     if abs(z.imag) < 1e-6 * max(abs(z.real), 1.0)],
    key=lambda x: abs(x),
)
print(f"\nEquilibria of g(s) = 0 (real, sorted by |s|):")
for j, r in enumerate(real_roots[:12]):
    tag = ""
    if abs(r) < 1e-8:
        tag = "  (trivial 0)"
    elif 0.05 * rho_domain < abs(r) < 1.5 * rho_domain:
        tag = "  <-- likely s* (inside rho_domain)"
    elif abs(r) > 1.5 * rho_domain:
        tag = "  (outside rho_domain, probably spurious)"
    print(f"   root {j}:  s = {r:+.4e}{tag}")
nontrivial = [r for r in real_roots if abs(r) > 1e-8]
if nontrivial:
    smallest_nontrivial = nontrivial[0]
    print(f"\ns0 = {s0[0].real}   "
          f"→  {'PAST' if abs(s0[0].real) > abs(smallest_nontrivial) else 'BELOW'} "
          f"the smallest-nontrivial |s*| = {abs(smallest_nontrivial):.4e}")


# ── ROM latent + reconstruction ────────────────────────────────────────
n_t = 8000
t_eval = np.linspace(tau, t_end, n_t)
sol = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [tau, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-14,
)
S_rom = sol.y[0]
print(f"|s_ROM(0)| = {abs(S_rom[0]):.6f}, "
      f"|s_ROM(t_end)| = {abs(S_rom[-1]):.6f}")

X_ROM = np.zeros((n, n_t))
CSTAR = np.zeros((n, n_t))
for i in range(n_t):
    CSTAR[:, i] = c_star_at(t_eval[i])
    X_ROM[:, i] = CSTAR[:, i] + rom.decode(t_eval[i], S_rom[i:i+1]).real
DELTA_ROM = X_ROM - CSTAR


# ── FOM from IC = c_*(τ) + decode(τ, s0), integrated via RK45 ─────────
delta_0 = rom.decode(tau, s0).real
x0_fom = c_star_at(tau) + delta_0
print(f"|delta_0| = {np.linalg.norm(delta_0):.4e}")


def kse_rhs(_t, q):
    r"""Full KSE RHS: L q + B(q, q)."""
    return lam * q + N_fn(q)


sol_fom = sp.integrate.solve_ivp(
    kse_rhs, [tau, t_end], x0_fom,
    method="RK45", t_eval=t_eval, rtol=1e-12, atol=1e-12,
)
print(f"FOM: RK45,  nfev = {sol_fom.nfev},  nsteps = {sol_fom.t.size}")
X_FOM_r = sol_fom.y                                    # (n, n_t)
DELTA_FOM = X_FOM_r - CSTAR


# ── Plot perturbations δ_1, δ_2 ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
labels = [r"$\delta_1(t)$", r"$\delta_2(t)$"]
for j in range(2):
    ax = axes[j]
    ax.plot(t_eval / T_phys, DELTA_ROM[j], "-", color="tab:red", lw=1.4,
            label=r"ROM")
    ax.plot(t_eval / T_phys, DELTA_FOM[j], "--", color="black", lw=1.0,
            alpha=0.9, label=r"FOM")
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_ylabel(labels[j], fontsize=14)
    ax.grid(alpha=0.2)
    if j == 0:
        ax.legend(loc="best", fontsize=11)
axes[-1].set_xlabel(r"$t / T_{\rm phys}$", fontsize=13)
fig.suptitle(
    fr"Constant-$g$ SSM perturbations $\delta_j = a_j - c^*_j$,  "
    fr"$s_0 = {s0[0].real}$,  $t_{{\rm end}} = 20\,T_{{\rm phys}}$",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("results/rom_vs_fom_s01_20T.png", dpi=140,
            bbox_inches="tight")
print("Saved -> results/rom_vs_fom_s01_20T.png")
plt.show()
