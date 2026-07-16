r"""
Phase-lock diagnostic: is the FOM on the 2T orbit at a drifting phase?

At each t in the saturated window, find the phase τ that minimises
    ||x(t) − c_2T(τ)||_2
over τ ∈ [0, T_2T).  Do this for both x_ROM(t) and x_FOM(t).  Then:

  · If the residual  ||x − c_2T(τ*)||  is at machine-precision, x IS on
    the 2T orbit.
  · If τ_ROM(t) is roughly constant (or trivially phase-locked to t) and
    τ_FOM(t) − τ_ROM(t) grows LINEARLY in t, the FOM has neutral-direction
    phase drift — confirming the hypothesis.

Plots residual, τ_FOM(t), τ_ROM(t), and their difference (with a linear
fit).  A linear drift on the last panel closes the case.
"""
# Bootstrap: this script was moved into periodic_g_manifold/.  Make the
# parent kse/ directory importable (for spatial_operators, kse_differential_equation,
# eigendecomp_kse) and chdir to it so relative data/... paths still resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from functools import partial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM
from spatial_operators import linear_eigenvalues, nonlinear_rhs


# ── Load caches ─────────────────────────────────────────────────────────
ssm = np.load("data/ssm_cache.npz")
T_HB = float(ssm["T"])
T_phys = float(ssm["T_orbit_phys"])
n = int(ssm["n"])
n_pts = int(ssm["n_pts"])
nu = float(ssm["nu"])
omega_HB_ssm = 2.0 * np.pi / T_HB

orbit_2T = np.load("data/periodic_orbit_2T.npz")
C_2T = orbit_2T["C"]                             # (n, 1025) samples
T_2T = float(orbit_2T["T"])

tau_grid = np.linspace(0.0, T_2T, C_2T.shape[1])
_c2T_interp = interp1d(tau_grid, C_2T, axis=1, kind="cubic",
                       assume_sorted=True)


def c_2T_at(tau):
    return _c2T_interp(tau % T_2T)


C_periodic = ssm["C_periodic"]
time_orbit = ssm["time_orbit"]
t_ext = np.append(time_orbit, T_HB)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
_c_star_interp = interp1d(t_ext, C_ext, axis=1, kind="cubic",
                          assume_sorted=True)


def c_star_at(t):
    return _c_star_interp(t % T_HB)


v_neut = ssm["V_neut_hb"][:, :, 0] if "V_neut_hb" in ssm.files else None
w_neut = ssm["W_neut_hb"][:, :, 0] if "W_neut_hb" in ssm.files else None
rom = SpectralSubmanifoldROM(
    multiindices=ssm["multiindices"], Lams=ssm["Lams"], gs=ssm["gs"],
    PS=ssm["PS_hb"], W=ssm["W_hb"],
    conj_to_linear_dynamics=bool(ssm["conj_to_linear_dynamics"]),
    omega=omega_HB_ssm, v_neutral=v_neut, w_neutral=w_neut,
)

# s_star from g(s) = 0
gs_arr = ssm["gs"]
mi = ssm["multiindices"]
g_coeffs = np.zeros(int(mi[:, 0].max()) + 1, dtype=complex)
for j in range(len(mi)):
    g_coeffs[int(mi[j, 0])] = complex(gs_arr[j, 0])
roots = np.roots(g_coeffs[::-1])
nontrivial = sorted(
    [z.real for z in roots
     if abs(z.imag) < 1e-6 * max(abs(z.real), 1.0) and abs(z) > 1e-8],
    key=lambda x: abs(x),
)
s_star = float(abs(nontrivial[0]))
print(f"s_star = {s_star:.4e},  T_2T = {T_2T:.6f},  T_phys = {T_phys:.4f}")


# ── Integrate ROM + FOM (short saturated window) ──────────────────────
lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

s0 = np.array([4.0 + 0.0j], dtype=complex)
tau_start = 2.0 * T_phys / 3.0
t_end = tau_start + 20.0 * T_phys
n_t = 4000
t_eval = np.linspace(tau_start, t_end, n_t)

print("Integrating ROM ...")
sol_rom = sp.integrate.solve_ivp(
    rom.latent_space_dynamics, [tau_start, t_end], s0,
    method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12,
)
S_rom = sol_rom.y[0]
X_ROM = np.zeros((n, n_t))
for i in range(n_t):
    X_ROM[:, i] = c_star_at(t_eval[i]) + rom.decode(t_eval[i], S_rom[i:i+1]).real

print("Integrating FOM ...")
delta_0 = rom.decode(tau_start, s0).real
x0_fom = c_star_at(tau_start) + delta_0


def kse_rhs(_t, q):
    return lam * q + N_fn(q)


sol_fom = sp.integrate.solve_ivp(
    kse_rhs, [tau_start, t_end], x0_fom,
    method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12,
)
X_FOM = sol_fom.y


# ── Phase-locate each state on the 2T orbit ───────────────────────────
def find_best_phase(x_target, n_coarse=400):
    """(τ*, err*) minimising ||x_target − c_2T(τ)||_2 over τ ∈ [0, T_2T)."""
    tau_search = np.linspace(0.0, T_2T, n_coarse, endpoint=False)
    errs = np.array(
        [np.linalg.norm(x_target - c_2T_at(t)) for t in tau_search]
    )
    k_min = int(np.argmin(errs))
    tau_c = tau_search[k_min]
    dtau = T_2T / n_coarse

    def err_fn(tau):
        return float(np.linalg.norm(x_target - c_2T_at(tau)))

    res = minimize_scalar(
        err_fn, bracket=(tau_c - dtau, tau_c, tau_c + dtau),
        method="brent", options={"xtol": 1e-12},
    )
    return float(res.x % T_2T), float(res.fun)


# Only phase-lock in the saturated window (skip the transient).
t_sat_start = tau_start + 8.0 * T_phys
mask = t_eval >= t_sat_start
idx_sat = np.where(mask)[0]
n_sat = idx_sat.size

print(f"\nPhase-locating {n_sat} samples in [{t_sat_start / T_phys:.1f}, "
      f"{t_end / T_phys:.1f}] T_phys ...")
tau_FOM = np.zeros(n_sat)
err_FOM = np.zeros(n_sat)
tau_ROM = np.zeros(n_sat)
err_ROM = np.zeros(n_sat)
for k, i in enumerate(idx_sat):
    tau_FOM[k], err_FOM[k] = find_best_phase(X_FOM[:, i])
    tau_ROM[k], err_ROM[k] = find_best_phase(X_ROM[:, i])

# Unwrap τ so they're monotonic in t (not jumping across the [0, T_2T) branch cut)
tau_FOM_u = np.unwrap(2.0 * np.pi * tau_FOM / T_2T) * T_2T / (2.0 * np.pi)
tau_ROM_u = np.unwrap(2.0 * np.pi * tau_ROM / T_2T) * T_2T / (2.0 * np.pi)
t_sat = t_eval[mask]

drift = tau_FOM_u - tau_ROM_u
drift -= drift[0]                                 # start at zero for readability
slope, intercept = np.polyfit(t_sat - t_sat[0], drift, 1)
print(f"\nPhase drift (FOM − ROM), linear fit:")
print(f"   slope = {slope:.4e}  (units of τ per unit t)")
print(f"   over final 12 T_phys, total drift = "
      f"{slope * 12 * T_phys:.4e}   "
      f"({slope * 12 * T_phys / T_2T * 100:+.3f} % of one T_2T)")
print(f"\nResiduals on the 2T orbit:")
print(f"   FOM: max = {err_FOM.max():.3e},  mean = {err_FOM.mean():.3e}")
print(f"   ROM: max = {err_ROM.max():.3e},  mean = {err_ROM.mean():.3e}")


# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

ax = axes[0]
ax.semilogy(t_sat / T_phys, err_FOM, "-", color="black", lw=1.2,
            label=r"FOM")
ax.semilogy(t_sat / T_phys, err_ROM, "-", color="tab:red", lw=1.2,
            label=r"ROM")
ax.set_ylabel(r"$\|x - c_{2T}(\tau^*)\|_2$", fontsize=12)
ax.grid(alpha=0.2, which="both")
ax.legend(fontsize=10, loc="best")
ax.set_title(r"Residual at best-matching phase on the 2T orbit  "
             r"(small $\Rightarrow$ trajectory IS on the orbit)",
             fontsize=11)

ax = axes[1]
ax.plot(t_sat / T_phys, tau_FOM_u, "-", color="black", lw=1.2,
        label=r"$\tau_{\rm FOM}(t)$")
ax.plot(t_sat / T_phys, tau_ROM_u, "-", color="tab:red", lw=1.2,
        label=r"$\tau_{\rm ROM}(t)$")
ax.set_ylabel(r"$\tau(t)$ (unwrapped)", fontsize=12)
ax.grid(alpha=0.2)
ax.legend(fontsize=10, loc="best")
ax.set_title(r"Best-matching phase on the 2T orbit", fontsize=11)

ax = axes[2]
ax.plot(t_sat / T_phys, drift, "-", color="tab:blue", lw=1.4,
        label=r"$\tau_{\rm FOM} - \tau_{\rm ROM}$  (shifted to 0 at start)")
ax.plot(t_sat / T_phys, slope * (t_sat - t_sat[0]) + intercept, "--",
        color="black", lw=0.9,
        label=fr"linear fit: slope = {slope:+.3e}/t")
ax.set_xlabel(r"$t / T_{\rm phys}$", fontsize=12)
ax.set_ylabel(r"$\tau_{\rm FOM} - \tau_{\rm ROM}$", fontsize=12)
ax.grid(alpha=0.2)
ax.legend(fontsize=10, loc="best")
ax.set_title(r"Phase drift FOM vs ROM  "
             r"(linear $\Rightarrow$ neutral-direction drift)",
             fontsize=11)

fig.tight_layout()
os.makedirs("results", exist_ok=True)
outpath = "results/phase_locate_2T.png"
fig.savefig(outpath, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {outpath}")
plt.show()
