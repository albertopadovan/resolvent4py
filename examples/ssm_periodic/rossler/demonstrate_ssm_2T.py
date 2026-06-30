"""
Step 4 of the Rössler **2T-periodic** (period-doubling) SSM workflow:
load the cached SSM and run the on-/off-manifold ROM vs truth
comparison + plotting.

Inputs:
    data/periodic_orbit.npz         (compute_periodic_orbit.py)
    data/eigendecomp_cache_2T.npz   (save_eigendecomp_2T.py)
    data/ssm_cache_2T.npz           (save_ssm_2T.py)

Outputs (PNG/PDF figures into results/):
    rom_vs_truth_2T          perturbation time-series, on-manifold IC
    rom_vs_truth_off_2T      perturbation time-series, off-manifold IC
    phase_portrait_on_2T     3D (x,y,z) phase space, on-manifold
    phase_portrait_off_2T    3D (x,y,z) phase space, off-manifold

No PETSc / MPI — purely numpy + scipy + matplotlib.
"""

import os

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM
from rossler_rhs import perturbation_linear_action, quadratic_bilinear


res_path = "results/"


def savefig(fig, name):
    os.makedirs(res_path, exist_ok=True)
    fig.savefig(res_path + name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(res_path + name + ".pdf", bbox_inches="tight")
    print(f"Saved to {res_path}{name}.png/.pdf")


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)


# %% Parameters that aren't in the cache

n_periods = 50  # in units of T (= 2*T_base) → 10 base-period cycles
n_t = 2000
rtol_rom = 1e-12
atol_rom = 1e-12
rtol_truth = 1e-10
atol_truth = 1e-10
s0_fraction = 0.3
scaling_off = 0.3

figsize = (7, 5)
clr_truth = "#2D3142"
clr_rom = "#E85D04"
labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]


# %% Load SSM cache

cache = np.load("data/ssm_cache_2T.npz")
PS_hb = cache["PS_hb"]
W_hb = cache["W_hb"]
V_neut_hb = cache["V_neut_hb"]
W_neut_hb = cache["W_neut_hb"]
multiindices = cache["multiindices"]
Lams = cache["Lams"]
gs = cache["gs"]
conj_to_linear = bool(cache["conj_to_linear_dynamics"])

c = float(cache["c"])
n = int(cache["n"])
T = float(cache["T"])  # = 2 * T_base
T_base = float(cache["T_base"])
nf = int(cache["nf"])
nfb = int(cache["nfb"])
r = int(cache["r"])
m = int(cache["m"])
rho_domain = float(cache["rho_domain"])

C_periodic = cache["C_periodic"]  # tiled over [0, 2T_base)
time_orbit = cache["time_orbit"]

omega_2T = 2.0 * np.pi / T  # fundamental of the 2T SSM
omega_base = 2.0 * np.pi / T_base

print(
    f"Loaded SSM cache (2T): c={c}, n={n}, T_base={T_base:.4f}, "
    f"T=2*T_base={T:.4f}, nf={nf}, nfb={nfb}, r={r}, m={m}, "
    f"rho_domain={rho_domain:.4f}, n_terms={len(multiindices)}"
)
print(f"  Master Floquet exponent(s) (real-valued in 2T basis): {Lams}")


# %% Build the serial numpy-only ROM

rom = SpectralSubmanifoldROM(
    multiindices=multiindices,
    Lams=Lams,
    gs=gs,
    PS=PS_hb,
    W=W_hb,
    conj_to_linear_dynamics=conj_to_linear,
    omega=omega_2T,
    v_neutral=V_neut_hb[:, :, 0],
    w_neutral=W_neut_hb[:, :, 0],
)


# %% Truth RHS — Rössler perturbation around the *T-periodic* orbit

# The cached C_periodic spans [0, 2T_base) but is just the T_base-periodic
# orbit tiled twice; interpolating with period 2T_base or T_base gives
# the same answer.
time_ext = np.append(time_orbit, T)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_ext,
    C_ext,
    axis=1,
    kind="cubic",
    assume_sorted=True,
)


def c_star_at(t):
    return c_star_interp(t % T)


def perturbation_rhs(t, v):
    """v̇ = A(t)v + B(v,v) — Rössler linearised about c*(t)."""
    cs = c_star_at(t)
    return perturbation_linear_action(cs, v, c) + quadratic_bilinear(v, v)


# %% Common grids

t_end = n_periods * T
t_eval = np.linspace(0, t_end, n_t)

os.makedirs(res_path, exist_ok=True)


# %% On-manifold IC

s0 = s0_fraction * rho_domain * np.array([1.0], dtype=complex)

print(f"\nROM vs Truth (on-manifold), t_end = {t_end:.3f}")
print("  Integrating ROM ...")
S = sp.integrate.solve_ivp(
    rom.latent_space_dynamics,
    [0, t_end],
    s0,
    method="RK45",
    t_eval=t_eval,
    rtol=rtol_rom,
    atol=atol_rom,
).y

Vapp = np.zeros((n, n_t))
for i in range(n_t):
    Vapp[:, i] = rom.decode(t_eval[i], S[:, i])

v0 = rom.decode(0.0, s0)
print("  Integrating truth ...")
Vtruth = sp.integrate.solve_ivp(
    perturbation_rhs,
    [0, t_end],
    v0,
    method="RK45",
    t_eval=t_eval,
    rtol=rtol_truth,
    atol=atol_truth,
).y

fig, axes = plt.subplots(n, 1, sharex=True, figsize=figsize)
for i in range(n):
    axes[i].plot(t_eval, Vtruth[i], color=clr_truth, lw=1.5, label="Truth")
    axes[i].plot(t_eval, Vapp[i], color=clr_rom, ls="--", lw=1.5, label="ROM")
    axes[i].set_ylabel(labels_v[i])
axes[0].legend()
axes[0].set_title(
    rf"2T SSM on-manifold (c={c}, $\rho={s0_fraction:.2f}\rho_{{\rm dom}}$)"
)
axes[-1].set_xlabel(r"Time $t$")
plt.tight_layout()
savefig(fig, "rom_vs_truth_2T")
plt.show()


# %% Off-manifold IC

rng = np.random.default_rng(42)
x0_off = rng.standard_normal(n)
x0_off = rom.neutral_project(0.0, x0_off).real
x0_off /= np.linalg.norm(x0_off)

s0_off = rom.encode(0.0, x0_off)
scaling_factor = rho_domain / np.linalg.norm(s0_off) * scaling_off
s0_off *= scaling_factor
x0_off *= scaling_factor

print(f"\nROM vs Truth (off-manifold)")
print(
    f"  |s0_off| = {np.linalg.norm(s0_off):.3f}  "
    f"(rho_domain = {rho_domain:.3f}, ratio = "
    f"{np.linalg.norm(s0_off) / rho_domain:.2f})"
)
print("  Integrating ROM ...")
S_off = sp.integrate.solve_ivp(
    rom.latent_space_dynamics,
    [0, t_end],
    s0_off,
    method="RK45",
    t_eval=t_eval,
    rtol=rtol_rom,
    atol=atol_rom,
).y
S_abs_max = np.max(np.abs(S_off))
print(
    f"  max |s(t)| during ROM integration = {S_abs_max:.3f}  "
    f"(rho_domain = {rho_domain:.3f})"
)
if S_abs_max > rho_domain:
    print(
        "  ⚠ latent trajectory left the convergence domain — "
        "polynomial truncation is unreliable here."
    )

Vapp_off = np.zeros((n, n_t))
for i in range(n_t):
    Vapp_off[:, i] = rom.decode(t_eval[i], S_off[:, i])

print("  Integrating truth ...")
Vtruth_off = sp.integrate.solve_ivp(
    perturbation_rhs,
    [0, t_end],
    x0_off,
    method="RK45",
    t_eval=t_eval,
    rtol=rtol_truth,
    atol=atol_truth,
).y

fig, axes = plt.subplots(n, 1, sharex=True, figsize=figsize)
for i in range(n):
    axes[i].plot(t_eval, Vtruth_off[i], color=clr_truth, lw=1.5, label="Truth")
    axes[i].plot(
        t_eval, Vapp_off[i], color=clr_rom, ls="--", lw=1.5, label="ROM"
    )
    axes[i].set_ylabel(labels_v[i])
axes[0].legend()
axes[0].set_title(rf"2T SSM off-manifold (c={c})")
axes[-1].set_xlabel(r"Time $t$")
plt.tight_layout()
savefig(fig, "rom_vs_truth_off_2T")
plt.show()


# %% 3D phase portrait of the full state (orbit + perturbation)


def orbit_at(t_arr):
    return c_star_interp(t_arr % T)


C_orbit_eval = orbit_at(t_eval)
X_truth = C_orbit_eval + Vtruth
X_rom = C_orbit_eval + Vapp
X_truth_off = C_orbit_eval + Vtruth_off
X_rom_off = C_orbit_eval + Vapp_off

t_one_period = np.linspace(0, T_base, 200)
C_ref = c_star_interp(t_one_period)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    C_ref[0],
    C_ref[1],
    C_ref[2],
    color="0.6",
    lw=1,
    ls="--",
    label="periodic orbit (T)",
)
ax.plot(
    X_truth[0],
    X_truth[1],
    X_truth[2],
    color=clr_truth,
    lw=1.0,
    label="Truth (on)",
)
ax.plot(
    X_rom[0],
    X_rom[1],
    X_rom[2],
    color=clr_rom,
    lw=1.0,
    ls="--",
    label="ROM (on)",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(fontsize=8)
ax.set_title(f"Rössler 2T SSM — on-manifold (c={c})")
plt.tight_layout()
savefig(fig, "phase_portrait_on_2T")
plt.show()

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    C_ref[0],
    C_ref[1],
    C_ref[2],
    color="0.6",
    lw=1,
    ls="--",
    label="periodic orbit (T)",
)
ax.plot(
    X_truth_off[0],
    X_truth_off[1],
    X_truth_off[2],
    color=clr_truth,
    lw=1.0,
    label="Truth (off)",
)
ax.plot(
    X_rom_off[0],
    X_rom_off[1],
    X_rom_off[2],
    color=clr_rom,
    lw=1.0,
    ls="--",
    label="ROM (off)",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(fontsize=8)
ax.set_title(f"Rössler 2T SSM — off-manifold (c={c})")
plt.tight_layout()
savefig(fig, "phase_portrait_off_2T")
plt.show()

plt.close("all")
os._exit(0)
