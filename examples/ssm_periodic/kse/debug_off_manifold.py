"""
Debug the off-manifold ROM-vs-truth run without rerunning the SSM solve.

Loads ``data/ssm_cache.npz`` (produced by ``save_ssm.py``) and reproduces
the physical-space encoder / decoder / neutral projection / latent
dynamics / perturbation RHS in pure numpy, then runs the off-manifold
case with a battery of diagnostics printed to stdout.
"""

import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from functools import partial

import matplotlib.pyplot as plt

from spatial_operators import linear_eigenvalues, nonlinear_rhs


# %% Load cache

cache = np.load("data/ssm_cache.npz")
PS_hb = cache["PS_hb"]              # (n_terms, n_harmonics, n) complex
W_hb = cache["W_hb"]                # (n_harmonics, n, r)       complex
V_neut_hb = cache["V_neut_hb"]      # (n_harmonics, n, n_neut)  complex
W_neut_hb = cache["W_neut_hb"]      # (n_harmonics, n, n_neut)  complex
multiindices = cache["multiindices"]  # (n_terms, r) int
Lams = cache["Lams"]                # (r,) complex
gs = cache["gs"]                    # (n_terms, r) complex
conj_to_linear = bool(cache["conj_to_linear_dynamics"])

nu = float(cache["nu"])
n = int(cache["n"])
n_pts = int(cache["n_pts"])
T = float(cache["T"])
nf = int(cache["nf"])
r = int(cache["r"])
m = int(cache["m"])
rho_domain = float(cache["rho_domain"])

C_periodic = cache["C_periodic"]
time_orbit = cache["time_orbit"]

omega = 2.0 * np.pi / T
n_harmonics = 2 * nf + 1
k_idx = np.arange(-nf, nf + 1)

print(f"Loaded SSM cache: n={n}, nf={nf}, r={r}, m={m}, "
      f"T={T:.4f}, rho_domain={rho_domain:.4f}")
print(f"n_terms={len(multiindices)}, n_neutral={V_neut_hb.shape[-1]}")
print(f"Lams = {Lams}")


# %% Physical-space encoder / decoder / projection (numpy replicas of
#    what lives in demonstrate_ssm.py)

def _ifft_weights(t):
    return np.exp(1j * k_idx * omega * t)


def decode_phys(s, t):
    s = np.asarray(s)
    coefs = np.array(
        [np.prod(s ** j) for j in multiindices], dtype=complex,
    )
    hb = np.einsum("i,ihj->hj", coefs, PS_hb)
    return (_ifft_weights(t) @ hb).real


def encode_phys(x0, t):
    """Simple physical encoder: ``s = W_phys(t)^H x0``.

    This is a LEFT INVERSE of the decoder only if ``W_phys(t)`` is
    biorthogonal to ``V_phys(t)`` at that time.  HB biorthogonality
    ``W^* V = I`` does *not* imply pointwise physical biorthogonality,
    so this encoder is only approximate unless the orbit is weakly
    unsteady (most HB energy in the ``k = 0`` block).
    """
    W_phys = np.einsum("h,hjr->jr", _ifft_weights(t), W_hb)
    return W_phys.conj().T @ x0


# Linear part of the SSM decoder: ps[1:r+1] are the r right eigenvectors
# (embedded in HB).  Reshape to (n_harmonics, n, r) to match W_hb.
V_hb = np.transpose(PS_hb[1:r + 1], (1, 2, 0))   # (n_harmonics, n, r)


def encode_phys_proper(x0, t):
    """Oblique encoder: ``s = (W_phys(t)^H V_phys(t))^{-1} W_phys(t)^H x0``.

    This IS a left inverse of the linearised physical decoder at time t.
    """
    w = _ifft_weights(t)
    V_phys = np.einsum("h,hjr->jr", w, V_hb)
    W_phys = np.einsum("h,hjr->jr", w, W_hb)
    Gram = W_phys.conj().T @ V_phys
    return np.linalg.solve(Gram, W_phys.conj().T @ x0)


# The tangent-to-orbit Floquet direction (exponent = 0 + 0j) is stored at
# column 0 of the neutral bases by construction in
# ``_compute_neutral_eigentriples`` (the loop starts with ``k = 0``).
# The other 10 columns are Floquet modes at shifts ``±kiω`` — in HB they
# are linearly independent, but after IFFT at a single ``t`` they collapse
# to the same physical direction.  Using only the σ = 0 mode avoids the
# rank deficiency entirely.
v_neut_hb = V_neut_hb[:, :, 0]   # (n_harmonics, n)
w_neut_hb = W_neut_hb[:, :, 0]


def neutral_project_phys(x0, t):
    """Remove the orbit-tangent direction at time ``t``.

    Uses only the ``σ = 0`` neutral Floquet mode (``v``, ``w``), IFFT-ed at
    time ``t``: ``x_proj = x - v · (w^H x) / (w^H v)``.
    """
    w_ifft = _ifft_weights(t)
    v_t = w_ifft @ v_neut_hb      # (n,) complex
    w_t = w_ifft @ w_neut_hb      # (n,) complex
    return x0 - v_t * (np.vdot(w_t, x0) / np.vdot(w_t, v_t))


def latent_space_dynamics(t, s):
    """``ds/dt = Lams * s + g(s)`` (pure numpy reproduction)."""
    ds = Lams * s
    if not conj_to_linear:
        _start = len(Lams) + 1
        J = multiindices[_start:]
        G = gs[_start:]
        monomials = np.prod(s[None, :] ** J, axis=1)
        ds = ds + monomials @ G
    return ds


# %% Truth RHS (KSE perturbation about the periodic orbit)

lam = linear_eigenvalues(n, nu)
N_fn = partial(nonlinear_rhs, n_pts=n_pts)

time_ext = np.append(time_orbit, T)
C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
c_star_interp = interp1d(
    time_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
)


def c_star_at(t):
    return c_star_interp(t % T)


def _evaluate_quadratic_term_numpy(q1, q2):
    """``B(q1, q2)`` for KSE — copy of the KuramotoSivashinskyPeriodic method."""
    j = np.arange(1, n + 1, dtype=float)
    half_N = n_pts / 2.0

    def _to_physical(c):
        spec_u = np.zeros(n_pts, dtype=complex)
        spec_ux = np.zeros(n_pts, dtype=complex)
        spec_u[1:n+1] = -1j * half_N * c
        spec_ux[1:n+1] = half_N * j * c
        spec_u[n_pts - n:n_pts] = 1j * half_N * c[::-1]
        spec_ux[n_pts - n:n_pts] = half_N * j[::-1] * c[::-1]
        return ifft(spec_u), ifft(spec_ux)

    u1, u1x = _to_physical(q1)
    u2, u2x = _to_physical(q2)
    B_spec = fft(-0.5 * (u1 * u2x + u2 * u1x))
    result = 2j * B_spec[1:n + 1] / n_pts
    if np.isrealobj(q1) and np.isrealobj(q2):
        return result.real
    return result


def perturbation_rhs(t, v):
    cs = c_star_at(t)
    return lam * v + 2.0 * _evaluate_quadratic_term_numpy(cs, v) + N_fn(v)


# %% ===================================================================
#    Diagnostics
#    ===================================================================

rng = np.random.default_rng(42)
scaling_off = 0.5


# -- (a) random physical IC, project out neutral, normalise -------------

x0_raw = rng.standard_normal(n)

# Biorthogonality of the σ = 0 neutral mode: w^H v at t = 0.
w0 = _ifft_weights(0.0)
v0_neut = w0 @ v_neut_hb
w0_neut = w0 @ w_neut_hb
bio_H = np.vdot(w0_neut, v0_neut)       # w^H v
bio_T = np.dot(w0_neut, v0_neut)        # w^T v

print("\n[a] Random IC + σ=0 neutral mode at t=0")
print(f"    ||x0_raw||           = {np.linalg.norm(x0_raw):.4e}")
print(f"    ||v||                = {np.linalg.norm(v0_neut):.4e}")
print(f"    ||w||                = {np.linalg.norm(w0_neut):.4e}")
print(f"    ||Im(v)||/||v||      = "
      f"{np.linalg.norm(v0_neut.imag) / np.linalg.norm(v0_neut):.4e}")
print(f"    w^H v (should = 1)   = {bio_H}")
print(f"    w^T v                = {bio_T}")

# Idempotence check: apply the projection twice.
x0_proj_complex = neutral_project_phys(x0_raw, 0.0)
x0_proj_twice = neutral_project_phys(x0_proj_complex, 0.0)
print(f"    ||x0_proj (complex)||= {np.linalg.norm(x0_proj_complex):.4e}")
print(f"    ||Im(x0_proj)||      = "
      f"{np.linalg.norm(x0_proj_complex.imag):.4e}")
print(f"    ||P(P x) - P x||     = "
      f"{np.linalg.norm(x0_proj_twice - x0_proj_complex):.4e}")
# After projection, w^H x should vanish.
print(f"    |w^H x0_raw|         = "
      f"{abs(np.vdot(w0_neut, x0_raw)):.4e}")
print(f"    |w^H x0_proj|        = "
      f"{abs(np.vdot(w0_neut, x0_proj_complex)):.4e}")

x0_proj = x0_proj_complex.real
x0_unit = x0_proj / np.linalg.norm(x0_proj)
print(f"    ||x0_raw - x0_proj.real|| = "
      f"{np.linalg.norm(x0_raw - x0_proj):.4e}")


# -- (b) encode / decode consistency checks -----------------------------

# The encoder is lossy (x -> s -> decode(s,0) is not identity because
# the SSM only captures the r-dim master subspace).  What should hold:
#   * encode(decode(s, 0), 0) is close to s                (for small s)
#   * decode(encode(x, 0), 0) is the manifold-nearest x    (projection)

s_test = np.array([0.3 * rho_domain, 0.3 * rho_domain], dtype=complex)
v_test = decode_phys(s_test, 0.0)
s_rt_simple = encode_phys(v_test, 0.0)
s_rt_proper = encode_phys_proper(v_test, 0.0)

w0 = _ifft_weights(0.0)
V_phys0 = np.einsum("h,hjr->jr", w0, V_hb)
W_phys0 = np.einsum("h,hjr->jr", w0, W_hb)
Gram_0 = W_phys0.conj().T @ V_phys0

print("\n[b] Biorthogonality and round-trip at t = 0")
print(f"    W_phys(0)^H V_phys(0) =\n"
      f"      {Gram_0[0]}\n      {Gram_0[1]}")
print(f"    ||Gram - I||_F = {np.linalg.norm(Gram_0 - np.eye(r)):.4e}")
print(f"    cond(Gram)     = {np.linalg.cond(Gram_0):.4e}")
print(f"    s_test               = {s_test}")
print(f"    s_rt  (W^H)          = {s_rt_simple}")
print(f"    s_rt  (oblique)      = {s_rt_proper}")
print(f"    |s - s_rt|  (W^H)     = "
      f"{np.linalg.norm(s_test - s_rt_simple):.4e}")
print(f"    |s - s_rt|  (oblique) = "
      f"{np.linalg.norm(s_test - s_rt_proper):.4e}")


# -- (c) manifold distance of the off-manifold IC -----------------------

s0_simple = encode_phys(x0_unit, 0.0)
s0_proper = encode_phys_proper(x0_unit, 0.0)
# Use the proper encoder for the off-manifold run.
scaling_factor = rho_domain / np.linalg.norm(s0_proper) * scaling_off
s0_off = s0_proper * scaling_factor
x0_off = x0_unit * scaling_factor

v_on_manifold = decode_phys(s0_off, 0.0)
off_residual = x0_off - v_on_manifold

print("\n[c] Off-manifold IC geometry (proper encoder)")
print(f"    scaling_off          = {scaling_off}")
print(f"    |s0 simple encoder|  = {np.linalg.norm(s0_simple):.4e}")
print(f"    |s0 proper encoder|  = {np.linalg.norm(s0_proper):.4e}")
print(f"    ||x0_off||           = {np.linalg.norm(x0_off):.4e}")
print(f"    |s0_off|             = {np.linalg.norm(s0_off):.4e} "
      f"(rho_domain={rho_domain:.4e})")
print(f"    ||decode(s0_off,0)|| = {np.linalg.norm(v_on_manifold):.4e}")
print(f"    ||x0_off - decode|| = {np.linalg.norm(off_residual):.4e}")
print(f"    off-manifold frac    = "
      f"{np.linalg.norm(off_residual) / np.linalg.norm(x0_off):.4e}")


# -- (d) neutral content of x0_off at t=0 ------------------------------

x0_reproj = neutral_project_phys(x0_off, 0.0).real
neutral_residue = x0_off - x0_reproj
print("\n[d] Neutral content of x0_off at t=0")
print(f"    ||x0_off - P_neut(x0_off)|| = "
      f"{np.linalg.norm(neutral_residue):.4e}")


# -- (e) integrate ROM and truth ---------------------------------------

n_periods = 10
t_end = n_periods * T
n_t = 2500
t_eval = np.linspace(0, t_end, n_t)

print(f"\n[e] Integrating over {n_periods} periods (t_end={t_end:.3f}) ...")

S_off = sp.integrate.solve_ivp(
    latent_space_dynamics,
    [0, t_end], s0_off, method="RK45", t_eval=t_eval,
    rtol=1e-12, atol=1e-12,
).y

Vrom = np.zeros((n, n_t))
for i in range(n_t):
    Vrom[:, i] = decode_phys(S_off[:, i], t_eval[i])

Vtruth = sp.integrate.solve_ivp(
    perturbation_rhs, [0, t_end], x0_off,
    method="Radau", t_eval=t_eval,
    rtol=1e-10, atol=1e-10,
).y

# Encode the truth trajectory at every eval time with BOTH encoders and
# compare with the ROM latent state.  If the simple encoder disagrees
# with the oblique encoder, that is direct evidence that ``W_phys(t)^H``
# is not a left inverse of the decoder in physical space.
S_truth_simple = np.zeros_like(S_off)
S_truth_proper = np.zeros_like(S_off)
for i in range(n_t):
    S_truth_simple[:, i] = encode_phys(Vtruth[:, i], t_eval[i])
    S_truth_proper[:, i] = encode_phys_proper(Vtruth[:, i], t_eval[i])

print(f"    max |s_rom - s_truth(simple)| = "
      f"{np.max(np.abs(S_off - S_truth_simple)):.4e}")
print(f"    max |s_rom - s_truth(proper)| = "
      f"{np.max(np.abs(S_off - S_truth_proper)):.4e}")
# For the plots we keep S_truth_encoded = the proper encoder's result.
S_truth_encoded = S_truth_proper

norm_truth = np.linalg.norm(Vtruth, axis=0)
norm_rom = np.linalg.norm(Vrom, axis=0)
norm_err = np.linalg.norm(Vtruth - Vrom, axis=0)

print(f"    ||v_truth||_0                = {norm_truth[0]:.4e}")
print(f"    ||v_truth||_end              = {norm_truth[-1]:.4e}")
print(f"    ||v_rom||_0                  = {norm_rom[0]:.4e}")
print(f"    ||v_rom||_end                = {norm_rom[-1]:.4e}")
print(f"    max ||v_truth - v_rom||      = {np.max(norm_err):.4e}")


# -- (f) plots ---------------------------------------------------------

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
ax[0].semilogy(t_eval, norm_truth, "k", label="Truth")
ax[0].semilogy(t_eval, norm_rom, "r--", label="ROM")
ax[0].semilogy(t_eval, norm_err, "b:", label="Truth − ROM")
ax[0].set_ylabel(r"$\|v\|$")
ax[0].legend()

ax[1].plot(t_eval, np.abs(S_off[0]), "r--", label=r"$|s_1^{\mathrm{ROM}}|$")
ax[1].plot(t_eval, np.abs(S_truth_encoded[0]), "k",
           label=r"$|W_\phi^H v_{\mathrm{truth}}|_1$")
ax[1].set_ylabel(r"$|s_1|$")
ax[1].legend()

for idx, c in zip([0, 1, 2], ["tab:blue", "tab:green", "tab:orange"]):
    ax[2].plot(t_eval, Vtruth[idx], color=c, lw=1.2,
               label=rf"$v_{{{idx+1}}}$ truth")
    ax[2].plot(t_eval, Vrom[idx], color=c, lw=1.2, ls="--",
               label=rf"$v_{{{idx+1}}}$ ROM")
ax[2].set_ylabel("Modes")
ax[2].set_xlabel(r"$t$")
ax[2].legend(ncol=3, fontsize=8)

plt.tight_layout()
plt.savefig("results/debug_off_manifold.png", dpi=200, bbox_inches="tight")
plt.show()
