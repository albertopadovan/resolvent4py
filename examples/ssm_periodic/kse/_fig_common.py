"""
Shared setup for the KSE-periodic presentation figures.

Kept in one place so the four figure scripts stay short and consistent
(matplotlib style, palette, cache load, ROM build, orbit interpolant,
2T-fixed-point locator).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import SpectralSubmanifoldROM


# ── matplotlib style (WCCM26, mirrors toy_model/generate_figures.py) ──────
def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.sans-serif": ["Computer Modern"],
            "font.size": 14,
            "text.usetex": True,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "legend.fontsize": 15,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


# ── Palette (same as toy_model) ───────────────────────────────────────────
C_1T = "black"        # T-periodic base flow  (spine of the ribbon)
C_2T = "#b2182b"      # brick red — 2T orbit / ROM predictions
C_SSM = "#f1c40f"     # goldenrod — SSM manifold
C_FOM = "black"       # full-order model (truth)
C_ROM = "#b2182b"     # ROM (SSM-based prediction)

# Output directory (WCCM26 presentation)
OUTDIR = "/Users/albertopadovan/Documents/Presentations/WCCM26/Figures/kse_periodic"


# ── Cache + ROM builder ───────────────────────────────────────────────────
def build_context(cache_path="data/ssm_cache.npz"):
    """
    Load ``data/ssm_cache.npz`` and return a bundle with everything the
    figure scripts need:

        cache, rom, c_star_at, s_star, T_HB, T_phys, rho_domain,
        n, n_pts, nu

    ``T_HB`` is what save_ssm.py calls ``T`` (= 2·T_phys, the HB period).
    """
    cache = np.load(cache_path)
    T_HB = float(cache["T"])
    T_phys = float(cache["T_orbit_phys"])
    rho_domain = float(cache["rho_domain"])
    n = int(cache["n"])
    n_pts = int(cache["n_pts"])
    nu = float(cache["nu"])
    omega = 2.0 * np.pi / T_HB

    # Pass the orbit-tangent neutral mode (index 0 of the deflated neutral
    # subspace, per save_ssm.py's convention) so ROM.neutral_project() is
    # available downstream — used by fig4 to strip the tangent component
    # of an off-manifold IC before FOM integration.
    v_neutral = None
    w_neutral = None
    if "V_neut_hb" in cache.files and "W_neut_hb" in cache.files:
        v_neutral = cache["V_neut_hb"][:, :, 0]
        w_neutral = cache["W_neut_hb"][:, :, 0]

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

    # Periodic-orbit interpolant (over the HB grid, which covers [0, T_HB)).
    C_periodic = cache["C_periodic"]
    time_orbit = cache["time_orbit"]
    time_ext = np.append(time_orbit, T_HB)
    C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
    _c_star_interp = interp1d(
        time_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
    )

    def c_star_at(t):
        return _c_star_interp(t % T_HB)

    # 2T-orbit fixed point:  real root of the scalar polynomial g(s) = 0
    # (r=1 chart-1 SSM).
    gs = cache["gs"]
    multiindices = cache["multiindices"]
    g_coeffs_low_to_high = np.array(
        [complex(gs[j, 0]) for j in range(len(multiindices))]
    )
    roots = np.roots(g_coeffs_low_to_high[::-1])
    real_roots = [
        z.real for z in roots
        if abs(z.imag) < 1e-3 * max(abs(z.real), 1.0)
           and 0.1 * rho_domain < abs(z) < 1.1 * rho_domain
    ]
    if not real_roots:
        raise RuntimeError(
            "No non-trivial real root of g(s)=0 inside rho_domain — the "
            "SSM cache may not be in the period-doubled regime."
        )
    s_star = float(abs(sorted(real_roots, key=lambda x: -abs(x))[0]))

    return dict(
        cache=cache, rom=rom, c_star_at=c_star_at, s_star=s_star,
        T_HB=T_HB, T_phys=T_phys, rho_domain=rho_domain,
        n=n, n_pts=n_pts, nu=nu,
    )


# ── Bilinear form B(q1, q2) for the KSE (copied from demonstrate_ssm.py) ──
# The KSE nonlinearity N(u) = -u u_x is quadratic, so
#     N(a + b) = N(a) + N(b) + 2 B(a, b)
# with the symmetric bilinear form ``B(a, b) = -1/2 · d/dx(u_a · u_b)``.
# In spectral coefficients:
def make_bilinear_form(n, n_pts):
    j_arr = np.arange(1, n + 1, dtype=float)
    j_rev = j_arr[::-1]
    half_N = n_pts / 2.0

    def _to_physical(c):
        spec_u = np.zeros(n_pts, dtype=complex)
        spec_ux = np.zeros(n_pts, dtype=complex)
        spec_u[1 : n + 1] = -1j * half_N * c
        spec_ux[1 : n + 1] = half_N * j_arr * c
        spec_u[n_pts - n : n_pts] = 1j * half_N * c[::-1]
        spec_ux[n_pts - n : n_pts] = half_N * j_rev * c[::-1]
        return ifft(spec_u), ifft(spec_ux)

    def B_fn(q1, q2):
        u1, u1x = _to_physical(q1)
        u2, u2x = _to_physical(q2)
        B_spec = fft(-0.5 * (u1 * u2x + u2 * u1x))
        out = 2j * B_spec[1 : n + 1] / n_pts
        if np.isrealobj(q1) and np.isrealobj(q2):
            return out.real
        return out

    return B_fn


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)
    return OUTDIR
