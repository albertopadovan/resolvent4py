"""
Shared setup for the Rössler-periodic presentation figures.

Kept in one place so the four figure scripts stay short and consistent
(matplotlib style, palette, cache load, ROM build, orbit interpolant,
2T-fixed-point locator).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from resolvent4py.spectral_submanifold import (
    SpectralSubmanifoldROM,
    SpectralSubmanifoldROMPeriodicG,
)


# ── matplotlib style (WCCM26, mirrors kse/_fig_common.py) ────────────────
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


# ── Palette (matches kse/_fig_common.py) ─────────────────────────────────
C_1T = "black"        # T-periodic base flow
C_2T = "#b2182b"      # brick red — 2T orbit / ROM predictions
C_SSM = "#f1c40f"     # goldenrod — SSM manifold
C_FOM = "black"       # full-order model (truth)
C_ROM = "#b2182b"     # ROM (SSM-based prediction)

# Output directory (WCCM26 presentation)
OUTDIR = "/Users/albertopadovan/Documents/Presentations/WCCM26/Figures/rossler_periodic"


# ── Cache + ROM builder ──────────────────────────────────────────────────
def build_context(cache_path="data/ssm_cache_2T.npz"):
    """
    Load ``data/ssm_cache_2T.npz`` and return a bundle with everything the
    figure scripts need:

        cache, rom, c_star_at, s_star, T_HB, T_base, rho_domain, n, c

    Naming convention for the Rössler 2T pipeline (mirrors ``save_ssm_2T.py``):
        ``T_HB`` = 2·T_base   (the harmonic-balance / 2T embedding period)
        ``T_base``            (the underlying 1T base-flow period)
    """
    cache = np.load(cache_path)
    T_HB = float(cache["T"])              # = 2·T_base by construction
    T_base = float(cache["T_base"])
    rho_domain = float(cache["rho_domain"])
    n = int(cache["n"])
    c = float(cache["c"])
    omega = 2.0 * np.pi / T_HB

    v_neutral = None
    w_neutral = None
    if "V_neut_hb" in cache.files and "W_neut_hb" in cache.files:
        v_neutral = cache["V_neut_hb"][:, :, 0]
        w_neutral = cache["W_neut_hb"][:, :, 0]

    # Auto-dispatch: periodic-g cache stores 'gs_periodic'; constant-g
    # stores 'gs'.
    if "gs_periodic" in cache.files:
        variant = "periodic-g"
        rom = SpectralSubmanifoldROMPeriodicG(
            multiindices=cache["multiindices"],
            Lams=cache["Lams"],
            gs=cache["gs_periodic"],
            PS=cache["PS_hb"],
            W=cache["W_hb"],
            omega=omega,
        )
        rom.use_g_spline = False
    else:
        variant = "constant-g"
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

    # Periodic-orbit interpolant over the HB grid (covers [0, T_HB)).
    C_periodic = cache["C_periodic"]
    time_orbit = cache["time_orbit"]
    time_ext = np.append(time_orbit, T_HB)
    C_ext = np.column_stack([C_periodic, C_periodic[:, 0]])
    _c_star_interp = interp1d(
        time_ext, C_ext, axis=1, kind="cubic", assume_sorted=True,
    )

    def c_star_at(t):
        return _c_star_interp(t % T_HB)

    # 2T-orbit fixed point: real root of the scalar polynomial g(s) = 0
    # (r=1 chart-1 SSM), taken inside the SSM's convergence domain.
    # For periodic-g, use the DC (k=0) coefficient of each g_j(t) —
    # zeros of the DC polynomial are the T_HB-averaged fixed points.
    multiindices = cache["multiindices"]
    if variant == "periodic-g":
        gs_pg = cache["gs_periodic"]           # (n_terms, n_harm_g, r)
        nf_g = (gs_pg.shape[1] - 1) // 2
        g_coeffs_low_to_high = np.array(
            [complex(gs_pg[j, nf_g, 0]) for j in range(len(multiindices))]
        )
    else:
        gs = cache["gs"]
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
        T_HB=T_HB, T_base=T_base, rho_domain=rho_domain,
        n=n, c=c,
    )


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)
    return OUTDIR
