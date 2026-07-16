r"""
Radius of convergence of the ROM's LATENT-space dynamics
   ds/dt = g(t, s) = Σ_j g_j(t) s^j,

separately from the manifold-parameterisation radius that
``SSM.estimate_convergence_radius()`` returns (that one is built from
``||P_j||``, not ``||g_j||``).

Runs against either cache — auto-dispatches on the ``.npz`` keys:
    data/ssm_cache_2T.npz            → constant-g:  |g_j|
    data/ssm_cache_2T_periodic_g.npz → periodic-g:  ||g_j(t)||_∞

Outputs:
  · per-order coefficient magnitude sequence,
  · fitted geometric-decay slope → Cauchy–Hadamard radius R_g,
  · real fixed point s* of the DC polynomial g^DC(s) = 0,
  · verdict:  s* vs R_g.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc


# ── Which cache to inspect ────────────────────────────────────────────
if len(sys.argv) > 1:
    cache_path = sys.argv[1]
else:
    cache_path = "data/ssm_cache_2T_periodic_g.npz"

if not os.path.exists(cache_path):
    raise FileNotFoundError(cache_path)

cache = np.load(cache_path)
multiindices = cache["multiindices"]              # (n_terms, r)
degs = multiindices[:, 0].astype(int)
max_deg = int(degs.max())

if "gs_periodic" in cache.files:
    variant = "periodic-g"
    gs_arr = cache["gs_periodic"][:, :, 0]        # (n_terms, n_harm_g)
    n_harm_g = gs_arr.shape[1]
    nf_g = (n_harm_g - 1) // 2
    T_HB = float(cache["T"])
elif "gs" in cache.files:
    variant = "constant-g"
    gs_arr = cache["gs"][:, 0]                    # (n_terms,) — scalar per multi-index
    T_HB = float(cache["T"])
else:
    raise KeyError(
        f"{cache_path}: neither 'gs' nor 'gs_periodic' found"
    )

print(f"{variant} latent-space dynamics from {cache_path}")
print(f"  T_HB = {T_HB:.6f},  max polynomial order = {max_deg},  "
      f"n_terms = {len(degs)}")


# ── Per-order coefficient magnitude ‖g_j‖ ─────────────────────────────
# constant-g:  ‖g_j‖ = |g_j|          (single scalar)
# periodic-g:  ‖g_j‖ = max_t |g_j(t)| = ‖IDFT(g_j,hb)‖_∞
if variant == "periodic-g":
    n_t = 4 * n_harm_g                            # oversample IDFT
    t_grid = np.linspace(0.0, T_HB, n_t, endpoint=False)
    omega = 2.0 * np.pi / T_HB
    k_idx = np.arange(-nf_g, nf_g + 1)
    weights = np.exp(1j * np.outer(t_grid, k_idx) * omega)   # (n_t, n_harm_g)
    g_j_of_t = weights @ gs_arr.T                            # (n_t, n_terms)
    # For real Rössler around a real orbit, g_j(t) is real (Hermitian
    # symmetric HB storage).  Take Re() to be safe.
    norms_sup = np.max(np.abs(g_j_of_t.real), axis=0)        # (n_terms,)
    norms_L2 = np.sqrt(np.mean(g_j_of_t.real ** 2, axis=0))   # (n_terms,)
else:
    norms_sup = np.abs(gs_arr)
    norms_L2 = norms_sup


# ── Collapse across multi-indices of the same polynomial degree ──────
# For r=1 there's a unique multi-index per degree so this is pass-through.
per_deg_sup = np.zeros(max_deg + 1)
per_deg_L2  = np.zeros(max_deg + 1)
for k, d in enumerate(degs):
    per_deg_sup[d] = max(per_deg_sup[d], norms_sup[k])
    per_deg_L2[d]  = max(per_deg_L2[d],  norms_L2[k])


# ── Print per-order table ─────────────────────────────────────────────
print(f"\n{'deg j':>6s}  {'‖g_j‖_∞ (sup_t)':>18s}  {'‖g_j‖_L2 (rms_t)':>18s}")
for d in range(min(max_deg + 1, 25)):
    print(f"  {d:>4d}    {per_deg_sup[d]:>16.4e}    {per_deg_L2[d]:>16.4e}")
if max_deg >= 25:
    print(f"  ... (skipping {max_deg - 24} rows) ...")
    for d in range(max_deg - 4, max_deg + 1):
        print(f"  {d:>4d}    {per_deg_sup[d]:>16.4e}    {per_deg_L2[d]:>16.4e}")


# ── Cauchy–Hadamard fit ───────────────────────────────────────────────
# R = 1 / lim_sup |a_j|^{1/j};  equivalently, slope of log|a_j| vs j.
def fit_R(norms, tag, fit_window=None):
    js = np.arange(len(norms))
    valid = norms > 0
    if valid.sum() < 5:
        print(f"  [{tag}] not enough non-zero coefficients");  return None
    logn = np.log10(norms[valid])
    js_v = js[valid]
    if fit_window is None:
        # skip j=0 (which is 0 anyway) and any low-order noise floor;
        # focus on the geometric tail.
        j_lo = max(4, len(js_v) // 4)
        j_hi = len(js_v) - 2
        m = (js_v >= j_lo) & (js_v <= j_hi)
    else:
        m = (js_v >= fit_window[0]) & (js_v <= fit_window[1])
    if m.sum() < 3:
        print(f"  [{tag}] fit window too narrow");  return None
    slope, intercept = np.polyfit(js_v[m], logn[m], 1)
    R = 10.0 ** (-slope)
    print(f"  [{tag}] fit on j ∈ [{js_v[m][0]}, {js_v[m][-1]}]:  "
          f"slope = {slope:.4f},  R = {R:.4f}")
    return R, slope, intercept, js_v[m]


print(f"\nCauchy–Hadamard fit → R_g   (latent-dynamics radius):")
R_sup = fit_R(per_deg_sup, "sup_t norm")
R_L2 = fit_R(per_deg_L2, "L2_t norm")


# ── Real fixed point s* of the DC polynomial g^DC(s) = Σ_j g_j^DC s^j ─
# For periodic-g we take the DC coefficient of each g_j(t); for
# constant-g the coefficient IS the DC.
g_dc = np.zeros(max_deg + 1, dtype=complex)
for k, d in enumerate(degs):
    if variant == "periodic-g":
        g_dc[d] += complex(gs_arr[k, nf_g])       # DC = k=0 harmonic
    else:
        g_dc[d] += complex(gs_arr[k])

print(f"\nDC polynomial  g^DC(s) = Σ_j g_j^DC · s^j:")
for d in range(min(max_deg + 1, 12)):
    if abs(g_dc[d]) > 0:
        print(f"   s^{d:>2d}:  {g_dc[d].real:+.4e}{g_dc[d].imag:+.4e}j")

roots = np.roots(g_dc[::-1])
real_roots = sorted(
    [z.real for z in roots
     if abs(z.imag) < 1e-6 * max(abs(z.real), 1.0)],
    key=lambda x: abs(x),
)
print(f"\nReal roots of g^DC(s) = 0 (sorted by |s|):")
for k, r in enumerate(real_roots[:8]):
    tag = "  (trivial 0)" if abs(r) < 1e-8 else ""
    print(f"   root {k}:  s = {r:+.4e}{tag}")

nontrivial = [r for r in real_roots if abs(r) > 1e-8]
s_star = nontrivial[0] if nontrivial else None
if s_star is not None:
    print(f"\ns* (smallest-|·| nontrivial root) = {s_star:+.4e}")


# ── Stability of s*: sign of g'(s*)  ─────────────────────────────────
# ds/dt = g(s) linearised at s*:  d(δs)/dt = g'(s*) · δs.
#   g'(s*) < 0  → stable (attractor)
#   g'(s*) > 0  → unstable (repellor)
# For periodic-g the average stability is set by g^DC'(s*); the pointwise
# linearisation g'(t, s*) = Σ_j j · g_j(t) · s*^{j-1} tells whether the
# fixed point is unstable AT SOME PHASES (even if attracting on average).
if s_star is not None:
    # DC derivative: g_DC'(s*) = Σ_{j≥1} j · g_j^DC · s*^{j-1}
    g_dc_prime_at_s_star = sum(
        j * g_dc[j] * (s_star ** (j - 1)) for j in range(1, max_deg + 1)
    ).real
    print(f"\ng^DC'(s*) = {g_dc_prime_at_s_star:+.4e}   "
          f"({'UNSTABLE' if g_dc_prime_at_s_star > 0 else 'stable'} "
          f"under the DC/averaged ROM)")
    if abs(g_dc_prime_at_s_star) > 1e-30:
        char_time = 1.0 / abs(g_dc_prime_at_s_star)
        print(f"          |1 / g^DC'(s*)|  = {char_time:.4e}   "
              f"(characteristic {'e-fold' if g_dc_prime_at_s_star > 0 else 'decay'} time)")

    if variant == "periodic-g":
        # Pointwise:  g'(t, s*) at 128 phases
        omega_pt = 2.0 * np.pi / T_HB
        n_phase = 128
        t_phases = np.linspace(0.0, T_HB, n_phase, endpoint=False)
        k_idx_pt = np.arange(-nf_g, nf_g + 1)
        weights_pt = np.exp(1j * np.outer(t_phases, k_idx_pt) * omega_pt)
        # Per-order derivative coefficient in the polynomial in s:
        # d/ds [g_j(t) · s^j] = j · g_j(t) · s^{j-1}
        g_prime_at_phase = np.zeros(n_phase)
        for k_term, d in enumerate(degs):
            if d < 1:
                continue
            g_j_t = np.real(weights_pt @ gs_arr[k_term])   # (n_phase,)
            g_prime_at_phase += d * g_j_t * (s_star ** (d - 1))
        g_pt_min = float(g_prime_at_phase.min())
        g_pt_max = float(g_prime_at_phase.max())
        g_pt_mean = float(g_prime_at_phase.mean())
        print(f"\nPointwise g'(t, s*) over t ∈ [0, T_HB):")
        print(f"  min  = {g_pt_min:+.4e}")
        print(f"  mean = {g_pt_mean:+.4e}   "
              f"(≡ g^DC'(s*) = {g_dc_prime_at_s_star:+.4e}, "
              f"gap = {g_pt_mean - g_dc_prime_at_s_star:.2e})")
        print(f"  max  = {g_pt_max:+.4e}")
        if g_pt_min > 0:
            print("  → UNSTABLE at every phase — s* is a pointwise repellor.")
        elif g_pt_max <= 0:
            print("  → stable at every phase.")
        else:
            frac_unstable = float(np.mean(g_prime_at_phase > 0))
            print(f"  → mixed:  g'(t, s*) > 0 for {100*frac_unstable:.1f}% "
                  f"of the phase.  Stability is set by the DC/Floquet "
                  f"average (see g^DC'(s*) above).")


# ── Verdict ───────────────────────────────────────────────────────────
if R_sup is not None:
    R_g = R_sup[0]
    print(f"\nVerdict:")
    print(f"  R_g (sup_t norm) = {R_g:.4f}")
    if s_star is not None:
        ratio = abs(s_star) / R_g
        print(f"  |s*|             = {abs(s_star):.4f}")
        print(f"  |s*| / R_g       = {ratio:.4f}")
        if ratio < 0.3:
            print("  → target well inside convergence radius.  "
                  "Blow-up at s ≈ s* would not be explained by polynomial "
                  "truncation; look elsewhere.")
        elif ratio < 0.8:
            print("  → target moderately inside R_g.  Truncation error "
                  "at s* is  (|s*|/R_g)^(m+1)  which may or may not be "
                  "significant depending on m.")
        else:
            print("  → target AT or PAST R_g.  Polynomial truncation "
                  "error at s ≈ s* is O(1); this is exactly the "
                  "regime where the ROM blows up when the trajectory "
                  "arrives at s*.")


# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
js = np.arange(max_deg + 1)
valid = per_deg_sup > 0
ax.semilogy(js[valid], per_deg_sup[valid], "o", color="tab:blue",
            markersize=6, label=r"$\|g_j\|_{\infty}$ (sup over $t$)")
if variant == "periodic-g":
    ax.semilogy(js[valid], per_deg_L2[valid], "s", color="tab:orange",
                markersize=5, alpha=0.7,
                label=r"$\|g_j\|_{L^2_t}$ (rms over $t$)")
    ax.semilogy(js[valid], np.abs(g_dc[valid.nonzero()[0]]), "^",
                color="0.5", markersize=5, alpha=0.7,
                label=r"$|g_j^{\rm DC}|$")

# Fit line(s)
if R_sup is not None:
    R_g, slope, intercept, js_fit = R_sup
    fit_line = 10.0 ** (slope * js + intercept)
    ax.semilogy(js, fit_line, "--", color="tab:blue", lw=1.2, alpha=0.5,
                label=fr"fit $\|g_j\|_\infty$: $R_g = {R_g:.3f}$")

if s_star is not None:
    ax.axvline(0, color="none")  # placeholder; instead mark s* on right axis
    # Show s* as a vertical guide only when |s*| lies within our j-axis range;
    # here we instead print it in the title.
ax.set_xlabel(r"polynomial order $j$", fontsize=13)
ax.set_ylabel(r"coefficient magnitude", fontsize=13)
title = (rf"{variant} ROM: latent-dynamics coefficient decay")
if s_star is not None and R_sup is not None:
    title += (rf"   ($|s^*| = {abs(s_star):.3f}$,  "
              rf"$R_g = {R_sup[0]:.3f}$,  "
              rf"$|s^*|/R_g = {abs(s_star)/R_sup[0]:.3f}$)")
ax.set_title(title, fontsize=11)
ax.grid(True, which="both", ls=":", lw=0.5)
ax.legend(loc="best", fontsize=10)
fig.tight_layout()

os.makedirs("results", exist_ok=True)
outpath = f"results/check_g_convergence_{variant.replace('-','_')}"
fig.savefig(outpath + ".png", dpi=140, bbox_inches="tight")
fig.savefig(outpath + ".pdf", bbox_inches="tight")
print(f"\nSaved -> {outpath}.png/.pdf")


# ── Plot g(s) vs s ────────────────────────────────────────────────────
# Range: bracket [0, 1.4 · max(|s*|, R_g)] so both zero crossings and the
# blow-up neighbourhood are visible.
s_scale = max(
    abs(s_star) if s_star is not None else 0.0,
    R_sup[0] if R_sup is not None else 0.0,
    1.0,
)
s_lo, s_hi = -1.4 * s_scale, 1.4 * s_scale
s_grid = np.linspace(s_lo, s_hi, 2000)


def eval_g_at(gs_row, s_arr, t=None, omega=None, k_idx=None, nf_g=None):
    """Return g(s) or g(t, s) at (t, s) given a single multi-index's HB row.
    For constant-g pass `t=None` (uses the scalar coefficient).
    For periodic-g pass t and the omega/k_idx/nf_g context.
    """
    if t is None:
        # constant-g: gs_row is a scalar
        return complex(gs_row).real
    # periodic-g: reconstruct at t via IDFT
    w = np.exp(1j * k_idx * omega * t)
    return float(np.real(np.dot(w, gs_row)))


# Build g(s) at chosen phases
if variant == "periodic-g":
    omega = 2.0 * np.pi / T_HB
    k_idx = np.arange(-nf_g, nf_g + 1)
    # Sample at 128 phases for envelope
    n_phase = 128
    t_phases = np.linspace(0.0, T_HB, n_phase, endpoint=False)
    weights_phase = np.exp(1j * np.outer(t_phases, k_idx) * omega)  # (n_phase, n_harm_g)
    # gs_arr shape (n_terms, n_harm_g).  g_j at each phase.
    g_j_of_phase = weights_phase @ gs_arr.T                        # (n_phase, n_terms)
    g_j_of_phase = g_j_of_phase.real                                # (n_phase, n_terms)
    # DC:
    g_j_dc = np.array([g_dc[d].real for d in degs])                # (n_terms,)
    # For every s: sum over multi-indices j of g_j(t)·s^degs[j]
    G_envelope = np.zeros((n_phase, len(s_grid)))
    G_dc = np.zeros(len(s_grid))
    for k_term, d in enumerate(degs):
        G_envelope += np.outer(g_j_of_phase[:, k_term], s_grid ** d)
        G_dc += g_j_dc[k_term] * (s_grid ** d)
    G_min = G_envelope.min(axis=0)
    G_max = G_envelope.max(axis=0)
else:
    g_coeffs = np.zeros(max_deg + 1, dtype=float)
    for k_term, d in enumerate(degs):
        g_coeffs[d] += float(np.real(complex(gs_arr[k_term])))
    G_dc = np.polyval(g_coeffs[::-1], s_grid)                       # (n_s,)
    G_min = G_max = None


fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# Clip y-axis to a viewable range so blow-up doesn't crush the interesting
# part of the curve.  Anchor to |G_dc| in the well-behaved band [-|s*|, |s*|].
mask_ok = np.abs(s_grid) <= (abs(s_star) if s_star is not None else s_scale/2)
y_ref = float(np.percentile(np.abs(G_dc[mask_ok]), 95)) if mask_ok.sum() else 1.0
y_clip = max(3.0 * y_ref, 1e-3)

if variant == "periodic-g":
    ax2.fill_between(s_grid, np.clip(G_min, -y_clip, y_clip),
                     np.clip(G_max, -y_clip, y_clip),
                     color="tab:orange", alpha=0.25,
                     label=r"$[\min_t g(t,s),\, \max_t g(t,s)]$")

ax2.plot(s_grid, np.clip(G_dc, -y_clip, y_clip), "-",
         color="tab:blue", lw=1.6,
         label=(r"$g^{\rm DC}(s)$" if variant == "periodic-g"
                else r"$g(s)$"))
ax2.axhline(0.0, color="0.5", lw=0.7, ls="--")
ax2.axvline(0.0, color="0.5", lw=0.7, ls="--")

# Mark s* and ±R_g
if s_star is not None:
    ax2.axvline(s_star, color="tab:green", lw=1.4, alpha=0.7)
    ax2.annotate(fr"$s^* = {s_star:+.3f}$",
                 xy=(s_star, 0.85 * y_clip),
                 xytext=(6, 0), textcoords="offset points",
                 color="tab:green", fontsize=10, va="top")
if R_sup is not None:
    Rg = R_sup[0]
    for xR in (-Rg, +Rg):
        ax2.axvline(xR, color="0.4", lw=1.0, ls=":")
    ax2.annotate(fr"$R_g = {Rg:.3f}$",
                 xy=(Rg, -0.85 * y_clip),
                 xytext=(6, 0), textcoords="offset points",
                 color="0.4", fontsize=10, va="bottom")

ax2.set_xlim(s_lo, s_hi)
ax2.set_ylim(-y_clip, y_clip)
ax2.set_xlabel(r"$s$", fontsize=13)
ax2.set_ylabel(r"$g(s)$", fontsize=13)
ax2.set_title(fr"{variant} latent-space RHS  "
              fr"($y$-axis clipped to $\pm{y_clip:.2g}$)",
              fontsize=11)
ax2.grid(True, ls=":", lw=0.5, alpha=0.5)
ax2.legend(loc="best", fontsize=10)
fig2.tight_layout()
outpath2 = f"results/check_g_of_s_{variant.replace('-','_')}"
fig2.savefig(outpath2 + ".png", dpi=140, bbox_inches="tight")
fig2.savefig(outpath2 + ".pdf", bbox_inches="tight")
print(f"Saved -> {outpath2}.png/.pdf")

plt.show()


PETSc.COMM_WORLD.Barrier()
os._exit(0)
