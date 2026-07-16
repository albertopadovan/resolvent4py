r"""
Two diagnostics on the periodic-g SSM cache:

  1.  Latent-space DYNAMICS.  Extract the DC part g_j^{DC} of each g_j(t),
      form the DC polynomial g^{DC}(s) = Σ_j g_j^{DC} s^{deg_j}, and check:
        · sign of the leading nontrivial coefficient (stabilising vs
          destabilising),
        · real roots of g^{DC}(s) = 0 sorted by |s|,
        · sign of ds/dt = g^{DC}(s) evaluated at a range of s.
      This tells us whether the ROM's autonomous drift saturates at all.

  2.  PARITY structure.  For the KSE 2T-cover the Z2 symmetry
        c_*(t + T/2) = -c_*(t)   ⇒   the SSM inherits
        g_{j,m} = 0    whenever   (j + m)  is EVEN.
      In particular  g_{j,0}  vanishes for every EVEN polynomial order j.
      We check whether that holds in the saved cache and, if not, quantify
      how badly the constraint is violated.
"""
# Bootstrap: this script was moved into periodic_g_manifold/.  Make the
# parent kse/ directory importable (for spatial_operators, kse_differential_equation,
# eigendecomp_kse) and chdir to it so relative data/... paths still resolve.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np


d = np.load("data/ssm_cache_periodic_g.npz")
gs = d["gs_periodic"][:, :, 0]                           # (n_terms, n_harm_g)
multiindices = d["multiindices"]
degs = multiindices[:, 0].astype(int)
n_terms, n_harm_g = gs.shape
nf_g = (n_harm_g - 1) // 2
rho_domain = float(d["rho_domain"])
Lam0 = complex(d["Lams"][0])
print(f"periodic-g cache:  n_terms = {n_terms},  n_harm_g = {n_harm_g},  "
      f"nf_g = {nf_g}")
print(f"|Λ| = {abs(Lam0.real):.4e},  rho_domain = {rho_domain:.4f}")


# ── (1) Latent-space DC dynamics ─────────────────────────────────────
g_DC = gs[:, nf_g]                                       # (n_terms,) complex
print(f"\n{'idx':>3s}  {'deg':>3s}  {'Re g_j^DC':>18s}  {'Im g_j^DC':>18s}  "
      f"{'|g_j|_∞ over t':>18s}")
for j_idx in range(min(15, n_terms)):
    j = degs[j_idx]
    Re = g_DC[j_idx].real
    Im = g_DC[j_idx].imag
    # sup_t |g_j(t)|  reconstruction max
    n_check = 200
    t_check = np.linspace(0.0, 1.0, n_check, endpoint=False)     # unit period
    weights = np.exp(1j * 2 * np.pi
                     * np.outer(t_check, np.arange(-nf_g, nf_g + 1)))
    g_t = np.real(weights @ gs[j_idx])
    sup_g = float(np.max(np.abs(g_t)))
    print(f"{j_idx:>3d}  {j:>3d}  {Re:+18.6e}  {Im:+18.6e}  {sup_g:18.6e}")


# ── Build DC polynomial and analyse it ──────────────────────────────
max_deg = int(degs.max())
g_coeffs_DC = np.zeros(max_deg + 1, dtype=complex)
for k in range(len(degs)):
    g_coeffs_DC[degs[k]] += g_DC[k]                      # accumulate if r>1

print(f"\nDC polynomial g^DC(s) = Σ_j g_j^DC · s^j:")
for k in range(min(15, max_deg + 1)):
    if abs(g_coeffs_DC[k]) > 0:
        print(f"   s^{k:>2d}:  {g_coeffs_DC[k].real:+.4e}"
              f"{g_coeffs_DC[k].imag:+.4e}j")

# Real roots — where does g^DC(s) = 0?
roots = np.roots(g_coeffs_DC[::-1])
real_roots = sorted(
    [z.real for z in roots
     if abs(z.imag) < 1e-6 * max(abs(z.real), 1.0)],
    key=lambda x: abs(x),
)
print(f"\nReal roots of g^DC(s) = 0  (sorted by |s|):")
for k, r in enumerate(real_roots[:12]):
    tag = ""
    if abs(r) < 1e-8:
        tag = "  (trivial 0)"
    elif abs(r) < rho_domain:
        tag = f"  <-- inside rho_domain = {rho_domain:.3f}"
    else:
        tag = f"  (outside rho_domain, likely spurious)"
    print(f"   root {k}:  s = {r:+.4e}{tag}")

# Probe sign of ds/dt = g^DC(s)  at a range of s
print(f"\nSign of g^DC(s) at probing amplitudes:")
for s_probe in [0.1, 1.0, 5.0, 10.0, 30.0, 60.0, 100.0]:
    val = float(sum(g_coeffs_DC[k].real * s_probe ** k
                    for k in range(max_deg + 1)))
    print(f"   s = {s_probe:6.1f}:  g^DC(s) = {val:+.4e}   "
          f"{'GROWS' if val > 0 else 'DECAYS'}")


# ── (2) Parity structure ────────────────────────────────────────────
# Z2 rule:  g_{j, m} = 0  whenever  (j + m) is EVEN.
# Storage:  gs[j_idx, nf_g + m] holds the m-th Fourier coefficient of g_j(t).
print("\n" + "─" * 60)
print("PARITY diagnostic  (should vanish when j+m is even):")

# Sum of |g_{j,m}|  over the "forbidden" (j+m even) vs "allowed" (j+m odd)
# sets for each polynomial order.
print(f"\n{'idx':>3s}  {'deg':>3s}  "
      f"{'|forbidden|₁ (j+m even)':>25s}  "
      f"{'|allowed|₁ (j+m odd)':>22s}  {'ratio':>10s}")
for j_idx in range(min(20, n_terms)):
    j = degs[j_idx]
    forb = 0.0
    allw = 0.0
    for m in range(-nf_g, nf_g + 1):
        v = abs(gs[j_idx, nf_g + m])
        if (j + m) % 2 == 0:
            forb += v
        else:
            allw += v
    ratio = forb / max(allw, 1e-30)
    print(f"{j_idx:>3d}  {j:>3d}  {forb:25.4e}  {allw:22.4e}  {ratio:10.3e}")

# Total across all orders
total_forb = 0.0
total_allw = 0.0
for j_idx in range(n_terms):
    j = degs[j_idx]
    for m in range(-nf_g, nf_g + 1):
        v = abs(gs[j_idx, nf_g + m])
        if (j + m) % 2 == 0:
            total_forb += v
        else:
            total_allw += v
print(f"\nTotal parity violation:")
print(f"   Σ |g_{{j,m}}| over (j+m) EVEN  = {total_forb:.4e}")
print(f"   Σ |g_{{j,m}}| over (j+m) ODD   = {total_allw:.4e}")
print(f"   ratio forbidden/allowed     = "
      f"{total_forb / max(total_allw, 1e-30):.3e}")

# Specifically for EVEN j, what does the DC (m=0) look like?  Under the Z2
# rule, (j + 0) is EVEN for even j, so g_{j, 0} = 0 must hold.
print(f"\ng_j^DC for EVEN j  (must be 0 under Z2):")
for j_idx in range(n_terms):
    j = degs[j_idx]
    if j % 2 == 0 and j > 0:
        print(f"   deg {j:>2d}:  g_j^DC = {g_DC[j_idx].real:+.4e}"
              f"{g_DC[j_idx].imag:+.4e}j     |value| = "
              f"{abs(g_DC[j_idx]):.4e}")
    if j >= 20:
        break
