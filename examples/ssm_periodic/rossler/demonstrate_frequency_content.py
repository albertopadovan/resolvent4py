r"""
Frequency-content diagnostic for the Rössler 2T-periodic SSM
coefficients.

In the 2T-doubled construction the HB basis lives at multiples of
:math:`\omega \;=\; \omega_{\rm base} / 2`, so HB block index :math:`j`
(``-nf ≤ j ≤ nf``) corresponds to the physical frequency
:math:`j \cdot \omega_{\rm base}/2`:

- ``j`` **odd**  → half-integer multiples of :math:`\omega_{\rm base}`
  (the period-doubled subharmonics): ``{…, -3/2, -1/2, +1/2, +3/2, …}``.
- ``j`` **even** → integer multiples of :math:`\omega_{\rm base}`
  (the base-orbit harmonics): ``{…, -1, 0, +1, …}``.

Question: for each polynomial term :math:`p_{(k,)}` in the SSM
expansion (`(k,)` is the multi-index for the single master mode), where
does its frequency support live?

Expected pattern.  The base-orbit Jacobian :math:`A(t)` has support
only on even HB blocks (it's :math:`T_{\rm base}`-periodic; in the 2T
basis its Fourier coefficients vanish on odd blocks).  The
homological equation that defines :math:`p_{(k,)}` therefore
preserves parity through convolution:

- :math:`k = 1` (eigenfunction):  odd-block support only.
- :math:`k = 2` (quadratic):  even-block support only
  (``odd ⊛ odd = even``).
- :math:`k = 3` (cubic):  odd-block support only
  (``odd ⊛ even = odd``).
- general: ``parity(p_{(k,)}) == parity(k)``.

The script verifies this by computing the L2 norm of each HB block,
splitting energy into odd / even blocks, and plotting the
distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ── Load SSM cache ─────────────────────────────────────────────────────────
data = np.load("data/ssm_cache_2T.npz")
PS_hb = data["PS_hb"]                      # (n_terms, n_harmonics, n_state)
multiindices = data["multiindices"]        # (n_terms, r)

n_terms, n_harmonics, n_state = PS_hb.shape
nf = (n_harmonics - 1) // 2
T_base = float(data["T_base"])
omega_base = 2.0 * np.pi / T_base

# 1-D order index for r=1.  For higher r, fall back to total order.
orders = multiindices.sum(axis=1).astype(int)
m = int(orders.max())

# Frequency axis in units of omega_base.  HB block j → j·omega/omega_base
#   = j·(omega_base/2)/omega_base = j/2.
freq_axis_units_of_omega_base = np.arange(-nf, nf + 1) / 2.0

print(
    f"Loaded SSM cache: n_terms={n_terms}, "
    f"n_harmonics={n_harmonics} (nf={nf}), "
    f"n_state={n_state}, T_base={T_base:.4f}, "
    f"omega_base={omega_base:.4f}, max order m={m}"
)


# ── Per-block energy and parity decomposition ──────────────────────────────
# block_energy[term, j] = ‖p_{term}[block j]‖₂
block_energy = np.linalg.norm(PS_hb, axis=2)

hb_index = np.arange(-nf, nf + 1)
odd_mask_block = (hb_index % 2 != 0)
even_mask_block = ~odd_mask_block

# Energy of each term split by block parity.
odd_energy_sq = (block_energy[:, odd_mask_block] ** 2).sum(axis=1)
even_energy_sq = (block_energy[:, even_mask_block] ** 2).sum(axis=1)
total_sq = odd_energy_sq + even_energy_sq
with np.errstate(divide="ignore", invalid="ignore"):
    odd_fraction = np.where(total_sq > 0.0, odd_energy_sq / total_sq, 0.0)


# ── Parity table ───────────────────────────────────────────────────────────
print()
print("Order  expected  odd-block frac   even-block frac   ‖p_(k,)‖")
print("─────  ────────  ──────────────   ───────────────   ────────")
for k in range(1, m + 1):
    mask = orders == k
    if not mask.any():
        continue
    odd_frac = odd_fraction[mask].mean()
    even_frac = 1.0 - odd_frac
    norm = np.sqrt(total_sq[mask].mean())
    expected = "odd" if k % 2 == 1 else "even"
    print(
        f"  {k:3d}    {expected:>4s}    {odd_frac:14.6e}   "
        f"{even_frac:14.6e}   {norm:.3e}"
    )


# ── Plot frequency content by order parity ────────────────────────────────
n_orders_to_plot = min(6, m)
order_examples_odd = [k for k in range(1, m + 1, 2)][:n_orders_to_plot]
order_examples_even = [k for k in range(2, m + 1, 2)][:n_orders_to_plot]

fig, (ax_odd, ax_even) = plt.subplots(
    nrows=2, figsize=(9.5, 7.5), sharex=True
)
cmap_odd = plt.cm.viridis(np.linspace(0.1, 0.9, len(order_examples_odd)))
cmap_even = plt.cm.plasma(np.linspace(0.1, 0.9, len(order_examples_even)))

floor = 1e-14
for k, color in zip(order_examples_odd, cmap_odd):
    mask = orders == k
    if not mask.any():
        continue
    y = np.maximum(block_energy[mask].mean(axis=0), floor)
    ax_odd.semilogy(
        freq_axis_units_of_omega_base, y,
        "o-", color=color, markersize=3, linewidth=0.8,
        label=f"k = {k}",
    )

for k, color in zip(order_examples_even, cmap_even):
    mask = orders == k
    if not mask.any():
        continue
    y = np.maximum(block_energy[mask].mean(axis=0), floor)
    ax_even.semilogy(
        freq_axis_units_of_omega_base, y,
        "s-", color=color, markersize=3, linewidth=0.8,
        label=f"k = {k}",
    )

for ax, title in (
    (ax_odd, "ODD orders — expected support on half-integer ω_base"),
    (ax_even, "EVEN orders — expected support on integer ω_base"),
):
    ax.set_ylabel(r"$\|p_{(k,)}[\mathrm{block}\;j]\|_2$")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, which="both", ls=":", lw=0.4)
    ax.set_title(title, fontsize=10)
    for x in np.arange(int(freq_axis_units_of_omega_base.min()),
                       int(freq_axis_units_of_omega_base.max()) + 1):
        ax.axvline(x, color="0.85", lw=0.4, zorder=-1)

ax_even.set_xlabel(r"frequency $j / 2$ (in units of $\omega_{\rm base}$)")

fig.suptitle(
    "Rössler 2T SSM — per-block energy of $p_{(k,)}$ by order",
    fontsize=11,
)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
out_png = "results/ssm_frequency_content_2T.png"
out_pdf = "results/ssm_frequency_content_2T.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print()
print(f"Saved figure → {out_png}, {out_pdf}")
plt.show()
