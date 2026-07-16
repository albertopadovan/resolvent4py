r"""
Spectral submanifold reduction with **time-periodic latent-space
dynamics coefficients** :math:`g_j(t)` and **pointwise biorthogonal
projection** — the "correct-but-expensive" extension of
:class:`SpectralSubmanifold`.

Theory
------

Parametrise

.. math::

    x(t) - c_\star(t) \;=\; P(t, s)
        \;=\; \sum_{j} P_j(t)\, s^j,
    \qquad
    \dot s \;=\; g(t, s) \;=\; \sum_{j} g_j(t)\, s^j,

with *both* :math:`P_j(t)` and :math:`g_j(t)` :math:`T_{HB}`-periodic
(the standard :class:`SpectralSubmanifold` forces :math:`g_j` to be
constant scalars).  Matching :math:`s^j`-orders in the SSM
invariance equation gives, for each multi-index :math:`j`,

.. math::

    (\partial_t + \Lambda_j - A(t))\, P_j(t)
        \;=\; R_j(t) - v(t)\, g_j(t),
    \tag{$*$}

where :math:`R_j` collects the bilinear + lower-order-:math:`g`
residual.  In the 2T-HB representation with harmonics
:math:`k \in [-n_f, +n_f]`, the natural pointwise normal-form
condition is :math:`\langle w(t), P_j(t)\rangle_{\text{pt}} = 0`
for :math:`j \ne (1, 0, \dots)`.  Because
:math:`\langle W_k, x_{HB}\rangle_{HB}` recovers the :math:`k`-th
Fourier coefficient of
:math:`\langle w(t), x(t)\rangle_{\text{pt}}` (via HB-shift +
Parseval), the resulting formula for the periodic ``g_j`` is

.. math::

    g_{j,\,k} \;=\; \langle W_k,\, R_j\rangle_{HB},
    \qquad k = -n_f,\dots,+n_f.
    \tag{$**$}

where :math:`W_k` is the HB-shift of :math:`w_{HB}` by :math:`k`
harmonics.  This requires the stacked-strip identity
:math:`\langle W_k, V_\ell\rangle_{HB} = \delta_{k\ell}`, which
holds iff :math:`\langle w(t), v(t)\rangle_{\text{pt}} = 1`
pointwise.  We verify this at the top of :meth:`solve` and
rescale :math:`v(t)` when it fails.

After :math:`g_j` is computed the RHS side-of-:math:`(*)` becomes
:math:`R_j - v \cdot g_j`, and in HB the product
:math:`v(t)\, g_j(t)` is a Fourier convolution.

Backwards-compatibility
-----------------------

The original :class:`SpectralSubmanifold` (with scalar
:math:`g_j`) is untouched.  Existing scripts / caches keep working.
This class writes ``gs`` as a list of length-:math:`n_{harm}`
arrays instead of length-:math:`r` arrays.  Downstream code that
loads a ``periodic_g`` cache must use
:class:`SpectralSubmanifoldROMPeriodicG` (companion file).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from .spectral_submanifold import SpectralSubmanifold
from ..utils.bv import reshape_bv_into_harmonic_balanced_vector
from ..utils.vector import reshape_harmonic_balanced_vector_into_bv


# ─────────────────────────────────────────────────────────────────────────
# HB-vector helpers
# ─────────────────────────────────────────────────────────────────────────
def _hb_time_series(bv_hb: SLEPc.BV, col: int) -> np.ndarray:
    """Return the ``col``-th column of ``bv_hb`` as a numpy (n_harm, n)
    complex array (harmonic-major)."""
    from ..utils.comms import gather_bv_to_rank

    arr = gather_bv_to_rank(bv_hb, 0)     # (n_state_hb, ncols)  on rank 0
    n_state_hb = arr.shape[0]
    return arr[:, col]                    # (n_state_hb,)


def _shift_hb_vector(v_hb: PETSc.Vec, k_shift: int, n_harm: int,
                     n_state: int) -> PETSc.Vec:
    r"""
    Return the HB vector representing
    :math:`v(t)\,\exp(i\, k_{\text{shift}}\,\omega_{HB}\, t)` —
    i.e. shift the HB harmonic ordering by ``k_shift`` blocks.

    Harmonic ``i`` of the shifted vector is harmonic ``i - k_shift``
    of the original.  Harmonics that fall outside ``[0, n_harm)``
    are set to zero (truncation).  Callers should ensure ``n_harm``
    is large enough that the truncated tail is negligible.
    """
    # Assumes v_hb is COMM_WORLD, layout n_harm × n_state stacked as one
    # long vector.  We work on the local slice with the harmonic block
    # boundaries.
    v_arr = v_hb.getArray().reshape(n_harm, -1)      # (n_harm, n_state)
    out = np.zeros_like(v_arr)
    if k_shift >= 0:
        out[k_shift:] = v_arr[: n_harm - k_shift]
    else:
        out[: n_harm + k_shift] = v_arr[-k_shift:]

    out_vec = v_hb.duplicate()
    out_vec.getArray()[:] = out.reshape(-1)
    return out_vec


def _shift_hb_column(hb_col_arr: np.ndarray, k_shift: int,
                     n_harm: int, n_state: int) -> np.ndarray:
    """Numpy-side HB shift: same semantics as :func:`_shift_hb_vector`
    but operating on an already-gathered ``(n_harm*n_state,)`` array."""
    v_arr = hb_col_arr.reshape(n_harm, n_state)
    out = np.zeros_like(v_arr)
    if k_shift >= 0:
        out[k_shift:] = v_arr[: n_harm - k_shift]
    else:
        out[: n_harm + k_shift] = v_arr[-k_shift:]
    return out.reshape(-1)


def _hb_convolve_axpy(rhs: PETSc.Vec, v_hb: PETSc.Vec,
                      g_hb: np.ndarray, alpha: complex,
                      n_harm_v: int, n_state: int) -> None:
    r"""
    In-place: ``rhs <- rhs + alpha * (v * g)_HB``

    where the product is HB Fourier-convolution:

    .. math::

        (v \star g)(n) \;=\; \sum_{k} v_{HB}(n - k)\, g_{HB}(k),
        \qquad n \in [0, n_{harm}_v),

    with out-of-range indices clipped to zero.  ``v_hb`` has
    ``n_harm_v`` harmonics; ``g_hb`` may have fewer
    (``n_harm_g = 2·nf_g + 1 ≤ n_harm_v``) — its indices are
    ``k = -nf_g, …, +nf_g``.
    """
    n_harm_g = len(g_hb)
    nf_g = (n_harm_g - 1) // 2
    v_arr = v_hb.getArray().reshape(n_harm_v, n_state)
    out = np.zeros_like(v_arr)
    # Convention (paired with shift_UP strip construction in solve()
    # and +1j reconstruction in the ROM): gs[nf_g + k] holds the +k-th
    # Fourier coefficient of g_j(t) (physics convention
    # g_j(t) = Σ_k gs[nf_g + k] · exp(+ikωt)).  Then
    # (v · g_j)_hb[n] = Σ_k v_hb[n − k] · gs[nf_g + k], because the
    # product v(t)·g_j(t) has Fourier harmonic n = l + k where l is
    # v's harmonic.  In array indices with i_g = nf_g + k:
    # v_arr[n_arr − (i_g − nf_g)] = v_arr[n_arr − i_g + nf_g].
    for i_g in range(n_harm_g):
        gk = g_hb[i_g]
        if gk == 0:
            continue
        shift = i_g - nf_g                     # true k-shift for this g coeff
        if shift >= 0:
            out[shift:] += alpha * gk * v_arr[: n_harm_v - shift]
        else:
            out[: n_harm_v + shift] += alpha * gk * v_arr[-shift:]
    rhs_arr = rhs.getArray().reshape(n_harm_v, n_state)
    rhs_arr += out


# ─────────────────────────────────────────────────────────────────────────
# The extended SSM class
# ─────────────────────────────────────────────────────────────────────────
class SpectralSubmanifoldPeriodicG(SpectralSubmanifold):
    r"""
    SSM with **time-periodic** latent-space dynamics coefficients
    :math:`g_j(t)` and **pointwise-biorthogonal** normal-form
    projection.

    Same constructor signature as :class:`SpectralSubmanifold`; only
    :meth:`solve` differs.  Downstream, ``self.gs`` is stored as a
    list of length-:math:`n_{harm}` complex arrays (Fourier
    coefficients of :math:`g_j(t)`) instead of length-:math:`r`
    arrays.
    """

    def solve(
        self,
        Phi: SLEPc.BV,
        Psi: SLEPc.BV,
        Lams: np.ndarray,
        scaling: float = 1.0,
        p0: Optional[PETSc.Vec] = None,
        verbose: int = 0,
        tol_biorth_pt: float = 1e-6,
        nf_g: Optional[int] = None,
    ) -> Tuple[List[PETSc.Vec], List[np.ndarray]]:
        r"""
        Solve for :math:`P_j(t)` and periodic :math:`g_j(t)`.

        The signature mirrors :meth:`SpectralSubmanifold.solve`; the
        additional ``tol_biorth_pt`` controls the pointwise
        biorthogonality check.  On return ``self.gs[j]`` is a
        length-:math:`(2\,n_{f,g}+1)` complex vector of Fourier
        coefficients of :math:`g_j(t)`, instead of a length-:math:`r`
        scalar coefficient tuple.

        :param nf_g: number of retained *positive* HB harmonics for the
            latent-space dynamics :math:`g_j(t)`.  Its Fourier basis is
            :math:`k = -n_{f,g},\dots,+n_{f,g}` (so
            :math:`2n_{f,g}+1` complex coefficients per multi-index).
            Must satisfy :math:`n_{f,g} \le n_f`.  Using
            :math:`n_{f,g} < n_f` restricts the strip-shifting of
            :math:`w_{HB}` to stay in the well-resolved harmonic
            range, avoiding truncation artefacts near
            :math:`|k| = n_f` for the master left mode.  Default is
            ``nf // 2``.
        """
        # For now only supports r = 1 chart-1 with a real periodic
        # SSM — the interesting case for KSE / Hopf-like problems.
        if len(Lams) != 1:
            raise NotImplementedError(
                "SpectralSubmanifoldPeriodicG currently supports r=1 only."
            )
        # Must be time-periodic (relies on HB representation)
        if not hasattr(self.diff_eq, "_omegas"):
            raise NotImplementedError(
                "SpectralSubmanifoldPeriodicG only makes sense for a "
                "time-periodic diff_eq."
            )

        r = 1
        n_harm = self.diff_eq._nblocks
        # ``get_state_dimension()`` returns the physical (per-time-slab)
        # state dim, not the HB total.  Read the HB size directly from
        # a Phi column instead so we always match the actual layout.
        _v_probe = Phi.getColumn(0)
        n_state_hb = int(_v_probe.getSize())
        Phi.restoreColumn(0, _v_probe)
        n_state = int(n_state_hb // n_harm)
        assert n_state * n_harm == n_state_hb, (
            f"HB size {n_state_hb} is not a multiple of n_harm={n_harm}"
        )
        nf = (n_harm - 1) // 2
        # nf_g: how many harmonics of g_j(t) do we retain?  Restrict
        # strip-shifting of w_hb to stay inside the well-resolved
        # harmonic band [-nf, +nf].  Default nf // 2 leaves half the
        # band as a "buffer" so the shifted W_k doesn't lose content.
        if nf_g is None:
            nf_g = nf // 2
        elif nf_g > nf:
            raise ValueError(
                f"nf_g = {nf_g} must be ≤ nf = {nf}"
            )
        # OUTPUT bandwidth (what the ROM sees / what we save).
        nf_g_out = nf_g
        n_harm_g_out = 2 * nf_g_out + 1
        # INTERNAL bandwidth: always solve with nf_g = nf so every
        # state strip |k| ≤ nf is projected out and cleaned to machine
        # precision.  Otherwise physical g_j Fourier content in strips
        # nf_g_out < |k| ≤ nf leaks into P_j via the linear solve and
        # contaminates all higher orders — measured empirically to be
        # ≳ |g_j|_max at every order, and it's what makes the ROM
        # unstable at moderate nf_g_out.  Truncate to nf_g_out at the
        # end (see the very bottom of solve()).
        nf_g = nf
        n_harm_g = 2 * nf_g + 1
        if verbose >= 1:
            from ..utils.miscellaneous import petscprint
            petscprint(
                self.diff_eq.get_comm(),
                f"[periodic_g] nf = {nf},  nf_g_out = {nf_g_out}  "
                f"(ROM sees 2·nf_g_out+1 = {n_harm_g_out} harmonics);  "
                f"internal projection uses nf_g_solve = {nf_g} (full-band, "
                f"no strip truncation).",
            )
        Lam = complex(Lams[0])

        # ── Scale eigenvectors (HB biorthonormalization) ──────────────
        scaling_arr = np.broadcast_to(
            np.asarray(scaling, dtype=PETSc.ScalarType), (r,)
        ).copy()
        V = Phi.copy(); W = Psi.copy()
        Vm = V.getMat(); Vm.scale(scaling_arr[0]); V.restoreMat(Vm)
        Wm = W.getMat(); Wm.scale(1.0 / scaling_arr[0]); W.restoreMat(Wm)

        # ── Step 1: check + fix pointwise biorthogonality ─────────────
        # Compute f(t) = <w(t), v(t)>_pt.  If ||f - 1||_inf > tol,
        # rescale v(t) pointwise so f(t) = 1 identically.
        v_col = V.getColumn(0)
        w_col = W.getColumn(0)
        v_hb_arr = v_col.getArray().copy().reshape(n_harm, n_state)
        w_hb_arr = w_col.getArray().copy().reshape(n_harm, n_state)
        V.restoreColumn(0, v_col)
        W.restoreColumn(0, w_col)

        # Diagnostic: check that v_hb and w_hb are conjugate-symmetric
        # in the HB harmonic axis — i.e. represent real functions v(t),
        # w(t) in physical time.  For a real KSE with a real master
        # Floquet exponent Λ this MUST hold, else g_j(t) reconstructed
        # from the projection will be complex-valued and s(t) drifts
        # into the imaginary axis.
        if verbose >= 1:
            asym_v = 0.0
            asym_w = 0.0
            for i_ in range(nf + 1):
                j_ = 2 * nf - i_
                if 0 <= j_ < n_harm:
                    asym_v = max(asym_v, float(np.max(np.abs(
                        v_hb_arr[i_] - np.conj(v_hb_arr[j_])
                    ))))
                    asym_w = max(asym_w, float(np.max(np.abs(
                        w_hb_arr[i_] - np.conj(w_hb_arr[j_])
                    ))))
            from ..utils.miscellaneous import petscprint
            petscprint(
                self.diff_eq.get_comm(),
                f"[periodic_g] v_hb conj-symm asymm = {asym_v:.3e}  "
                f"(rel = {asym_v / max(np.abs(v_hb_arr).max(), 1e-30):.3e})",
            )
            petscprint(
                self.diff_eq.get_comm(),
                f"[periodic_g] w_hb conj-symm asymm = {asym_w:.3e}  "
                f"(rel = {asym_w / max(np.abs(w_hb_arr).max(), 1e-30):.3e})",
            )

        # Reconstruct in time via IFFT (positive/negative sym is embedded
        # in v_hb via the eigendecomp; use inverse DFT).
        time = self.diff_eq._time                       # (n_t,)
        omegas = self.diff_eq._omegas                   # (n_harm,)
        weights = np.exp(1j * np.outer(omegas, time))   # (n_harm, n_t)
        v_time = v_hb_arr.T @ weights                   # (n_state, n_t)
        w_time = w_hb_arr.T @ weights
        f_time = np.sum(w_time.conj() * v_time, axis=0)  # (n_t,)  <w(t),v(t)>

        max_dev = float(np.max(np.abs(f_time - 1.0)))
        if verbose >= 1:
            from ..utils.miscellaneous import petscprint
            petscprint(
                self.diff_eq.get_comm(),
                f"[periodic_g] pointwise <w,v>: mean = {f_time.mean():+.4e}, "
                f"max deviation from 1 = {max_dev:.3e}",
            )
        if max_dev > tol_biorth_pt:
            # Rescale v(t) → v(t) / f(t) pointwise, then FFT back.
            v_time_rescaled = v_time / f_time[None, :]
            # Inverse: v_hb[k] = (1/n_t) sum_t v_time[:, t] exp(-i k ω t)
            weights_inv = np.exp(-1j * np.outer(omegas, time)) / len(time)
            v_hb_new = (v_time_rescaled @ weights_inv.T).T  # (n_harm, n_state)
            # Write back into V.
            v_col = V.getColumn(0)
            v_col.getArray()[:] = v_hb_new.reshape(-1)
            V.restoreColumn(0, v_col)
            v_hb_arr = v_hb_new
            if verbose >= 1:
                # Verify
                v_time_check = v_hb_new.T @ weights
                f_check = np.sum(w_time.conj() * v_time_check, axis=0)
                from ..utils.miscellaneous import petscprint
                petscprint(
                    self.diff_eq.get_comm(),
                    f"[periodic_g] after rescale: max |<w,v>-1| = "
                    f"{float(np.max(np.abs(f_check - 1.0))):.3e}",
                )

        # Reusable template PETSc.Vec (needed by both the strip check
        # below and the P_j linear solve further down).
        _v_ref = V.getColumn(0)
        template = _v_ref.duplicate()
        V.restoreColumn(0, _v_ref)

        # ── Precompute strip-shifted W_k for k = -nf_g..+nf_g ─────────
        # W_strips has shape (n_harm_g, n_harm, n_state).
        # CONVENTION: W_strips[nf_g + k] = shift-DOWN-by-k of w_hb.
        # This makes each pair (W_strips[nf_g + k], V_shift_down_by_k)
        # an eigenpair of L_HB at eigenvalue Λ + iωk — matches the
        # user's convention  w_k^H · L · v_k = Λ + iωk.
        # (Shift-UP would give Λ − iωk instead.)
        w_col = W.getColumn(0)
        w_hb_arr = w_col.getArray().copy().reshape(n_harm, n_state)
        W.restoreColumn(0, w_col)
        W_strips = np.zeros((n_harm_g, n_harm, n_state), dtype=complex)
        for i_k in range(n_harm_g):
            k_shift = i_k - nf_g
            if k_shift >= 0:
                # shift DOWN by k: W_new[i] = w[i + k], valid for i + k < n_harm
                W_strips[i_k, : n_harm - k_shift] = w_hb_arr[k_shift:]
            else:
                # shift DOWN by k (k<0 → shift UP by |k|):
                # W_new[i] = w[i + k], valid for i + k >= 0 ⟹ i >= -k
                W_strips[i_k, -k_shift:] = w_hb_arr[: n_harm + k_shift]
        # Flatten each strip: (n_harm_g, n_state_HB).  With the internal
        # full-band setting nf_g = nf, this is (2*nf+1, n_harm*n_state)
        # and covers EVERY state strip — no separate "leakage" projector
        # is needed.
        W_strips_flat = W_strips.reshape(n_harm_g, n_harm * n_state)

        # ── Also build shifted V strips and CHECK  <W_k, L_HB V_k>  ──────
        # With shift-DOWN on both W and V, each pair is a (left, right)
        # eigenvector pair of L_HB at eigenvalue Λ + iω·k.
        if verbose >= 1:
            v_col = V.getColumn(0)
            v_hb_arr = v_col.getArray().copy().reshape(n_harm, n_state)
            V.restoreColumn(0, v_col)
            omega_HB = float(self.diff_eq._omegas[nf + 1])   # fundamental
            from ..utils.miscellaneous import petscprint
            petscprint(
                self.diff_eq.get_comm(),
                f"[periodic_g] strip Floquet check:  Λ = {Lam.real:+.4e}"
                f"{Lam.imag:+.4e}j,  ω_HB = {omega_HB:.4e}",
            )
            L_Vk = template.duplicate()
            # Report only the output-band strips |k| ≤ nf_g_out (n_harm_g
            # is now the full state bandwidth 2*nf+1; running the check
            # over all 161 strips is expensive and uninteresting).
            for i_k in range(nf_g - nf_g_out, nf_g + nf_g_out + 1):
                k_shift = i_k - nf_g
                # Build V_k = shift_DOWN_by_k(v_hb)
                v_shifted = np.zeros((n_harm, n_state), dtype=complex)
                if k_shift >= 0:
                    v_shifted[: n_harm - k_shift] = v_hb_arr[k_shift:]
                else:
                    v_shifted[-k_shift:] = v_hb_arr[: n_harm + k_shift]
                v_shifted_flat = v_shifted.reshape(-1)
                Vk_vec = template.duplicate()
                Vk_vec.getArray()[:] = v_shifted_flat
                # Apply L_HB
                L_Vk.zeroEntries()
                L_Vk = self.diff_eq.evaluate_linear_term(0.0, Vk_vec, L_Vk)
                # Project onto W_k
                lambda_k = complex(W_strips_flat[i_k].conj() @ L_Vk.getArray())
                # Expected eigenvalue: Λ + iωk (shift-DOWN convention).
                lambda_expect = Lam + 1j * omega_HB * k_shift
                err = abs(lambda_k - lambda_expect)
                petscprint(
                    self.diff_eq.get_comm(),
                    f"[periodic_g]   k={k_shift:+d}: "
                    f"<W_k, L·V_k> = {lambda_k.real:+.4e}{lambda_k.imag:+.4e}j"
                    f"  |err vs Λ+ikω| = {err:.2e}",
                )
                Vk_vec.destroy()
            L_Vk.destroy()

        # ── Init ps, gs ────────────────────────────────────────────────
        # (template already created above, before the strip check)
        ps_zero = template.duplicate()
        if p0 is None:
            ps_zero.zeroEntries()
        else:
            p0.copy(ps_zero)
        ps: List[PETSc.Vec] = [ps_zero]
        v_col = V.getColumn(0)
        ps.append(v_col.copy())
        V.restoreColumn(0, v_col)

        # gs[j] is (n_harm_g,) complex.  gs[0] = 0 (orbit reference).
        # gs[1] = Lam · e_{DC}  (linear master term, HB-DC only,
        #                        matches constant Λ · s dynamics).
        gs: List[np.ndarray] = [np.zeros(n_harm_g, dtype=complex)]
        g_lin = np.zeros(n_harm_g, dtype=complex)
        g_lin[nf_g] = Lam                                     # DC index
        gs.append(g_lin)

        # ── Main loop over multi-indices ────────────────────────────────
        rhs = template.duplicate()
        rhsj = template.duplicate()

        for j_idx in range(r + 1, len(self.ssm_multiindices)):
            j = self.ssm_multiindices[j_idx]
            if verbose == 1:
                from ..utils.miscellaneous import petscprint
                petscprint(
                    self.diff_eq.get_comm(),
                    f"[periodic_g] j = {j} (order = {sum(j)})",
                )
            shift = PETSc.ScalarType(np.dot([Lam], np.asarray(j)))
            rhs.zeroEntries()

            # ── Bilinear residual (same as parent) ─────────────────────
            if self.n_workers == 1:
                for pair in self.ssm_quad_rhs_idc[j_idx]:
                    idces = [
                        self._get_multiindex_index(self.ssm_multiindices, p)
                        for p in pair
                    ]
                    rhsj = self._evaluate_quadratic_rhs_root(
                        0, ps[idces[0]], ps[idces[1]], rhsj
                    )
                    rhs.axpy(1.0, rhsj)
            else:
                self._evaluate_quadratic_rhs_parallel(j_idx, ps, rhs)

            # ── Lower-order g contribution to RHS — via HB convolution ──
            # For each lower-order pair (P_l, g_{j-l+1}), the term
            #   ell * P_l(t) * g_{j-l+1}(t)
            # is subtracted from the RHS.  In HB this is a convolution.
            for k_slot in range(r):
                for pair in self.ssm_nonlin_dynmc[j_idx][k_slot]:
                    idxp = self._get_multiindex_index(
                        self.ssm_multiindices, pair[0]
                    )
                    idxg = self._get_multiindex_index(
                        self.ssm_multiindices, pair[1]
                    )
                    alpha = complex(-pair[0][k_slot])
                    _hb_convolve_axpy(
                        rhs, ps[idxp], gs[idxg], alpha, n_harm, n_state,
                    )

            # ── Project onto W_k for each k → g_{j,k} ──────────────────
            # Strip build uses shift-DOWN, so w_k^H · L · v_k = Λ + iωk
            # (user's convention).  The raw projection
            #     proj[nf_g + k] = <shift-DOWN(w), rhs>_HB
            # equals the (−k)-th Fourier coefficient of the pointwise
            # pairing f(t) = <w(t), rhs(t)>_pt.  To match the user's
            # physics-convention storage where gs[nf_g + m] holds the
            # (+m)-th Fourier coefficient of g_j(t), we reverse the
            # projected array in place: reversal sends index nf_g + k
            # ↔ nf_g − k, so proj[nf_g − m] (which is f_{+m}) lands at
            # gs[nf_g + m].  After this reversal, gs is stored in
            # (obj_{−nf_g}, …, obj_0, …, obj_{+nf_g}) physics order and
            # both the convolution formula (v_arr[n_arr − i_g + nf_g])
            # and the ROM's +1j reconstruction agree with it.
            rhs_flat = rhs.getArray()                             # (n_state_HB,)
            gj_hb = W_strips_flat.conj() @ rhs_flat               # (n_harm_g,)
            gj_hb = gj_hb[::-1].copy()                            # reverse

            # Diagnostic + fix: for a REAL system g_j(t) must be real,
            # so its Fourier coefficients must satisfy g_{j,-m} = conj(g_{j,+m}).
            # Numerical drift can break this; symmetrise and report the
            # asymmetry.  If asymm >> machine noise, some upstream vector
            # (v_hb, w_hb, or lower-order ps) has lost conj symmetry —
            # investigate rather than mask.
            asym = 0.0
            for m_ in range(1, nf_g + 1):
                a = gj_hb[nf_g + m_]
                b = gj_hb[nf_g - m_]
                asym = max(asym, abs(a - np.conj(b)))
            gj_hb_sym = gj_hb.copy()
            for m_ in range(1, nf_g + 1):
                avg = 0.5 * (gj_hb[nf_g + m_] + np.conj(gj_hb[nf_g - m_]))
                gj_hb_sym[nf_g + m_] = avg
                gj_hb_sym[nf_g - m_] = np.conj(avg)
            gj_hb_sym[nf_g] = gj_hb[nf_g].real + 0.0j            # DC real
            if verbose >= 1:
                from ..utils.miscellaneous import petscprint
                petscprint(
                    self.diff_eq.get_comm(),
                    f"[periodic_g]   j={j}  conj-sym asymm max = {asym:.3e}"
                    f"  (relative = {asym / max(abs(gj_hb).max(), 1e-30):.3e})",
                )
                # Per-harmonic magnitude — look for a support pattern.
                # If content sits only at specific residue classes (e.g.
                # m ≡ 0 mod 4, or m ≡ 2 mod 4), the SSM inherits a
                # symmetry constraint we could enforce explicitly.
                # Show only the output-band harmonics |m| ≤ nf_g_out
                # (the internal band is 2*nf+1 = 161 harmonics which
                # would flood the log).
                gmax = float(abs(gj_hb).max()) + 1e-30
                m_lo = nf_g - nf_g_out
                m_hi = nf_g + nf_g_out + 1
                mags = np.array([
                    f"m={m_-nf_g:+d}:{abs(gj_hb[m_])/gmax:.1e}"
                    for m_ in range(m_lo, m_hi)
                ])
                # Print in a couple of lines
                per_line = 8
                for start in range(0, len(mags), per_line):
                    petscprint(
                        self.diff_eq.get_comm(),
                        f"[periodic_g]     "
                        + "  ".join(mags[start:start + per_line]),
                    )
            gj_hb = gj_hb_sym

            if (
                self.latent_space_components is not None
                and sum(j) not in self.latent_space_components
            ):
                gj_hb = np.zeros(n_harm_g, dtype=complex)

            # Enforce the spatiotemporal-Z₂ symmetry constraint the KSE
            # inherits from a half-period-antisymmetric base flow:
            #     g_{j,m} = 0  whenever  (j + m)  is EVEN.
            # The projection gives this near machine noise anyway (~1e-5
            # relative to the physical content), but that "noise" gets
            # amplified by exp(Re(Λ)·T) over long integrations and drives
            # the ROM unstable at fast Λ.  Zeroing it out kills the
            # amplifier at source.  Set `enforce_parity_symmetry = False`
            # on the class to disable.
            if getattr(self, "enforce_parity_symmetry", True):
                j_total = sum(j)
                for m_ in range(n_harm_g):
                    m_val = m_ - nf_g
                    if (j_total + m_val) % 2 == 0:
                        gj_hb[m_] = 0.0

            # ── Subtract v · g_j convolution from RHS ─────────────────
            v_col = V.getColumn(0)
            _hb_convolve_axpy(
                rhs, v_col, gj_hb, -1.0 + 0.0j, n_harm, n_state,
            )
            V.restoreColumn(0, v_col)

            # ── Consistency check: after subtracting v·g_j from rhs, the
            # residual should be pointwise-orthogonal to w at every
            # state strip |k| ≤ nf (internal projection cleans them
            # all).  Report:
            #   • L2 of the full post-subtract projection — should be at
            #     machine precision at every order.  Anything larger
            #     signals a bug (bad convolution, broken conj sym, etc.)
            #   • L2 of g_j Fourier content INSIDE the output band
            #     |k| ≤ nf_g_out (kept by the ROM) vs OUTSIDE (discarded
            #     when we truncate on save).  If the OUTSIDE content is
            #     large relative to inside, the ROM's truncated g_j is
            #     missing a physically meaningful tail.
            if verbose >= 1:
                rhs_flat_after = rhs.getArray()
                proj_all = W_strips_flat.conj() @ rhs_flat_after   # (n_harm_g,)
                resid_l2 = float(
                    np.sqrt(np.sum(np.abs(proj_all)**2))
                )
                resid_max = float(np.max(np.abs(proj_all)))
                # g_j band split (physics-order: gs[nf_g + m] = f_+m).
                m_lo = nf_g - nf_g_out
                m_hi = nf_g + nf_g_out + 1
                g_in = gj_hb[m_lo:m_hi]
                g_out = np.concatenate([gj_hb[:m_lo], gj_hb[m_hi:]])
                g_in_l2 = float(np.sqrt(np.sum(np.abs(g_in)**2)))
                g_out_l2 = float(np.sqrt(np.sum(np.abs(g_out)**2)))
                g_out_max = float(np.max(np.abs(g_out))) if g_out.size else 0.0
                if (
                    self.latent_space_components is None
                    or sum(j) in self.latent_space_components
                ):
                    scale = float(np.max(np.abs(gj_hb))) if np.any(gj_hb) else 1.0
                    from ..utils.miscellaneous import petscprint
                    petscprint(
                        self.diff_eq.get_comm(),
                        f"[periodic_g]   j={j}  post-subtract residual "
                        f"L2 = {resid_l2:.3e}  max = {resid_max:.3e}",
                    )
                    petscprint(
                        self.diff_eq.get_comm(),
                        f"[periodic_g]     g_j L2 inside |k|≤nf_g_out: "
                        f"{g_in_l2:.3e}  outside: {g_out_l2:.3e}  "
                        f"outside_max: {g_out_max:.3e}  "
                        f"(|g_j|_max = {scale:.3e})",
                    )

            # ── Solve (shift·I − L) P_j = RHS ──────────────────────────
            pj = self.diff_eq.solve_linear_system(shift, rhs)

            gs.append(gj_hb)
            ps.append(pj)

        template.destroy()
        # Truncate gs from internal full-band (length n_harm = 2*nf+1)
        # to the user-requested output band (length 2*nf_g_out+1).  The
        # internal band was used only to clean every state strip during
        # the SSM iteration; downstream code (save script, ROM) sees the
        # output-band gs.  Keep the full-band gs on self._gs_full for
        # diagnostics.
        self._gs_full = gs                                # length n_harm each
        m_lo = nf_g - nf_g_out                            # nf_g == nf here
        m_hi = nf_g + nf_g_out + 1
        gs_out = [g[m_lo:m_hi].copy() for g in gs]        # length n_harm_g_out
        self.ps = ps
        self.gs = gs_out
        self.Lams = Lams
        self.V = V
        self.W = W
        return ps, gs_out
