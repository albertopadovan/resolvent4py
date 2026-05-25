import os
import shutil

import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from typing import Optional, Tuple

from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py

from resolvent4py.spectral_submanifold import DifferentialEquation
from spatial_operators import linear_eigenvalues


class KuramotoSivashinskyPeriodic(DifferentialEquation):
    r"""
    Harmonic-balanced formulation of the Kuramoto-Sivashinsky equation
    linearised about a time-periodic orbit :math:`c^*(t)`.

    The perturbation dynamics are

    .. math::

        \dot{v} = A(t)\,v + B(v, v),
        \qquad A(t) = L + 2\,B(c^*(t),\,\cdot\,),

    where :math:`A(t) = A(t + T)`.  In the harmonic-balanced (Fourier)
    representation the state becomes a stacked vector of Fourier
    coefficients

    .. math::

        \hat{v} = (\hat{v}_{-n_f}, \ldots, \hat{v}_0, \ldots, \hat{v}_{n_f})

    and the linear operator is the harmonic resolvent generator

    .. math::

        \mathcal{L} = -\mathrm{diag}(i k\omega I) + \mathcal{A},

    where :math:`\mathcal{A}` is the block-Toeplitz matrix assembled from
    the Fourier coefficients of :math:`A(t)`.

    :param n: number of sine modes
    :type n: int
    :param nu: viscosity parameter
    :type nu: float
    :param nf: number of positive frequencies for the SSM computation
    :type nf: int
    :param nfb: number of positive base-flow frequencies to retain
    :type nfb: int
    :param c_star: periodic orbit, shape ``(n, n_time)``
    :type c_star: np.ndarray
    :param time: time vector of length ``n_time``, uniformly spaced in
        ``[0, T)`` with ``n_time >= 2 * nf + 1``
    :type time: np.ndarray
    :param n_pts: physical-space grid points (default ``4 * n``)
    :type n_pts: int, optional
    """

    def __init__(
        self,
        n: int,
        nu: float,
        nf: int,
        nfb: int,
        c_star: np.ndarray,
        time: np.ndarray,
        n_pts: int = None,
    ) -> None:
        # NOTE: callers must also pass ``periodic_diffeq`` as a keyword
        # argument; it is captured by
        # :meth:`DifferentialEquation.__new__` to trigger the
        # ``PeriodicDifferentialEquation`` mixin wrap and is consumed
        # by ``PeriodicDifferentialEquation.__init__`` before this
        # ``__init__`` is called via the MRO.
        comm = PETSc.COMM_WORLD

        # Validate inputs
        n_time = len(time)
        if n_time < 2 * nf + 1:
            raise ValueError(
                f"len(time) = {n_time} < 2*nf+1 = {2*nf+1}. "
                f"Time sampling does not satisfy Nyquist for nf={nf}."
            )
        if c_star.shape != (n, n_time):
            raise ValueError(
                f"c_star.shape = {c_star.shape}, expected ({n}, {n_time})."
            )
        if nfb > nf:
            raise ValueError(f"nfb={nfb} must be <= nf={nf}.")

        # PER-TIME (per-frequency-block) state dimension.  The PDE mixin
        # allocates BVs sized by self._state_dim, so this must be the
        # per-time block size; the HB vector that callers pass to
        # evaluate_quadratic_term has size n * (2*nf + 1).
        state_dim = (res4py.compute_local_size(n), n)

        T_period = time[-1] + (time[1] - time[0])
        omega = 2 * np.pi / T_period

        super().__init__(
            comm, "KuramotoSivashinskyPeriodic", state_dim, 2,
        )

        self.n = n
        self.nu = nu
        self.nf = nf
        self.nfb = nfb
        self.n_pts = n_pts if n_pts is not None else 4 * n
        self.c_star = c_star
        self.time = time
        self._lam = linear_eigenvalues(n, nu)

        # FFT-truncate the base flow to |k| <= nfb so the per-time
        # operator A(t) = L + 2 B(c*_trunc(t), .) has the same Fourier
        # support as the assembled HB matrix in self.A (which truncates
        # A_k to the same band).  Without this the mixin's HB matvec
        # (IFFT → per-t → FFT) and self.A would disagree by the energy
        # in the discarded harmonics of c*.
        rfft_c = np.fft.rfft(c_star, axis=1)
        rfft_c[:, nfb + 1:] = 0
        self.c_star_trunc = np.fft.irfft(rfft_c, n=n_time, axis=1)

        self.omega = omega
        self.T_period = T_period

        # Perturbation frequencies: -nf*omega, ..., 0, ..., nf*omega
        self.pertb_freqs = self.omega * np.arange(-nf, nf + 1)

        # ── Build A(t) Fourier coefficients and HB matrix ───────────────
        # The HB matrix is kept for solve_linear_system and the
        # shift-invert eigendecomposition (which need an explicit
        # PETSc.Mat to factor with MUMPS).  Matrix-vector products at
        # the HB level are handled by the PDE mixin via the per-time
        # methods below.
        self._build_harmonic_balanced_operator()

        # Eigendecomposition is the slow part of __init__ and is now
        # opt-in: call ``self.compute_eigendecomposition()`` explicitly,
        # or load a cached one from disk (see ``save_eigendecomp.py``
        # and the loader in the workflow scripts).  ``self.L``,
        # ``self.Phi``, ``self.Psi`` and ``self._neutral_proj`` are
        # populated by either path.

    # -----------------------------------------------------------------
    # FFT / IFFT helpers
    # -----------------------------------------------------------------

    def temporal_fft(
        self, X: np.ndarray, nf_out: int,
    ) -> np.ndarray:
        r"""
        Forward FFT along the last axis: time-domain → Fourier coefficients.

        Given ``X`` of shape ``(..., n_time)`` sampled at ``self.time``,
        return the Fourier coefficients for harmonics
        ``-nf_out, ..., 0, ..., nf_out`` with shape ``(..., 2*nf_out+1)``.

        Uses ``np.fft.rfft`` (real base flow) and mirrors negative
        frequencies via conjugate symmetry.
        """
        n_time = X.shape[-1]
        Xhat_pos = np.fft.rfft(X, axis=-1) / n_time  # harmonics 0..n_time//2
        Xhat_pos = Xhat_pos[..., : nf_out + 1]       # keep 0..nf_out

        # Build full two-sided spectrum: -nf_out, ..., -1, 0, 1, ..., nf_out
        Xhat_neg = np.conj(Xhat_pos[..., 1:][..., ::-1])
        return np.concatenate([Xhat_neg, Xhat_pos], axis=-1)

    def temporal_ifft(
        self, Xhat: np.ndarray, t_eval: np.ndarray,
    ) -> np.ndarray:
        r"""
        Inverse FFT: Fourier coefficients → time-domain samples.

        Given ``Xhat`` of shape ``(..., 2*nf+1)`` with harmonics
        ``-nf, ..., 0, ..., nf``, evaluate

        .. math::

            X(t) = \sum_{k=-n_f}^{n_f} \hat{X}_k \, e^{i k \omega t}

        at times ``t_eval``.  Returns array of shape ``(..., len(t_eval))``.
        """
        n_harm = Xhat.shape[-1]
        nf_loc = (n_harm - 1) // 2
        k = np.arange(-nf_loc, nf_loc + 1)
        # Exponential matrix: shape (len(t_eval), n_harm)
        E = np.exp(1j * np.outer(t_eval, k * self.omega))
        # Xhat @ E^T  → shape (..., len(t_eval))
        return Xhat @ E.T

    # -----------------------------------------------------------------
    # Build A(t) in harmonic-balanced form
    # -----------------------------------------------------------------

    def _build_harmonic_balanced_operator(self) -> None:
        r"""
        Compute :math:`A(t_i)` at every time sample, take the temporal
        FFT, write the Fourier coefficients as COO files, then assemble
        the harmonic-balanced matrix and the harmonic resolvent generator.
        """
        comm = self.get_comm()
        n = self.n
        n_time = len(self.time)

        # Evaluate A(t_i) = diag(lam) + 2 * B(c*(t_i), ·) for each t_i
        # Store as (n*n, n_time) where each column is A(t_i) flattened
        As = np.zeros((n * n, n_time))
        A_diag = np.diag(self._lam)
        for ti in range(n_time):
            A_ti = A_diag.copy()
            c_ti = self.c_star[:, ti]
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A_ti[:, j] += 2.0 * self._evaluate_quadratic_term_numpy(
                    e_j, c_ti,
                )
            As[:, ti] = A_ti.ravel()

        # Temporal FFT → positive Fourier coefficients: 0, 1, ..., nfb
        Ashat = np.fft.rfft(As, axis=-1) / n_time
        Ashat = Ashat[:, : self.nfb + 1]

        # Save each Fourier coefficient A_k as COO sparse files
        # (following the pattern in examples/toy_model/generate_matrices.py)
        tmp = "tmp_hb/"
        os.makedirs(tmp, exist_ok=True)

        filenames_lst = []
        for k in range(self.nfb + 1):
            Ak = Ashat[:, k].reshape((n, n))
            Ak_coo = sp.sparse.coo_matrix(Ak)
            rows = Ak_coo.row
            cols = Ak_coo.col
            data = Ak_coo.data

            # Drop near-zero entries
            keep = np.abs(data) >= 1e-16
            rows = rows[keep]
            cols = cols[keep]
            data = data[keep]

            fnames_k = (
                tmp + "rows_%02d.dat" % k,
                tmp + "cols_%02d.dat" % k,
                tmp + "vals_%02d.dat" % k,
            )
            for fname, array, dtype in zip(
                fnames_k,
                [rows, cols, data],
                [np.int32, np.int32, np.complex128],
            ):
                vec = PETSc.Vec().createWithArray(
                    np.asarray(array, dtype=dtype),
                    len(array), None, comm=PETSc.COMM_SELF,
                )
                res4py.write_to_file(fname, vec)
                vec.destroy()

            filenames_lst.append(fnames_k)

        # Assemble the block-Toeplitz HB matrix A_HB.  The HB global
        # size is n * (2*nf + 1); self._state_dim is now the per-time
        # block size, so we build the full HB size explicitly here.
        block_dim = (res4py.compute_local_size(n), n)
        block_sizes = (block_dim, block_dim)
        n_harmonics = 2 * self.nf + 1
        N_hb = n * n_harmonics
        hb_state_dim = (res4py.compute_local_size(N_hb), N_hb)
        full_sizes = (hb_state_dim, hb_state_dim)

        A_hb = res4py.read_harmonic_balanced_matrix(
            filenames_lst, real_bflow=True,
            block_sizes=block_sizes, full_sizes=full_sizes,
        )

        # Assemble the harmonic resolvent generator: L_HB = -D_omega + A_HB
        L_hb = res4py.assemble_harmonic_resolvent_generator(
            A_hb, self.pertb_freqs,
        )
        A_hb.destroy()

        self.A = res4py.linear_operators.MatrixLinearOperator(
            L_hb, nblocks=n_harmonics,
        )

        shutil.rmtree(tmp) if comm.getRank() == 0 else None

    # -----------------------------------------------------------------
    # Quadratic term (pure numpy, single time instant)
    # -----------------------------------------------------------------

    def _evaluate_quadratic_term_numpy(
        self, q1: np.ndarray, q2: np.ndarray,
    ) -> np.ndarray:
        r"""
        Evaluate :math:`B(q_1, q_2)` in pure numpy for a single
        spatial-coefficient pair of shape ``(n,)``.
        """
        n = self.n
        n_pts = self.n_pts
        j = np.arange(1, n + 1, dtype=float)
        half_N = n_pts / 2.0

        def _to_physical(c):
            spec_u = np.zeros(n_pts, dtype=complex)
            spec_ux = np.zeros(n_pts, dtype=complex)
            spec_u[1:n+1] = -1j * half_N * c
            spec_ux[1:n+1] = half_N * j * c
            spec_u[n_pts-n:n_pts] = 1j * half_N * c[::-1]
            spec_ux[n_pts-n:n_pts] = half_N * j[::-1] * c[::-1]
            return ifft(spec_u), ifft(spec_ux)

        u1, u1x = _to_physical(q1)
        u2, u2x = _to_physical(q2)
        B_spec = fft(-0.5 * (u1 * u2x + u2 * u1x))
        result = 2j * B_spec[1:n+1] / n_pts
        if np.isrealobj(q1) and np.isrealobj(q2):
            return result.real
        return result

    # -----------------------------------------------------------------
    # DifferentialEquation interface — per-time-instant methods.
    # The PeriodicDifferentialEquation mixin wraps these with the HB
    # scaffolding (IFFT → per-t → FFT) automatically.
    # -----------------------------------------------------------------

    def evaluate_linear_term(
        self,
        t: float,
        q: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Compute
        :math:`A(t)\, q = (L + 2\,B(c^*_{\text{trunc}}(t),\,\cdot\,))\, q`
        at a single time instant.  Uses ``self.c_star_trunc`` — the
        base flow Fourier-truncated to ``|k| <= nfb`` in
        ``__init__`` — so the result is consistent with the assembled
        HB matrix ``self.A``.  ``t`` is one of the sample times in
        ``self.time``; the truncated ``c^*(t)`` is looked up by
        closest-index match.
        """
        idx = int(np.argmin(np.abs(self.time - t)))
        c_t = self.c_star_trunc[:, idx]

        q_seq = res4py.distributed_to_sequential_vector(q)
        q_arr = q_seq.getArray().copy()

        Aq = self._lam * q_arr + 2.0 * self._evaluate_quadratic_term_numpy(
            c_t, q_arr,
        )

        y_seq = PETSc.Vec().createWithArray(
            np.asarray(Aq, dtype=np.complex128),
            len(Aq), comm=PETSc.COMM_SELF,
        )
        y = q.duplicate() if y is None else y
        y = res4py.sequential_to_distributed_vector(y_seq, y)
        q_seq.destroy()
        y_seq.destroy()
        return y

    def evaluate_quadratic_term(
        self,
        t: float,
        q1: PETSc.Vec,
        q2: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Evaluate :math:`B(q_1, q_2)` at a single time instant.  KSE
        is autonomous in the quadratic, so ``t`` is accepted but
        ignored.
        """
        q1_seq = res4py.distributed_to_sequential_vector(q1)
        q2_seq = res4py.distributed_to_sequential_vector(q2)
        result = self._evaluate_quadratic_term_numpy(
            q1_seq.getArray(), q2_seq.getArray(),
        )
        y_seq = PETSc.Vec().createWithArray(
            np.asarray(result, dtype=np.complex128),
            len(result), comm=PETSc.COMM_SELF,
        )
        y = q1.duplicate() if y is None else y
        y = res4py.sequential_to_distributed_vector(y_seq, y)
        for obj in (q1_seq, q2_seq, y_seq):
            obj.destroy()
        return y

    def solve_linear_system(
        self, s: complex, b: PETSc.Vec,
        x: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Solve :math:`(sI - \mathcal{L})\,x = b` after projecting out
        the neutrally stable Floquet direction:
        :math:`b \leftarrow (I - v\,w^*)\,b`.
        """
        # # Project out all neutral Floquet directions from the RHS
        # if hasattr(self, "_neutral_proj"):
        #     b = self._neutral_proj.apply(b)

        M = self.A.A.copy()
        M.scale(-1.0)
        # self.A is sized at the HB level (n * nblocks).  Use its row
        # size directly — self.get_state_dimension() now returns the
        # per-time block size after the PDE-mixin wrap.
        size = M.getSizes()[0]
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(s, I)
        I.destroy()
        # ICNTL(13)=1 disables MUMPS's parallel (ScaLAPACK) root-node
        # factorization — works around an MPICH/ScaLAPACK assertion
        # observed on macOS with complex MUMPS in parallel.
        ksp = res4py.create_mumps_solver(M, icntl={13: 1})
        res4py.check_lu_factorization(M, ksp)
        Lop = res4py.linear_operators.MatrixLinearOperator(M, ksp)
        x = Lop.solve(b, x)
        Lop.destroy()
        return x

    # The Floquet eigendecomposition and the neutral-projection
    # operator are computed off-class in ``eigendecomp_kse.py``; the
    # user is expected to set ``self.L``, ``self.Phi``, ``self.Psi``,
    # and ``self._neutral_proj`` from there (either by recomputing or
    # by loading a previously cached result from disk).
    # ``solve_linear_system`` above already gates the neutral
    # projection behind a ``hasattr(self, "_neutral_proj")`` check, so
    # both code paths work without those attributes being present.
