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

        # Harmonic-balanced state dimension: n * (2*nf + 1)
        n_harmonics = 2 * nf + 1
        N_hb = n * n_harmonics
        state_dim = (res4py.compute_local_size(N_hb), N_hb)
        super().__init__(comm, "KuramotoSivashinskyPeriodic", state_dim, 2)

        self.n = n
        self.nu = nu
        self.nf = nf
        self.nfb = nfb
        self.n_pts = n_pts if n_pts is not None else 4 * n
        self.c_star = c_star
        self.time = time
        self._lam = linear_eigenvalues(n, nu)

        T_period = time[-1] + (time[1] - time[0])
        self.omega = 2 * np.pi / T_period
        self.T_period = T_period

        # Perturbation frequencies: -nf*omega, ..., 0, ..., nf*omega
        self.pertb_freqs = self.omega * np.arange(-nf, nf + 1)

        # ── Build A(t) Fourier coefficients and HB matrix ───────────────
        self._build_harmonic_balanced_operator()

        # ── Eigendecomposition of the harmonic resolvent generator ───────
        self.L, self.Phi, self.Psi = self.compute_eigendecomposition()
        self.L = np.diag(self.L)

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

        # Assemble the block-Toeplitz HB matrix A_HB
        block_dim = (res4py.compute_local_size(n), n)
        block_sizes = (block_dim, block_dim)
        state_dim = self.get_state_dimension()
        full_sizes = (state_dim, state_dim)

        A_hb = res4py.read_harmonic_balanced_matrix(
            filenames_lst, real_bflow=True,
            block_sizes=block_sizes, full_sizes=full_sizes,
        )

        # Assemble the harmonic resolvent generator: L_HB = -D_omega + A_HB
        L_hb = res4py.assemble_harmonic_resolvent_generator(
            A_hb, self.pertb_freqs,
        )
        A_hb.destroy()

        n_harmonics = 2 * self.nf + 1
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
    # DifferentialEquation interface
    # -----------------------------------------------------------------

    def evaluate_linear_term(
        self, q: PETSc.Vec, y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""Apply the harmonic resolvent generator to an HB vector."""
        return self.A.apply(q, y)

    def evaluate_quadratic_term(
        self, q1: PETSc.Vec, q2: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Evaluate the quadratic term in harmonic-balanced form:
        IFFT both inputs to the time domain, evaluate :math:`B` at
        each time sample, then FFT back.
        """
        n = self.n
        nf = self.nf
        n_harmonics = 2 * nf + 1
        n_time = len(self.time)

        # Gather the HB vectors to sequential arrays
        q1_seq = res4py.distributed_to_sequential_vector(q1)
        q2_seq = res4py.distributed_to_sequential_vector(q2)
        q1_arr = q1_seq.getArray().copy().reshape(n_harmonics, n)  # (n_harm, n)
        q2_arr = q2_seq.getArray().copy().reshape(n_harmonics, n)
        q1_seq.destroy()
        q2_seq.destroy()

        # IFFT: (n_harm, n) → (n_time, n)
        # Each row k of q_arr corresponds to harmonic k-nf
        # q(t_i) = sum_k q_hat_k * exp(i*k*omega*t_i)
        k_idx = np.arange(-nf, nf + 1)
        E = np.exp(1j * np.outer(self.time, k_idx * self.omega))  # (n_time, n_harm)
        q1_t = E @ q1_arr  # (n_time, n)
        q2_t = E @ q2_arr  # (n_time, n)

        # Evaluate B(q1(t_i), q2(t_i)) at each time sample
        B_t = np.zeros((n_time, n), dtype=complex)
        for ti in range(n_time):
            B_t[ti, :] = self._evaluate_quadratic_term_numpy(
                q1_t[ti, :], q2_t[ti, :],
            )

        # FFT back: (n_time, n) → (n_harm, n)
        # B_hat_k = (1/n_time) * sum_i B(t_i) * exp(-i*k*omega*t_i)
        E_inv = np.exp(-1j * np.outer(k_idx * self.omega, self.time)) / n_time
        B_hat = E_inv @ B_t  # (n_harm, n)

        # Assemble into an HB PETSc vector
        result = B_hat.ravel()
        y_seq = PETSc.Vec().createWithArray(
            np.asarray(result, dtype=np.complex128),
            len(result), comm=PETSc.COMM_SELF,
        )
        y = q1.duplicate() if y is None else y
        y = res4py.sequential_to_distributed_vector(y_seq, y)
        y_seq.destroy()
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
        # Project out all neutral Floquet directions from the RHS
        if hasattr(self, "_neutral_proj"):
            b = self._neutral_proj.apply(b)

        M = self.A.A.copy()
        M.scale(-1.0)
        size = self.get_state_dimension()
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(s, I)
        I.destroy()
        ksp = res4py.create_mumps_solver(M)
        res4py.check_lu_factorization(M, ksp)
        L = res4py.linear_operators.MatrixLinearOperator(M, ksp)
        x = L.solve(b, x)
        L.destroy()
        return x

    def _shift_invert_eig(
        self,
        sigma: complex,
        n_evals: int,
        krylov_dim: int,
    ) -> Tuple[np.ndarray, SLEPc.BV, np.ndarray, SLEPc.BV]:
        r"""
        Shift-and-invert Arnoldi about ``sigma``.

        Returns matched and biorthogonalised ``(Dv, V, Dw, W)``
        where ``Dv``, ``Dw`` are diagonal matrices.
        """
        M = self.A.A.copy()
        M.scale(-1.0)
        size = self.get_state_dimension()
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(sigma, I)
        I.destroy()
        ksp = res4py.create_mumps_solver(M)
        res4py.check_lu_factorization(M, ksp)
        n_harmonics = 2 * self.nf + 1
        Linv = res4py.linear_operators.MatrixLinearOperator(
            M, ksp, nblocks=n_harmonics,
        )

        Dv, V = res4py.linalg.eig(
            Linv, Linv.solve, krylov_dim, n_evals,
            process_evals=lambda mu: sigma - 1.0 / mu,
        )
        Dw, W = res4py.linalg.eig(
            Linv, Linv.solve_hermitian_transpose, krylov_dim, n_evals,
            process_evals=lambda mu: np.conj(sigma) - 1.0 / mu,
        )
        Linv.destroy()

        V, W, Dv, Dw = res4py.linalg.match_right_and_left_eigenvectors(
            V, W, Dv, Dw,
        )
        return Dv, V, Dw, W

    def _compute_neutral_eigentriples(
        self,
        n_evals: int = 4,
        krylov_dim: int = 50,
    ) -> None:
        r"""
        Compute the neutrally stable Floquet eigentriples at
        :math:`\pm i k\omega` for :math:`k \in \{0,1,2,3,4,5\}` and
        store them for projection in :meth:`solve_linear_system`.

        Each shift locates the eigenvalue closest to :math:`i k\omega`;
        the biorthogonalised pair ``(v, w)`` is kept.
        """
        v_list = []
        w_list = []

        for k in range(6):
            for sign in ([0] if k == 0 else [1, -1]):
                sigma = sign * k * 1j * self.omega
                Dv, V, _, W = self._shift_invert_eig(
                    sigma, n_evals, krylov_dim,
                )
                evals = np.diag(Dv)
                idx = np.argmin(np.abs(evals - sigma))

                v_col = V.getColumn(idx)
                v_list.append(v_col.copy())
                V.restoreColumn(idx, v_col)

                w_col = W.getColumn(idx)
                w_list.append(w_col.copy())
                W.restoreColumn(idx, w_col)

        # Assemble into BVs for the ProjectionLinearOperator
        n_neutral = len(v_list)
        state_dim = self.get_state_dimension()
        comm = self.get_comm()

        V_neutral = SLEPc.BV().create(comm=comm)
        V_neutral.setSizes(state_dim, n_neutral)
        V_neutral.setType("mat")
        W_neutral = SLEPc.BV().create(comm=comm)
        W_neutral.setSizes(state_dim, n_neutral)
        W_neutral.setType("mat")

        for i in range(n_neutral):
            V_neutral.insertVec(i, v_list[i])
            W_neutral.insertVec(i, w_list[i])
            v_list[i].destroy()
            w_list[i].destroy()

        # Check W^* A V for the neutral directions
        comm = self.get_comm()
        AV = V_neutral.copy()
        for i in range(n_neutral):
            v = V_neutral.getColumn(i)
            av = AV.getColumn(i)
            av = self.evaluate_linear_term(v, av)
            AV.restoreColumn(i, av)
            V_neutral.restoreColumn(i, v)
        WtAV = AV.dot(W_neutral)
        res4py.petscprint(comm, "W^* A V (neutral directions):")
        res4py.petscprint(comm, np.diag(WtAV.getDenseArray()))
        WtAV.destroy()
        AV.destroy()

        n_harmonics = 2 * self.nf + 1
        self._neutral_proj = res4py.linear_operators.ProjectionLinearOperator(
            V_neutral, W_neutral, complement=True, nblocks=n_harmonics,
        )

    def compute_eigendecomposition(
        self,
        n_evals: int = 200,
        krylov_dim: int = 600,
        sigma: complex = 0.0,
    ) -> Tuple[np.ndarray, SLEPc.BV, SLEPc.BV]:
        r"""
        Compute Floquet exponents closest to ``sigma`` via shift-and-invert
        Arnoldi.  First computes and removes the neutrally stable directions
        at :math:`\pm i k\omega` for :math:`k = 0, \ldots, 5`.

        :param n_evals: number of eigenvalues to compute
        :param krylov_dim: Krylov subspace dimension (must be > n_evals)
        :param sigma: shift for the main eigenvalue computation
        """
        # Compute neutral Floquet directions across all strips
        self._compute_neutral_eigentriples()

        # Main eigenvalue computation
        Dv, V, _, W = self._shift_invert_eig(sigma, n_evals, krylov_dim)

        # Keep only Floquet exponents in the principal strip
        # |Im(lambda)| <= omega/2, sorted by descending Re(lambda)
        evals = np.diag(Dv)
        half_omega = self.omega / 2.0
        in_strip = np.abs(evals.imag) <= half_omega + 1e-10
        idces = np.where(in_strip)[0]
        idces = idces[np.argsort(-evals[idces].real)]

        Dv = np.diag(evals[idces])
        V = res4py.bv_slice(V, idces.astype(np.int32))
        W = res4py.bv_slice(W, idces.astype(np.int32))

        # Remove the neutral direction (eigenvalue closest to 0)
        evals_strip = np.diag(Dv)
        idx_neutral = np.argmin(np.abs(evals_strip))
        keep = np.delete(np.arange(len(evals_strip)), idx_neutral)
        Dv = np.diag(evals_strip[keep])
        V = res4py.bv_slice(V, keep.astype(np.int32))
        W = res4py.bv_slice(W, keep.astype(np.int32))

        return Dv, V, W
