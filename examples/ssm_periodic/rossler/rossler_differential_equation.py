"""
Harmonic-balanced ``DifferentialEquation`` for the Rössler system
(Padovan & Rowley 2022, Sec. IV) linearised about a T-periodic orbit.

State dimension is fixed at n = 3.  The per-time linear operator
A(t) = M + 2 B(q*(t), ·) and the symmetric bilinear B come from
``rossler_rhs``.  The HB matrix is assembled from the time-FFT of
A(t_i) using the same COO-file pipeline as the KSE example.
"""

import os
import shutil
from typing import Optional, Tuple

import numpy as np
import scipy as sp

from petsc4py import PETSc
import resolvent4py as res4py

from resolvent4py.spectral_submanifold import DifferentialEquation
from rossler_rhs import (
    linear_matrix, quadratic_bilinear, perturbation_linear_action,
)


N_STATE = 3   # Rössler is 3-D


class RosslerPeriodic(DifferentialEquation):
    r"""
    Rössler system linearised about a T-periodic orbit, in
    harmonic-balanced form.

    :param c: Rössler parameter (paper uses ``c = 5.3``).
    :param nf: number of positive perturbation frequencies.
    :param nfb: number of positive base-flow frequencies to retain.
    :param c_star: T-periodic base flow, shape ``(3, n_time)``.
    :param time: uniformly spaced times on ``[0, T)``.
    """

    def __init__(
        self,
        c: float,
        nf: int,
        nfb: int,
        c_star: np.ndarray,
        time: np.ndarray,
    ) -> None:
        comm = PETSc.COMM_WORLD

        n_time = len(time)
        if n_time < 2 * nf + 1:
            raise ValueError(
                f"len(time) = {n_time} < 2*nf+1 = {2*nf+1}."
            )
        if c_star.shape != (N_STATE, n_time):
            raise ValueError(
                f"c_star.shape = {c_star.shape}, expected ({N_STATE}, {n_time})."
            )
        if nfb > nf:
            raise ValueError(f"nfb={nfb} must be <= nf={nf}.")

        state_dim = (res4py.compute_local_size(N_STATE), N_STATE)
        T_period = time[-1] + (time[1] - time[0])
        omega = 2 * np.pi / T_period

        super().__init__(
            comm, "RosslerPeriodic", state_dim, 2,
        )

        self.c = c
        self.n = N_STATE
        self.nf = nf
        self.nfb = nfb
        self.c_star = c_star
        self.time = time
        self.omega = omega
        self.T_period = T_period
        self._M = linear_matrix(c)

        # FFT-truncate the base flow to |k| <= nfb so the per-time
        # operator A(t) = M + 2 B(c*_trunc(t), .) has the same Fourier
        # support as the assembled HB matrix.
        rfft_c = np.fft.rfft(c_star, axis=1)
        rfft_c[:, nfb + 1:] = 0
        self.c_star_trunc = np.fft.irfft(rfft_c, n=n_time, axis=1)

        self.pertb_freqs = self.omega * np.arange(-nf, nf + 1)

        self._build_harmonic_balanced_operator()

    # -----------------------------------------------------------------
    # HB matrix assembly
    # -----------------------------------------------------------------

    def _build_harmonic_balanced_operator(self) -> None:
        comm = self.get_comm()
        n = self.n
        n_time = len(self.time)

        # Evaluate A(t_i) = M + 2 B(c*(t_i), ·) for each t_i, store as
        # (n*n, n_time) where each column is A(t_i) flattened row-major.
        As = np.zeros((n * n, n_time))
        for ti in range(n_time):
            A_ti = self._M.copy()
            c_ti = self.c_star[:, ti]
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A_ti[:, j] += 2.0 * quadratic_bilinear(c_ti, e_j)
            As[:, ti] = A_ti.ravel()

        Ashat = np.fft.rfft(As, axis=-1) / n_time
        Ashat = Ashat[:, : self.nfb + 1]

        tmp = "tmp_hb/"
        os.makedirs(tmp, exist_ok=True)

        filenames_lst = []
        for k in range(self.nfb + 1):
            Ak = Ashat[:, k].reshape((n, n))
            Ak_coo = sp.sparse.coo_matrix(Ak)
            rows = Ak_coo.row
            cols = Ak_coo.col
            data = Ak_coo.data

            keep = np.abs(data) >= 1e-16
            rows, cols, data = rows[keep], cols[keep], data[keep]

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
        L_hb = res4py.assemble_harmonic_resolvent_generator(
            A_hb, self.pertb_freqs,
        )
        A_hb.destroy()

        self.A = res4py.linear_operators.MatrixLinearOperator(
            L_hb, nblocks=n_harmonics,
        )

        shutil.rmtree(tmp) if comm.getRank() == 0 else None

    # -----------------------------------------------------------------
    # DifferentialEquation interface (per-time-instant)
    # -----------------------------------------------------------------

    def evaluate_linear_term(
        self, t: float, q: PETSc.Vec, y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        idx = int(np.argmin(np.abs(self.time - t)))
        c_t = self.c_star_trunc[:, idx]

        q_seq = res4py.distributed_to_sequential_vector(q)
        q_arr = q_seq.getArray().copy()

        Aq = perturbation_linear_action(c_t, q_arr, self.c)

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
        self, t: float, q1: PETSc.Vec, q2: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        q1_seq = res4py.distributed_to_sequential_vector(q1)
        q2_seq = res4py.distributed_to_sequential_vector(q2)
        result = quadratic_bilinear(q1_seq.getArray(), q2_seq.getArray())

        if np.isrealobj(q1_seq.getArray()) and np.isrealobj(q2_seq.getArray()):
            result = result.astype(np.complex128)

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
        self, s: complex, b: PETSc.Vec, x: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        M = self.A.A.copy()
        M.scale(-1.0)
        size = M.getSizes()[0]
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(s, I)
        I.destroy()
        ksp = res4py.create_mumps_solver(M, icntl={13: 1})
        res4py.check_lu_factorization(M, ksp)
        Lop = res4py.linear_operators.MatrixLinearOperator(M, ksp)
        x = Lop.solve(b, x)
        Lop.destroy()
        return x
