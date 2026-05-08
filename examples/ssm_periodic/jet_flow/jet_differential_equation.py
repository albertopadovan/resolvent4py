from typing import Any, Optional, Tuple

import numpy as np

from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py

from resolvent4py.spectral_submanifold import DifferentialEquation


class JetFlowPeriodic(DifferentialEquation):
    r"""
    Harmonic-balanced formulation of the 2D incompressible Navier-Stokes
    equations linearised about a time-periodic forced jet base flow
    :math:`c^*(t) = c^*(t + T)`.

    The perturbation dynamics are

    .. math::

        \dot{v} = A(t)\,v + B(v, v),\qquad A(t) = A(t + T),

    where :math:`A(t)` is the Jacobian of the (DAE) Navier-Stokes
    right-hand side about :math:`c^*(t)` and :math:`B` is the constant-
    coefficient quadratic advection (the BC-induced affine term cancels
    in the perturbation equation).  In the harmonic-balanced
    representation the state becomes a stacked vector of Fourier
    coefficients

    .. math::

        \hat{v} = (\hat{v}_{-n_f}, \ldots, \hat{v}_0, \ldots, \hat{v}_{n_f})

    with global size ``n*(2*nf+1)``, where ``n = n_vel + n_p_pinned``.

    The Fourier coefficients :math:`A_k` of :math:`A(t)` are produced
    once by ``incompreso.extract_jacobian_fourier_coefficients``, then
    assembled into the harmonic resolvent generator
    :math:`\mathcal{L} = -\mathrm{diag}(i k\omega I) + \mathcal{A}` by
    a sibling script (via :func:`resolvent4py.read_harmonic_balanced_matrix`
    and :func:`resolvent4py.assemble_harmonic_resolvent_generator`).
    The wrapped operator is passed in as ``A``.

    The quadratic :math:`B(v, v)` is evaluated, on rank 0 only, by
    polarisation of ``spops.evaluate_right_hand_side`` from
    ``incompreso``.  The result is broadcast to all ranks before the
    final scatter to a distributed PETSc vector.

    :param spops: ``incompreso.SpatialOperators`` instance built (e.g.
        via ``incompreso.parse_input_file``) by the caller.  Required
        on every rank — incompreso is serial, but a replicated copy
        per rank is cheap and lets ``n_vel`` / ``n_p_pinned`` be
        inferred from ``spops.mesh``.  The quadratic is evaluated on
        rank 0 only.
    :param A: pre-assembled harmonic resolvent generator
        :math:`\mathcal{L} = -\mathrm{diag}(i k\omega I) + \mathcal{A}`
        as a :class:`petsc4py.PETSc.Mat`.  Must be square; the number
        of Fourier harmonics ``nf`` is inferred from its global size,
        which must equal ``(n_vel + n_p_pinned) * (2*nf + 1)``.
    :param Phi: right Floquet eigenvectors as a
        :class:`slepc4py.SLEPc.BV`, biorthogonalised against ``Psi``.
    :param Psi: left Floquet eigenvectors as a
        :class:`slepc4py.SLEPc.BV`, biorthogonalised against ``Phi``.
    :param L: Floquet exponents as a 1D ``numpy`` array of complex
        values; stored internally as ``np.diag(L)``.
    :param time: time grid sampled uniformly on ``[0, T)`` with
        ``len(time) >= 2*nf + 1`` (Nyquist).  ``omega = 2*pi/T`` and
        ``T_period`` are inferred from this array.
    """

    def __init__(
        self,
        spops: Any,
        A: "PETSc.Mat",
        M: "PETSc.Mat",
        Phi: "SLEPc.BV",
        Psi: "SLEPc.BV",
        L: np.ndarray,
        time: np.ndarray,
    ) -> None:
        self._spops = spops
        mesh = self._spops.mesh
        self.n_vel = int(mesh.u_int.size + mesh.v_int.size)
        self.n_p_pinned = int(mesh.p.size - 1)
        self.n = self.n_vel + self.n_p_pinned

        N_hb, _ = A.getSize()
        if N_hb % self.n != 0:
            raise ValueError(
                f"A global size {N_hb} not divisible by n = {self.n}."
            )
        self.nblocks = N_hb // self.n
        if self.nblocks % 2 != 1:
            raise ValueError(
                f"A block count {self.nblocks} must be odd (= 2*nf + 1)."
            )
        self.nf = (self.nblocks - 1) // 2

        if len(time) < 2 * self.nf + 1:
            raise ValueError(
                f"len(time)={len(time)} < 2*nf+1={2*self.nf+1} "
                f"violates Nyquist."
            )

        state_dim = A.getSizes()[0]
        super().__init__(A.getComm(), "JetFlowPeriodic", state_dim, 2)

        self.A = A
        self.M = M
        self._time = np.asarray(time)
        self._n_time = len(time)
        dt = float(self._time[1] - self._time[0])
        self.T_period = float(self._time[-1]) + dt
        self.omega = 2.0 * np.pi / self.T_period

        self.Phi = Phi
        self.Psi = Psi
        self.L = np.diag(L)

    # -----------------------------------------------------------------
    # Quadratic term — bilinear B(q1, q2) at a single time instant
    # -----------------------------------------------------------------

    def _evaluate_advection(
        self, q: np.ndarray,
    ) -> np.ndarray:
        r"""
        Evaluate the Navier-Stokes advection (the quadratic part of
        the RHS) for the velocity state ``q`` at fixed :math:`t = 0`.

        Reuses :meth:`SpatialOperators.evaluate_right_hand_side` —
        which already populates ``spops.mesh`` and imposes BCs via
        :func:`incompreso.utils.helpers.vector_to_fields` — but
        temporarily sets ``spops.Re = 1e16`` so the diffusion term
        :math:`(1/Re)\,\Delta q` is numerically zero and the result
        is effectively :math:`M^{-1}` times the advection integral.
        ``spops.Re`` is restored on the way out.

        :param q: real velocity-only vector of length ``n_vel``.
        :return: :math:`M^{-1}` times advection, length ``n_vel``.
        Rank 0 only.
        """
        spops = self._spops
        Re_save = spops.Re
        spops.Re = 1e16
        qnl = spops.evaluate_right_hand_side(0.0, q)
        spops.Re = Re_save
        return qnl

    def _evaluate_quadratic_term(
        self, q1: np.ndarray, q2: np.ndarray,
    ) -> np.ndarray:
        r"""
        Evaluate the symmetric bilinear advection :math:`B(q_1, q_2)`
        via polarisation of :meth:`_evaluate_advection`:

        .. math::

            2\,B(q_1, q_2) = \mathrm{adv}(q_1 + q_2)
                           - \mathrm{adv}(q_1) - \mathrm{adv}(q_2).

        The perturbation satisfies homogeneous Dirichlet BCs, so
        :math:`\mathrm{adv}(0) = 0` and the standard quadratic
        polarisation applies (no affine correction needed).

        :param q1: real velocity-only vector of length ``n_vel``.
        :param q2: real velocity-only vector of length ``n_vel``.
        :return: real velocity-only vector of length ``n_vel``.
        Rank 0 only.
        """
        return 0.5 * (
            self._evaluate_advection(q1 + q2)
            - self._evaluate_advection(q1)
            - self._evaluate_advection(q2)
        )

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
        Evaluate the quadratic term in harmonic-balanced form.

        IFFT both inputs to the time domain on rank 0, evaluate
        :math:`B(q_1(t_i), q_2(t_i))` using ``incompreso`` at each time
        sample, FFT back, broadcast the result, and scatter to a
        distributed PETSc vector.
        """
        comm = self.get_comm()
        rank = comm.getRank()
        n_time = self._n_time

        q1_seq = res4py.distributed_to_sequential_vector(q1)
        q2_seq = res4py.distributed_to_sequential_vector(q2)
        result = np.empty(self.nblocks * self.n_vel, dtype=np.complex128)
        if rank == 0:
            q1_arr = q1_seq.getArray().reshape((self.nblocks, self.n_vel))
            q2_arr = q2_seq.getArray().reshape((self.nblocks, self.n_vel))

            k_idx = np.arange(-self.nf, self.nf + 1)
            E = np.exp(1j * np.outer(self._time, k_idx * self.omega))
            q1_t = E @ q1_arr
            q2_t = E @ q2_arr

            B_t = np.zeros((n_time, self.n_vel), dtype=np.complex128)
            for ti in range(n_time):
                B_t[ti, :] = self._evaluate_quadratic_term(
                    q1_t[ti, :], q2_t[ti, :],
                )

            E_inv = np.exp(
                -1j * np.outer(k_idx * self.omega, self._time),
            ) / n_time
            B_hat = E_inv @ B_t
            result = np.ascontiguousarray(B_hat.ravel(), dtype=np.complex128)
            
        q1_seq.destroy()
        q2_seq.destroy()

        comm.tompi4py().Bcast(result, root=0)

        y_seq = PETSc.Vec().createWithArray(
            result, len(result), comm=PETSc.COMM_SELF,
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
        Solve :math:`(sI - \mathcal{L})\,x = b` directly via MUMPS.

        No neutral-direction projection is needed: the jet is forced
        periodically, so there is no phase-shift gauge symmetry and
        :math:`\mathcal{L}` has no exact zero (or :math:`\pm i k\omega`)
        Floquet exponents.
        """
        M = self.A.copy()
        M.scale(-1.0)
        size = self.get_state_dimension()
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(s, I)
        I.destroy()
        ksp = res4py.create_mumps_solver(M)
        res4py.check_lu_factorization(M, ksp)
        Lop = res4py.linear_operators.MatrixLinearOperator(M, ksp)
        x = Lop.solve(b, x)
        Lop.destroy()
        return x
