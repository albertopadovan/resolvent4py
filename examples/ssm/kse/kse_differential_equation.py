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


class KuramotoSivashinsky(DifferentialEquation):
    r"""
    Kuramoto-Sivashinsky equation on :math:`[0, 2\pi]`:

    .. math::

        u_t = -u\,u_x - u_{xx} - \nu\,u_{xxxx}

    with sine-series spectral ansatz
    :math:`u(x,t) = \sum_{j=1}^{n} c_j(t) \sin(jx)`.

    The linear operator is diagonal: :math:`\lambda_j = j^2 - \nu j^4`.
    The quadratic bilinear form is the polarisation of
    :math:`N(u) = -u\,u_x`.

    When :math:`c^*` is provided, the system is formulated in perturbation
    variables :math:`v = c - c^*` around the equilibrium :math:`c^*`:

    .. math::

        \dot{v} = \bigl(L + 2\,B(c^*,\,\cdot\,)\bigr)\,v + B(v, v)

    :param n: number of sine modes
    :type n: int
    :param nu: viscosity parameter :math:`\nu`
    :type nu: float
    :param n_pts: physical-space grid points (default: :math:`4n`)
    :type n_pts: int, optional
    :param c_star: equilibrium sine coefficients of shape :math:`(n,)`.
        If provided, the system is linearised about this equilibrium.
    :type c_star: np.ndarray, optional
    """

    def __init__(
        self,
        n: int,
        nu: float,
        n_pts: int = None,
        c_star: np.ndarray = None,
    ) -> None:
        comm = PETSc.COMM_WORLD
        state_dim = (res4py.compute_local_size(n), n)
        super().__init__(comm, "KuramotoSivashinsky", state_dim, 2)

        self.n = n
        self.nu = nu
        self.n_pts = n_pts if n_pts is not None else 4 * n
        self._lam = linear_eigenvalues(n, nu)
        self.c_star = c_star

        # Build A as a numpy dense matrix
        if c_star is None:
            A_np = np.diag(self._lam)
        else:
            A_np = np.diag(self._lam)
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A_np[:, j] += 2.0 * self._evaluate_quadratic_term_numpy(
                    e_j, c_star
                )

        A_coo = sp.sparse.coo_matrix(A_np.astype(np.complex128))
        A_coo.eliminate_zeros()
        A_petsc = res4py.assemble_matrix_from_coo(
            comm, [A_coo.row, A_coo.col, A_coo.data], (state_dim, state_dim)
        )
        self.A = res4py.linear_operators.MatrixLinearOperator(A_petsc)
        self.L, self.Phi, self.Psi = self.compute_eigendecomposition()
        self.L = np.diag(self.L)

    def _evaluate_quadratic_term_numpy(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> np.ndarray:
        r"""
        Evaluate :math:`B(q_1, q_2)` in pure numpy (used during
        ``__init__`` to build the Jacobian at :math:`c^*`).
        """
        j = np.arange(1, self.n + 1, dtype=float)
        half_N = self.n_pts / 2.0

        def _to_physical(c):
            spec_u = np.zeros(self.n_pts, dtype=complex)
            spec_ux = np.zeros(self.n_pts, dtype=complex)
            spec_u[1 : self.n + 1] = -1j * half_N * c
            spec_ux[1 : self.n + 1] = half_N * j * c
            spec_u[self.n_pts - self.n : self.n_pts] = 1j * half_N * c[::-1]
            spec_ux[self.n_pts - self.n : self.n_pts] = (
                half_N * j[::-1] * c[::-1]
            )
            return ifft(spec_u), ifft(spec_ux)

        u1, u1x = _to_physical(q1)
        u2, u2x = _to_physical(q2)
        B_spec = fft(-0.5 * (u1 * u2x + u2 * u1x))
        result = 2j * B_spec[1 : self.n + 1] / self.n_pts
        if np.isrealobj(q1) and np.isrealobj(q2):
            return result.real
        return result

    def evaluate_linear_term(
        self,
        t: float,
        q: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        # Autonomous system: t argument is accepted (for the abstract
        # signature) but ignored.
        return self.A.apply(q, y)

    def evaluate_quadratic_term(
        self,
        t: float,
        q1: np.ndarray,
        q2: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Autonomous bilinear: t argument is accepted but ignored.
        result = self._evaluate_quadratic_term_numpy(q1, q2)
        result = np.asarray(result, dtype=np.complex128)
        if y is not None:
            y[:] = result
            return y
        return result

    def solve_linear_system(
        self,
        s: complex,
        b: PETSc.Vec,
        x: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        M = self.A.A.copy()
        M.scale(-1.0)
        size = self.get_state_dimension()
        I = res4py.create_AIJ_identity(self.get_comm(), (size, size))
        M.axpy(s, I)
        I.destroy()
        # ICNTL(13)=1 disables ScaLAPACK at the root frontal — required
        # whenever the matrix is small enough that ScaLAPACK chokes
        # (MUMPS INFOG(1)=-3 otherwise).  At n=64 this fires already on
        # 4 ranks; safe and effectively free at any scale.
        ksp = res4py.create_mumps_solver(M, icntl={13: 1})
        res4py.check_lu_factorization(M, ksp)
        L = res4py.linear_operators.MatrixLinearOperator(M, ksp)
        x = L.solve(b, x)
        L.destroy()
        return x

    def compute_eigendecomposition(
        self,
    ) -> Tuple[np.ndarray, SLEPc.BV, SLEPc.BV]:
        N = self.get_state_dimension()[-1]
        Dv, V = res4py.linalg.eig(
            self.A,
            self.A.apply,
            N,
            N,
            lambda x: x,
        )
        Dw, W = res4py.linalg.eig(
            self.A,
            self.A.apply_hermitian_transpose,
            N,
            N,
            lambda x: x,
        )
        V, W, Dv, Dw = res4py.linalg.match_right_and_left_eigenvectors(
            V,
            W,
            Dv,
            Dw,
        )
        return Dv, V, W

    def evaluate_dynamics_numpy(
        self,
        t: float,
        q: np.ndarray,
    ) -> np.ndarray:
        q_seq = PETSc.Vec().createWithArray(
            np.asarray(q, dtype=np.complex128),
            len(q),
            comm=PETSc.COMM_SELF,
        )
        q_dist = PETSc.Vec().create(comm=self.get_comm())
        q_dist.setSizes(self.get_state_dimension())
        q_dist.setFromOptions()
        q_dist = res4py.sequential_to_distributed_vector(q_seq, q_dist)
        y_dist = self.evaluate_dynamics(t, q_dist)
        y_seq = res4py.distributed_to_sequential_vector(y_dist)
        result = y_seq.getArray().copy().real
        objs = [q_seq, q_dist, y_dist, y_seq]
        for obj in objs:
            obj.destroy()
        return result
