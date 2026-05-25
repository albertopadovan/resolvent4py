import os
import shutil
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py

from resolvent4py.spectral_submanifold import DifferentialEquation


class Hopf3D(DifferentialEquation):
    """
    3D Hopf normal form:
        x_dot =  mu*x - y - alpha*x*z - beta*x*y
        y_dot =  x + mu*y - alpha*y*z + beta*x^2
        z_dot = -alpha*z + alpha*(x^2 + y^2)
    """
    
    def __init__(
        self,
        mu: float = -0.2,
        alpha: float = 0.15,
        beta: float = 0.0,
    ) -> None:
        
        comm = PETSc.COMM_WORLD
        name = "Hopf3D"
        N = 3
        state_dim = (res4py.compute_local_size(N), N)
        poly_deg = 2
        super().__init__(comm, name, state_dim, poly_deg)

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        
        # Build the A matrix as a distributed PETSc matrix via res4py COO I/O
        tmp = "tmp/"
        os.makedirs(tmp, exist_ok=True)
        A_coo = sp.sparse.coo_matrix(
            np.array([[mu, -1, 0], [1, mu, 0], [0, 0, -alpha]])
        )
        fnames = [tmp + "rows.dat", tmp + "cols.dat", tmp + "vals.dat"]
        dtypes = [np.int32, np.int32, np.complex128]
        for (i, zipped) in enumerate(zip(fnames, [A_coo.row, A_coo.col, A_coo.data])):
            fname, array = zipped
            vec = PETSc.Vec().createWithArray(
                np.asarray(array, dtype=dtypes[i]), len(array), None, comm=PETSc.COMM_SELF
            )
            res4py.write_to_file(fname, vec)
            vec.destroy()

        sizes = (state_dim, state_dim)
        A_petsc = res4py.read_coo_matrix(fnames, sizes)
        self.A = res4py.linear_operators.MatrixLinearOperator(A_petsc)
        shutil.rmtree(tmp) if comm.getRank() == 0 else None

    def evaluate_linear_term(
        self, t: float, q: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        # Autonomous system: t argument is accepted (for the abstract
        # signature) but ignored.
        return self.A.apply(q, y)

    def evaluate_quadratic_term(
        self,
        t: float,
        q1: PETSc.Vec,
        q2: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        # Autonomous bilinear: t argument is accepted but ignored.
        q1seq = res4py.distributed_to_sequential_vector(q1)
        q2seq = res4py.distributed_to_sequential_vector(q2)
        alpha, beta = self.alpha, self.beta
        x1, y1, z1 = q1seq.getArray()
        x2, y2, z2 = q2seq.getArray()
        y_np = np.zeros(3, dtype=np.complex128)
        y_np[0] = -0.5 * alpha * (x1 * z2 + x2 * z1) - 0.5 * beta * (
            x1 * y2 + x2 * y1
        )
        y_np[1] = -0.5 * alpha * (y1 * z2 + y2 * z1) + beta * x1 * x2
        y_np[2] = alpha * (x1 * x2 + y1 * y2)
        y_seq = PETSc.Vec().createWithArray(
            y_np, len(y_np), comm=PETSc.COMM_SELF
        )
        y = q1.duplicate() if y is None else y
        y = res4py.sequential_to_distributed_vector(y_seq, y)
        return y

    def solve_linear_system(
        self, s: complex, b: PETSc.Vec, x: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        # Build (sI - A)
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

    def compute_eigendecomposition(
        self,
    ) -> Tuple[np.ndarray, SLEPc.BV, SLEPc.BV]:
        N = self.get_state_dimension()[-1]
        Df, V = res4py.linalg.eig(
            self.A, self.A.apply, N, N, lambda x: x
        )
        Da, W = res4py.linalg.eig(
            self.A, self.A.apply_hermitian_transpose, N, N, lambda x: x
        )
        V, W, Df, Da = res4py.linalg.match_right_and_left_eigenvectors(
            V, W, Df, Da
        )
        return Df, V, W

    def evaluate_dynamics_numpy(self, t: float, q: np.ndarray) -> np.ndarray:
        q_seq = PETSc.Vec().createWithArray(
            np.asarray(q, dtype=np.complex128), len(q), comm=PETSc.COMM_SELF
        )
        q_dist = PETSc.Vec().create(comm=self.get_comm())
        q_dist.setSizes(self.get_state_dimension())
        q_dist.setFromOptions()
        q_dist = res4py.sequential_to_distributed_vector(q_seq, q_dist)
        y_dist = self.evaluate_dynamics(t, q_dist)
        y_seq = res4py.distributed_to_sequential_vector(y_dist)
        result = y_seq.getArray().copy()
        q_seq.destroy()
        q_dist.destroy()
        y_dist.destroy()
        y_seq.destroy()
        return result.real