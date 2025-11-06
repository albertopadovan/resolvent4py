import scipy as sp
import numpy as np
import random
import resolvent4py as res4py
import resolvent4py.linalg.resolvent_analysis_time_stepping as res_ts
from petsc4py import PETSc
from slepc4py import SLEPc
from .. import pytest_utils


def _evaluate_forcing(Fhat, omegas, t, tf):
    tau = t if tf < 0 else tf - t
    q = np.exp(1j * omegas * tau)
    if np.min(omegas) == 0:
        c = 2 * np.ones(len(omegas))
        c[0] = 1.0
        q *= c
        f = Fhat.dot(q).real
    else:
        f = Fhat.dot(q)
    return f


def _evaluate_dynamics(t, x, A, Fhat, omegas, tf):
    return A.dot(x) + _evaluate_forcing(Fhat, omegas, t, tf)


def test_time_stepping_forced(comm, square_matrix_size):
    r"""Test time stepping function against scipy.integrate.solve_ivp"""

    N, _ = square_matrix_size
    complex_lst = [False, True]
    adjoint_lst = [False, True]
    error_lst = []
    for complex in complex_lst:
        for adjoint in adjoint_lst:
            # Generate the operators
            Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
                comm, (N, N), complex
            )
            linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
            action = (
                linop.apply if not adjoint else linop.apply_hermitian_transpose
            )
            Apython = Apython.conj().T if adjoint else Apython

            # Generate frequency vector
            T = 2 * np.pi
            omega = 2 * np.pi / T
            omegas = (
                omega * np.arange(-2, 3, 1)
                if complex
                else omega * np.arange(0, 3, 1)
            )
            sz = (Apython.shape[0], len(omegas))
            Fpetsc, Fpython = pytest_utils.generate_random_bv(comm, sz, True)
            if not complex:
                Fpython[:, 0] = Fpython[:, 0].real
                f = Fpetsc.getColumn(0)
                f = res4py.vec_real(f, True)
                Fpetsc.restoreColumn(0, f)

            # Generate initial condition
            vpetsc, vpython = pytest_utils.generate_random_vector(
                comm, N, complex
            )

            nsteps = 10000
            t_eval = (T / (nsteps - 1)) * np.arange(0, nsteps, 1)
            tf = T if adjoint else -1
            sol_python = sp.integrate.solve_ivp(
                _evaluate_dynamics,
                [0, T],
                vpython,
                t_eval=t_eval,
                rtol=1e-13,
                atol=1e-13,
                args=(Apython, Fpython, omegas, tf),
            ).y
            sol_python = np.fliplr(sol_python) if adjoint else sol_python
            sol_petsc = res4py.solve_ivp(
                vpetsc,
                action,
                0,
                T,
                nsteps,
                method="RK3",
                m=1,
                adjoint=adjoint,
                periodic_forcing=(Fpetsc, omegas),
            )

            sol_petsc_mat = sol_petsc.getMat()
            sol_petsc_mat_seq = res4py.distributed_to_sequential_matrix(
                sol_petsc_mat
            )
            error = np.linalg.norm(
                sol_python - sol_petsc_mat_seq.getDenseArray()
            )
            error /= np.linalg.norm(sol_python)
            sol_petsc.restoreMat(sol_petsc_mat)
            sol_petsc_mat_seq.destroy()
            error_lst.append(error)

    assert np.max(np.asarray(error_lst)) < 1e-8
