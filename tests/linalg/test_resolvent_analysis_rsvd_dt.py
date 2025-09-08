import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from slepc4py import SLEPc
import matplotlib.pyplot as plt
import random
from .. import pytest_utils
import resolvent4py.linalg.resolvent_analysis_time_stepping as res_ts


def _compute_exact_svd(Apython, Bpython, Cpython, omegas, n_svals):
    Ulst, Slst, Vlst = [], [], []
    Id = np.eye(Apython.shape[0])
    for k in range(len(omegas)):
        R = sp.linalg.inv(1j * omegas[k] * Id - Apython)
        R = Cpython.conj().T @ R @ Bpython
        u, s, v = sp.linalg.svd(R)
        v = v.conj().T
        u = u[:, :n_svals]
        v = v[:, :n_svals]
        s = s[:n_svals]

        Ulst.append(u)
        Slst.append(s)
        Vlst.append(v)

    return Ulst, Slst, Vlst

def test_post_transient_response(comm, square_matrix_size):
    r"""Test post-transient response."""

    N, _ = square_matrix_size

    n_periods = 50
    omega = np.pi
    dt = 1e-4
    n_omegas = 3

    Nl = res4py.compute_local_size(N)
    Id = res4py.create_AIJ_identity(comm, ((Nl, N), (Nl, N)))
    Idop = res4py.linear_operators.MatrixLinearOperator(Id)
    B = Idop
    C = Idop

    errors = []
    for adjoint in [True, False]:
        for real in [True, False]:
            Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
                comm, (N, N), not real
            )
            sizes = Apetsc.getSizes()[0]
            L = res4py.linear_operators.MatrixLinearOperator(Apetsc)
            Laction = L.apply_hermitian_transpose if adjoint else L.apply

            # Generate frequency vector
            tsim, nsave, omegas = res4py.create_time_and_frequency_arrays(
                dt, omega, n_omegas, real
            )
            n_omegas = len(omegas)
            sz = (Apython.shape[0], n_omegas)
            Fhat, _ = pytest_utils.generate_random_bv(comm, sz, True)
            if real:
                f = Fhat.getColumn(0)
                f = res4py.vec_real(f, True)
                Fhat.restoreColumn(0, f)

            Xhat = Fhat.duplicate()
            x = Xhat.createVec()
            x.zeroEntries()

            # Compute post-transient response using time stepping
            X = SLEPc.BV().create(comm=comm)
            X.setSizes(sizes, len(tsim[::nsave]))
            X.setType("mat")

            Xhat = res4py.compute_post_transient_solution(
                L,
                B,
                C,
                Laction,
                tsim,
                nsave,
                n_periods,
                omegas,
                x,
                Fhat,
                Xhat,
                X,
                1e-8,
                "RK3",
                1,
            )

            Xhat_mat = Xhat.getMat()
            Xhat_mat_seq = res4py.distributed_to_sequential_matrix(Xhat_mat)
            Xhat_a = Xhat_mat_seq.getDenseArray().copy()
            Xhat.restoreMat(Xhat_mat)
            Xhat_mat_seq.destroy()

            Fhat_mat = Fhat.getMat()
            Fhat_mat_seq = res4py.distributed_to_sequential_matrix(Fhat_mat)
            Fhat_a = Fhat_mat_seq.getDenseArray().copy()
            Fhat.restoreMat(Fhat_mat)
            Fhat_mat_seq.destroy()

            # Compute post-transient response using the resolvent operator
            Id = np.eye(sizes[-1])
            Xhat_a_python = np.zeros_like(Fhat_a)
            for i in range(len(omegas)):
                R = sp.linalg.inv(1j * omegas[i] * Id - Apython)
                R = R.conj().T if adjoint else R
                Xhat_a_python[:, i] = R @ Fhat_a[:, i]

            error = np.linalg.norm(Xhat_a_python - Xhat_a)
            error *= 100 / np.linalg.norm(Xhat_a_python)
            errors.append(error)

            L.destroy()
    assert np.max(error) < 1e-5


def test_resolvent_analysis_time_stepping(comm, square_matrix_size):
    r"""Test RSVD-dt algorithm."""

    N, _ = square_matrix_size
    N = 5 if N > 5 else N

    n_periods = 200
    omega = random.uniform(0.5, 1.5)
    n_omegas = np.random.randint(1, 10)
    dt = random.uniform(1e-4, 2 * np.pi / omega / 1000)
    comm_mpi = comm.tompi4py()
    omega = comm_mpi.bcast(omega, root=0)
    n_omegas = comm_mpi.bcast(n_omegas, root=0)
    dt = comm_mpi.bcast(dt, root=0)

    m = 10
    p = 8
    Bpetsc, Bpython = pytest_utils.generate_random_matrix(comm, (N, m), False)
    Cpetsc, Cpython = pytest_utils.generate_random_matrix(comm, (N, p), False)
    B = res4py.linear_operators.MatrixLinearOperator(Bpetsc)
    C = res4py.linear_operators.MatrixLinearOperator(Cpetsc)

    errors = []
    for real in [True, False]:
        Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
            comm, (N, N), not real
        )

        L = res4py.linear_operators.MatrixLinearOperator(Apetsc)

        n_rand = N
        n_loops = 2
        n_svals = 1
        tol = 1e-5
        _, Slst, _ = res_ts.resolvent_analysis_rsvd_dt(
            L,
            dt,
            omega,
            n_omegas,
            n_periods,
            n_rand,
            n_loops,
            n_svals,
            B,
            C,
            tol,
            'RK3',
            0,
        )

        _, _, omegas = res4py.create_time_and_frequency_arrays(
            dt, omega, n_omegas, real
        )
        _, Slst_, _ = _compute_exact_svd(Apython, Bpython, Cpython, omegas, n_svals)
        error = 0
        for i in range(len(Slst)):
            error += np.abs(Slst_[i][0] - Slst[i][0, 0]) / Slst_[i][0]
        errors.append(error)
    print(errors)
    assert np.max(errors) < 1e-2
