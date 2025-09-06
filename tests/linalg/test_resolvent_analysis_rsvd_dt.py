import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from slepc4py import SLEPc
import matplotlib.pyplot as plt
import random
from .. import pytest_utils
import resolvent4py.linalg.resolvent_analysis_time_stepping as res_ts


def _compute_exact_svd(Apython, omegas, n_svals):
    Ulst, Slst, Vlst = [], [], []
    Id = np.eye(Apython.shape[0])
    for k in range(len(omegas)):
        R = sp.linalg.inv(1j * omegas[k] * Id - Apython)
        u, s, v = sp.linalg.svd(R)
        v = v.conj().T
        u = u[:, :n_svals]
        v = v[:, :n_svals]
        s = s[:n_svals]

        Ulst.append(u)
        Slst.append(s)
        Vlst.append(v)

    return Ulst, Slst, Vlst


def _create_random_fourier_coefficients(comm, sizes, omegas, tstore, real):
    Fhat = SLEPc.BV().create(comm=comm)
    Fhat.setSizes(sizes, len(omegas))
    Fhat.setType("mat")
    Fhat.setRandomNormal()
    if real:
        f = Fhat.getColumn(0)
        f = res4py.vec_real(f, True)
        Fhat.restoreColumn(0, f)

    F = SLEPc.BV().create(comm=comm)
    F.setSizes(sizes, len(tstore))
    F.setType("mat")

    f = F.createVec()
    for i in range(len(tstore)):
        f = res_ts._ifft(Fhat, f, omegas, tstore[i])
        F.insertVec(i, f)

    return Fhat, F


def test_fft(comm, square_matrix_size):
    """Test fft and ifft for the RSVD-dt algorithm."""

    N, _ = square_matrix_size
    Nl = res4py.compute_local_size(N)
    sizes = (Nl, N)

    omega = random.uniform(0.5, 1.5)
    n_omegas = np.random.randint(1, 10)
    dt = random.uniform(1e-4, 2 * np.pi / omega / 100)
    comm_mpi = comm.tompi4py()
    omega = comm_mpi.bcast(omega, root=0)
    n_omegas = comm_mpi.bcast(n_omegas, root=0)
    dt = comm_mpi.bcast(dt, root=0)

    for real in [True, False]:
        _, tstore, omegas, n_omegas = res_ts._create_time_and_frequency_arrays(
            dt, omega, n_omegas, 10, real
        )
        Fhat, F = _create_random_fourier_coefficients(
            comm, sizes, omegas, tstore, real
        )
        Fhat2 = Fhat.duplicate()
        Fhat2 = res_ts._fft(F, Fhat2, real)
        error = 0
        for i in range(n_omegas):
            f1 = Fhat.getColumn(i)
            f2 = Fhat2.getColumn(i)
            x = f1.copy()
            x.axpy(-1.0, f2)
            error += x.norm() / n_omegas
            Fhat.restoreColumn(i, f1)
            Fhat2.restoreColumn(i, f2)
            x.destroy()

        Fhat.destroy()
        F.destroy()
        Fhat2.destroy()

    assert error < 1e-13


def test_time_stepper(comm, square_matrix_size):
    r"""Test time stepper in RSVD-dt algorithm."""

    N, _ = square_matrix_size

    n_periods = 50
    omega = random.uniform(0.5, 1.5)
    n_omegas = np.random.randint(1, 10)
    dt = random.uniform(1e-4, 5e-4)
    comm_mpi = comm.tompi4py()
    omega = comm_mpi.bcast(omega, root=0)
    n_omegas = comm_mpi.bcast(n_omegas, root=0)
    dt = comm_mpi.bcast(dt, root=0)

    errors = []
    for adjoint in [True, False]:
        for real in [True, False]:
            Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
                comm, (N, N), not real
            )
            sizes = Apetsc.getSizes()[0]
            L = res4py.linear_operators.MatrixLinearOperator(Apetsc)

            tsim, tstore, omegas, n_omegas = (
                res_ts._create_time_and_frequency_arrays(
                    dt, omega, n_omegas, n_periods, real
                )
            )
            Fhat, F = _create_random_fourier_coefficients(
                comm, sizes, omegas, tstore, real
            )
            X = F.duplicate()
            Xhat = Fhat.duplicate()
            x = Xhat.createVec()

            # Compute post-transient response using rsvd_dt
            Laction = L.apply_hermitian_transpose if adjoint else L.apply
            Xhat = res_ts._action(
                L,
                Laction,
                tsim,
                tstore,
                omegas,
                x,
                Fhat,
                Xhat,
                X,
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

    assert np.max(error) < 5


def test_resolvent_analysis_time_stepping(comm, square_matrix_size):
    r"""Test RSVD-dt algorithm."""

    N, _ = square_matrix_size
    N = 5 if N > 5 else N

    n_periods = 50
    omega = random.uniform(0.5, 1.5)
    n_omegas = np.random.randint(1, 10)
    dt = random.uniform(1e-4, 2 * np.pi / omega / 1000)
    comm_mpi = comm.tompi4py()
    omega = comm_mpi.bcast(omega, root=0)
    n_omegas = comm_mpi.bcast(n_omegas, root=0)
    dt = comm_mpi.bcast(dt, root=0)

    errors = []
    for real in [True, False]:
        Apetsc, Apython = pytest_utils.generate_stable_random_matrix(
            comm, (N, N), not real
        )

        L = res4py.linear_operators.MatrixLinearOperator(Apetsc)

        n_rand = N
        n_loops = 2
        n_svals = 1
        tol = 1e-3
        _, Slst, _ = res_ts.resolvent_analysis_rsvd_dt(
            L, dt, omega, n_omegas, n_periods, n_rand, n_loops, n_svals, tol, 1
        )

        _, _, omegas, _ = res_ts._create_time_and_frequency_arrays(
            dt, omega, n_omegas, n_periods, not complex
        )
        _, Slst_, _ = _compute_exact_svd(Apython, omegas, n_svals)
        error = 0
        for i in range(len(Slst)):
            error += 100 * np.abs(Slst_[i][0] - Slst[i][0, 0]) / Slst_[i][0]
        errors.append(error)

    assert np.max(errors) < 5
