import pytest
import scipy as sp
import numpy as np
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc


def test_randomized_time_stepping_svd(comm, square_random_negative_semidefinite_matrix):
    """Test randomized timestepping SVD with different matrix sizes based on test level."""
    s = 10000.0
    nfreq = 10
    omega = np.arange(-nfreq*s,nfreq*s,s)
    Apetsc, Apython = square_random_negative_semidefinite_matrix
    krylov_dim = np.min([30, np.min(Apython.shape)])
    linop = res4py.linear_operators.MatrixLinearOperator(comm, Apetsc)
    n_cycles = 2
    n_timesteps = nfreq*2*10
    n_periods = 2
    r = np.min([3, krylov_dim])
    _, S, _ = res4py.linalg.randomized_time_stepping_svd(
        linop, omega, n_periods, n_timesteps, krylov_dim, n_cycles, r
    )

    if PETSc.COMM_WORLD.getRank() == 0:
        S_arr = S.getDenseArray()
        for i in range(len(omega)):
            w = omega[i]
            temp = np.linalg.inv(1j * w * np.eye(np.min(Apython.shape)) - Apython)
            _, s_act, _ = sp.linalg.svd(temp, full_matrices=False)
        
            print(f"w = {w}")
            print(f"s_act = {s_act[:r]}")
            print(f"S_arr = {S_arr[i,:]}")

            if w == s:
                error = 100 * np.abs(np.max((S_arr[i,:] - s_act[:r]) / s_act[:r]))
                assert error < 1

