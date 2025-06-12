__all__ = ["construct_dft_mats", "randomized_time_stepping_svd"]

import copy
from math import ceil

import numpy as np
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.matrix import create_dense_matrix, create_AIJ_identity
from ..utils.ksp import create_gmres_asm_solver
from ..utils.comms import compute_local_size
from ..utils.timesteppers import timestep, setup, estimate_dt_max as estimate_dt_max_util

from ..linear_operators import MatrixLinearOperator, ProductLinearOperator

def construct_dft_mats(n_omega, n_omega_eff, n_timesteps, n, real_op):
    dft_mat = create_dense_matrix(PETSc.COMM_WORLD, (n_omega, n_omega_eff))
    i_dft_mat = create_dense_matrix(PETSc.COMM_SELF, (2 * n_timesteps, n_omega_eff))
    j_local = np.arange(2 * n_timesteps)
    alpha = -np.pi * 1j / n_timesteps
    for i in range(n_omega_eff):
        if real_op:
            exp_val = alpha * i
        else:
            exp_val = alpha * i if i <= (n_omega_eff/2 - 1) else alpha * ((i - n_omega_eff) + 2*n_timesteps)
        col = i_dft_mat.getDenseColumnVec(i)
        col_arr = col.getArray()
        col_arr[:] = np.conj(np.exp(exp_val * j_local)) / n_timesteps
        i_dft_mat.restoreDenseColumnVec(i, col)
    
    r0, r1 = dft_mat.getOwnershipRange()
    j_local = np.arange(r0, r1)
    alpha = -2 * np.pi * 1j / (2*n_omega_eff) if real_op else -2 * np.pi * 1j / n_omega_eff
    for i in range(n_omega_eff):
        col = dft_mat.getDenseColumnVec(i)
        col_arr = col.getArray()
        col_arr[:] = np.exp(alpha * i * j_local)
        dft_mat.restoreDenseColumnVec(i, col)
    
    for M in (i_dft_mat, dft_mat):
        M.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        M.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
    return dft_mat, i_dft_mat.transpose()

def randomized_time_stepping_svd(lin_op, omega, n_periods, n_timesteps, n_rand, n_loops, n_svals, ts_method: str = 'RK4', transient_removal: bool = True):
    r"""
        Compute the SVD of the linear operator :math:`iwI - A`
        specified by :code:`lin_op` using a time-stepping randomized SVD algorithm

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param n_rand: number of random vectors to use
        :type n_rand: int
        :param n_loops: number of randomized svd power iterations
        :type n_loops: int
        :param n_svals: number of singular triplets to return
        :type n_svals: int 

        :return: :math:`(U,\,\Sigma,\, V)` a 3-tuple with the leading
            :code:`n_svals` singular values and corresponding left and \
            right singular vectors
        :rtype: (SLEPc.BV with :code:`n_svals` columns,
            numpy.ndarray of size :code:`n_svals x n_svals`,
            SLEPc.BV with :code:`n_svals` columns)
    """
    
    real_op = lin_op._real
    n_omega = len(omega)
    n_omega_eff = n_omega / 2 if real_op else n_omega
    n_omega_samples = n_omega + (1 if transient_removal else 0)
    t_s = 2 * np.pi / np.min(np.abs(omega[omega != 0]))
    delta_t = t_s / n_omega
    dt = delta_t / ceil(delta_t / (t_s / n_timesteps))
    t_ratio = int(delta_t / dt)
    omega = omega[omega >= 0] if real_op else omega

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    try:
        dt_max = estimate_dt_max_util(lin_op, scheme=ts_method)
        if dt > 0.5 * dt_max:
            print(
                f"dt = {dt:.3e} is large compared with "
                f"the {ts_method} accuracy guard (≈ {dt_max:.3e})."
            )
        else:
            print(
                f"dt = {dt:.3e} is reasonable compared with "
                f"the {ts_method} accuracy guard (≈ {dt_max:.3e})."
            )
    except RuntimeError as err:
        print(f"Eigenvalue probe failed: {err}")

    # Assemble random BV
    N = lin_op.get_dimensions()[0][1]
    Nl = compute_local_size(N)
    sizes = ((Nl, N), (Nl, N))
    X = []
    for k in range(n_rand):
        comm.tompi4py().barrier()
        X_k = SLEPc.BV().create(comm=comm)
        X_k.setSizes(N, n_omega_eff)
        X_k.setType("vecs")
        X_k.setRandomNormal()
        X.append(X_k)

    # Assemble DFT matrices
    dft_mat, i_dft_mat = construct_dft_mats(n_omega, n_omega_eff, n_timesteps, N, real_op)

    q_temp = SLEPc.BV().create(comm=comm)
    q_temp.setSizes(N, n_rand)
    q_temp.setType("vecs")

    temp_rhs = SLEPc.BV().create(comm=comm)
    temp_rhs.setSizes(N, n_rand)
    temp_rhs.setType("vecs")

    f = SLEPc.BV().create(comm=comm)
    f.setSizes(N, n_rand)
    f.setType("vecs")

    f_next = SLEPc.BV().create(comm=comm)
    f_next.setSizes(N, n_rand)
    f_next.setType("vecs")

    def bv_has_nans(bv):
        bv_mat = bv.getMat()
        bv_arr = bv_mat.getDenseArray()
        if np.isnan(bv_arr).any() or np.isinf(bv_arr).any():
            bv.restoreMat(bv_mat)
            return True
        bv.restoreMat(bv_mat)
        return False

    # Forcing sampling function
    def sample_forcing(f_bv, forcing_mat_list, idx):
        idft_col = i_dft_mat.getColumnVector(idx)
        for i in range(n_rand):
            f_col = f_bv.getColumn(i)
            forcing_mat_list[i].multVec(1, 0, f_col, idft_col.getArray())
            f_bv.restoreColumn(i, f_col)

    # Timestepping functions
    def direct_action(forcing_mat):
        q_temp.scale(0.0)
        temp_rhs.scale(0.0)
        f.scale(0.0)
        f_next.scale(0.0)
        Y_hat = []
        setup(lin_op, q_temp, dt, method=ts_method)
        for period in range(n_periods):
            n_s = n_timesteps + (t_ratio if transient_removal and period == n_periods - 1 else 0)
            for i in range(n_s):
                # print(f"Period {period}, Timestep {i}")
                sample_forcing(f, forcing_mat, (2*i) % (2*n_timesteps))
                sample_forcing(f_next, forcing_mat, (2*i + 2) % (2*n_timesteps))

                forcing_arg = None
                if ts_method.upper() == 'RK4' or ts_method.upper() == 'CN':
                    forcing_arg = (f, f_next)
                elif ts_method.upper() == 'BE':
                    forcing_arg = f_next

                q_temp_next_step = timestep(lin_op, q_temp, f=forcing_arg, method=ts_method)
                q_temp_next_step.copy(q_temp)

                if period == n_periods - 1 and (i+1) % t_ratio == 0:
                    Y_new = SLEPc.BV().create(comm=comm)
                    Y_new.setSizes(N, n_rand)
                    Y_new.setType("vecs")
                    q_temp.copy(Y_new)
                    Y_hat.append(Y_new)
        Y_temp = []
        if transient_removal:
            Q_ident = create_dense_matrix(PETSc.COMM_SELF, (n_omega_eff, n_omega_eff))
            for i in range(n_omega_eff):
                Q_ident.setValue(i, i, 1.0)
            Q_ident.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            Q_ident.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

            Y_hat_1 = []
            Y_hat_2 = []
            for i in range(n_omega_samples - 1):
                Y_new = SLEPc.BV().create(comm=comm)
                Y_new.setSizes(N, n_rand)
                Y_new.setType("vecs")
                Y_hat[i].copy(Y_new)
                Y_hat_1.append(Y_new)
            for i in range(1, n_omega_samples):
                Y_new = SLEPc.BV().create(comm=comm)
                Y_new.setSizes(N, n_rand)
                Y_new.setType("vecs")
                Y_hat[i].copy(Y_new)
                Y_hat_2.append(Y_new)
            
            Y_hat = permute_mat_to_k_list_full(Y_hat)
            Y_hat_1 = permute_mat_to_k_list_full(Y_hat_1)
            Y_hat_2 = permute_mat_to_k_list_full(Y_hat_2)
            w = np.min(np.abs(omega[omega != 0]))

            M_tilde = SLEPc.BV().create(comm=comm)
            M_tilde.setSizes(n_omega_samples, n_omega_samples)
            M_tilde.setType("vecs")

            for i in range(n_rand):
                phi = Y_hat[i]
                phi.orthogonalize(None)
                psi = phi.copy()
                phi_temp = phi.copy()

                setup(lin_op, phi_temp, dt, method=ts_method)
                for j in range(t_ratio):
                    phi_next = timestep(lin_op, phi_temp, f=None, method=ts_method)
                    phi_next.copy(phi_temp)
                psi_mat = psi.getMat()
                phi_temp.matMultHermitianTranspose(psi_mat, M_tilde)

                y_1_temp = SLEPc.BV().create(comm=comm)
                y_1_temp.setSizes(N, n_omega_eff)
                y_1_temp.setType("vecs")
                y_2_temp = y_1_temp.duplicate()
                Y_1_mat = Y_hat_1[i].getMat()
                y_1_temp_mat = y_1_temp.getMat()
                Y_1_mat.matMult(dft_mat, y_1_temp_mat)
                y_1_temp.restoreMat(y_1_temp_mat)
                Y_hat_1[i].restoreMat(Y_1_mat)
                y_1_temp.scale(t_ratio)

                Y_2_mat = Y_hat_2[i].getMat()
                y_2_temp_mat = y_2_temp.getMat()
                Y_2_mat.matMult(dft_mat, y_2_temp_mat)
                y_2_temp.restoreMat(y_2_temp_mat)
                Y_hat_2[i].restoreMat(Y_2_mat)
                y_2_temp.scale(t_ratio)

                b = y_1_temp.copy()
                for s in range(n_omega_eff):
                    b.scaleColumn(s, np.exp(1j * w * s * delta_t))
                b.mult(-1, 1, y_2_temp, Q_ident)

                psi_b = SLEPc.BV().create(comm=comm)
                psi_b.setSizes(n_omega_samples, n_omega_eff)
                psi_b.setType("vecs")
                b.matMultHermitianTranspose(psi_mat, psi_b)
                psi.restoreMat(psi_mat)

                M_psi_b = SLEPc.BV().create(comm=comm)
                M_psi_b.setSizes(n_omega_samples, n_omega_eff)
                M_psi_b.setType("vecs")

                for s in range(n_omega_eff):
                    psi_b_col = psi_b.getColumn(s)
                    M_psi_b_col = M_psi_b.getColumn(s)
                    M_tilde_mat = M_tilde.getMat()
                    M_tilde_mat_c = M_tilde_mat.copy()
                    M_tilde_mat_c.shift(np.exp(1j * w * s * delta_t))
                    ksp = create_gmres_asm_solver(comm, M_tilde_mat_c)
                    ksp.solve(psi_b_col, M_psi_b_col)
                    ksp.destroy()
                    M_tilde.restoreMat(M_tilde_mat)
                    M_psi_b.restoreColumn(s, M_psi_b_col)
                    psi_b.restoreColumn(s, psi_b_col)
                
                phi_mat = phi.getMat()
                y_1_t = y_2_temp.duplicate()
                M_psi_b.matMult(phi_mat, y_1_t)
                phi.restoreMat(phi_mat)
                y_1_temp.mult(-1, 1, y_1_t, Q_ident)

                Y_temp_i = SLEPc.BV().create(comm=comm)
                Y_temp_i.setSizes(N, n_omega_eff)
                Y_temp_i.setType("vecs")
                y_1_temp.copy(Y_temp_i)

                Y_temp.append(Y_temp_i)
        else:
            Y_hat = permute_mat_to_k_list_full(Y_hat)
            for i in range(n_rand):
                Y_temp_i = SLEPc.BV().create(comm=comm)
                Y_temp_i.setSizes(N, n_omega_eff)
                Y_temp_i.setType("vecs")
                Y_temp_mat = Y_temp_i.getMat()

                Y_curr = Y_hat[i].getMat()
                Y_curr.matMult(dft_mat, Y_temp_mat)
                Y_hat[i].restoreMat(Y_curr)
                Y_temp_i.restoreMat(Y_temp_mat)
                Y_temp_i.scale(t_ratio)
                Y_temp.append(Y_temp_i)
        return Y_temp

    def adjoint_action(forcing_mat):
        q_temp.scale(0.0)
        temp_rhs.scale(0.0)
        f.scale(0.0)
        f_next.scale(0.0)
        S_hat = []
        lin_op.hermitian_transpose()
        setup(lin_op, q_temp, dt, method=ts_method)
        for period in range(n_periods):
            n_s = n_timesteps + (t_ratio if transient_removal and period == n_periods - 1 else 0)
            for i in range(n_s):
                # print(f"Period {period}, Timestep {i}")
                sample_forcing(f, forcing_mat, (2*(n_timesteps - i - 1)) % (2*n_timesteps))
                sample_forcing(f_next, forcing_mat, (2*(n_timesteps - i - 1) - 2) % (2*n_timesteps))

                forcing_arg = None
                if ts_method.upper() == 'RK4' or ts_method.upper() == 'CN':
                    forcing_arg = (f, f_next)
                elif ts_method.upper() == 'BE':
                    forcing_arg = f_next

                q_temp_next_step = timestep(lin_op, q_temp, f=forcing_arg, method=ts_method)
                q_temp_next_step.copy(q_temp) 
                    
                if period == n_periods - 1 and (i+1) % t_ratio == 0:
                    S_new = SLEPc.BV().create(comm=comm)
                    S_new.setSizes(N, n_rand)
                    S_new.setType("vecs")
                    q_temp.copy(S_new)
                    S_hat.append(S_new)
        lin_op.hermitian_transpose()
        S_hat.reverse()
        S_temp = []

        if transient_removal:
            Q_ident = create_dense_matrix(PETSc.COMM_SELF, (n_omega_eff, n_omega_eff))
            for i in range(n_omega_eff):
                Q_ident.setValue(i, i, 1.0)
            Q_ident.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            Q_ident.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

            S_hat_1 = []
            S_hat_2 = []
            for i in range(1, n_omega_samples):
                S_new = SLEPc.BV().create(comm=comm)
                S_new.setSizes(N, n_rand)
                S_new.setType("vecs")
                S_hat[i].copy(S_new)
                S_hat_1.append(S_new)
            for i in range(n_omega_samples - 1):
                S_new = SLEPc.BV().create(comm=comm)
                S_new.setSizes(N, n_rand)
                S_new.setType("vecs")
                S_hat[i].copy(S_new)
                S_hat_2.append(S_new)
            
            S_hat = permute_mat_to_k_list_full(S_hat)
            S_hat_1 = permute_mat_to_k_list_full(S_hat_1)
            S_hat_2 = permute_mat_to_k_list_full(S_hat_2)
            w = np.min(np.abs(omega[omega != 0]))

            M_tilde = SLEPc.BV().create(comm=comm)
            M_tilde.setSizes(n_omega_samples, n_omega_samples)
            M_tilde.setType("vecs")

            for i in range(n_rand):
                phi = S_hat[i]
                phi.orthogonalize(None)
                psi = phi.copy()
                
                setup(lin_op, phi, dt, method=ts_method)
                for j in range(t_ratio):
                    phi_next = timestep(lin_op, phi, f=None, method=ts_method)
                    phi_next.copy(phi)
                psi_mat = psi.getMat()
                phi.matMultHermitianTranspose(psi_mat, M_tilde)

                s_1_temp = SLEPc.BV().create(comm=comm)
                s_1_temp.setSizes(N, n_omega_eff)
                s_1_temp.setType("vecs")
                s_2_temp = s_1_temp.duplicate()
                S_1_mat = S_hat_1[i].getMat()
                s_1_temp_mat = s_1_temp.getMat()
                S_1_mat.matMult(dft_mat, s_1_temp_mat)
                s_1_temp.restoreMat(s_1_temp_mat)
                S_hat_1[i].restoreMat(S_1_mat)
                s_1_temp.scale(t_ratio)
                
                S_2_mat = S_hat_2[i].getMat()
                s_2_temp_mat = s_2_temp.getMat()
                S_2_mat.matMult(dft_mat, s_2_temp_mat)
                s_2_temp.restoreMat(s_2_temp_mat)
                S_hat_2[i].restoreMat(S_2_mat)
                s_2_temp.scale(t_ratio)
                
                b = s_1_temp.copy()
                for s in range(n_omega_eff):
                    b.scaleColumn(s, np.exp(-1j * w * s * delta_t))
                b.mult(-1, 1, s_2_temp, Q_ident)

                psi_b = SLEPc.BV().create(comm=comm)
                psi_b.setSizes(n_omega_samples, n_omega_eff)
                psi_b.setType("vecs")
                b.matMultHermitianTranspose(psi_mat, psi_b)
                psi.restoreMat(psi_mat)

                M_psi_b = SLEPc.BV().create(comm=comm)
                M_psi_b.setSizes(n_omega_samples, n_omega_eff)
                M_psi_b.setType("vecs")

                for s in range(n_omega_eff):
                    psi_b_col = psi_b.getColumn(s)
                    M_psi_b_col = M_psi_b.getColumn(s)
                    M_tilde_mat = M_tilde.getMat()
                    M_tilde_mat_c = M_tilde_mat.copy()
                    M_tilde_mat_c.shift(np.exp(-1j * w * s * delta_t))
                    ksp = create_gmres_asm_solver(comm, M_tilde_mat_c)
                    ksp.solve(psi_b_col, M_psi_b_col)
                    ksp.destroy()
                    M_tilde.restoreMat(M_tilde_mat)
                    M_psi_b.restoreColumn(s, M_psi_b_col)
                    psi_b.restoreColumn(s, psi_b_col)
                
                phi_mat = phi.getMat()
                M_psi_b.matMult(phi_mat, s_2_temp)
                phi.restoreMat(phi_mat)
                s_1_temp.mult(-1, 1, s_2_temp, Q_ident)

                S_temp_i = SLEPc.BV().create(comm=comm)
                S_temp_i.setSizes(N, n_omega_eff)
                S_temp_i.setType("vecs")
                s_1_temp.copy(S_temp_i)

                S_temp.append(S_temp_i)
        else:
            S_hat = permute_mat_to_k_list_full(S_hat)
            for i in range(n_rand):
                S_temp_i = SLEPc.BV().create(comm=comm)
                S_temp_i.setSizes(N, n_omega_eff)
                S_temp_i.setType("vecs")
                S_temp_mat = S_temp_i.getMat()

                S_curr = S_hat[i].getMat()
                S_curr.matMult(dft_mat, S_temp_mat)
                S_hat[i].restoreMat(S_curr)
                S_temp_i.restoreMat(S_temp_mat)
                S_temp_i.scale(t_ratio)
                S_temp.append(S_temp_i)
        return S_temp

    def permute_mat_to_Nw_list(mat):
        mat_hat_mod = []
        for ww in range(n_omega_eff):
            mat_tt = SLEPc.BV().create(comm=comm)
            mat_tt.setSizes(N, n_rand)
            mat_tt.setType("vecs")
            for k in range(n_rand):
                curr = mat[k]
                col = curr.getColumn(ww)
                col_tt = mat_tt.getColumn(k)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(k, col_tt)
                curr.restoreColumn(ww, col)
            mat_hat_mod.append(mat_tt)
        for k in range(n_rand):
            mat[k].destroy()
        return mat_hat_mod

    def permute_mat_to_k_list(mat):
        mat_hat_mod = []
        for k in range(n_rand):
            mat_tt = SLEPc.BV().create(comm=comm)
            mat_tt.setSizes(N, n_omega_eff)
            mat_tt.setType("vecs")
            for ww in range(n_omega_eff):
                curr = mat[ww]
                col = curr.getColumn(k)
                col_tt = mat_tt.getColumn(ww)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(ww, col_tt)
                curr.restoreColumn(k, col)
            mat_hat_mod.append(mat_tt)
        for ww in range(n_omega_eff):
            mat[ww].destroy()
        return mat_hat_mod

    def permute_mat_to_k_list_full(mat):
        mat_hat_mod = []
        for k in range(n_rand):
            mat_tt = SLEPc.BV().create(comm=comm)
            mat_tt.setSizes(N, len(mat))
            mat_tt.setType("vecs")
            for ww in range(len(mat)):
                curr = mat[ww]
                col = curr.getColumn(k)
                col_tt = mat_tt.getColumn(ww)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(ww, col_tt)
                curr.restoreColumn(k, col)
            mat_hat_mod.append(mat_tt)
        for ww in range(len(mat)):
            mat[ww].destroy()
        return mat_hat_mod

    def orthogonalize_timestepped_mat(Z):
        for i in range(n_omega_eff):
            Z_i = Z[i]
            Z_i.orthogonalize(None)
            Z[i] = Z_i
        return Z

    Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(X))))
    for q in range(n_loops):
        S_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(adjoint_action(Y_hat))))
        Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(S_hat))))
    S_hat = permute_mat_to_Nw_list(adjoint_action(Y_hat))
    Y_hat = permute_mat_to_Nw_list(Y_hat)

    if rank == 0:
        S = PETSc.Mat().createDense(
            [n_omega_eff, n_svals], comm=PETSc.COMM_SELF
        )
        S.setUp()
    else:
        S = 0
    U = []
    V = []
    
    mpi_comm = comm.tompi4py()
    for i in range(n_omega_eff):
        Y_local = Y_hat[i].getMat().getDenseArray()
        S_local = S_hat[i].getMat().getDenseArray()

        Y_all = mpi_comm.gather(Y_local, root=0)
        S_all = mpi_comm.gather(S_local, root=0)

        if rank == 0:
            Y_full = np.vstack(Y_all)
            S_local = np.vstack(S_all).conj().T
            if not np.isfinite(S_local).all():
                raise ValueError("S_local contains NaN/Inf")
            if np.max(np.abs(S_local)) == 0:
                raise ValueError("S_local is identically zero - cannot compute SVD")

            u_tilde, s, vh = sp.sparse.linalg.svds(S_local, k=n_svals)
            s = np.flip(s)
            S[i, :] = s
            S.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            S.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
            
            factor = u_tilde @ np.diag(s)
            U_arr = Y_full @ factor
            U_i = SLEPc.BV().create(comm=PETSc.COMM_SELF)
            U_i.setSizes(N, n_svals)
            U_i.setType("vecs")
            for j in range(n_svals):
                col = U_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = U_arr[:, j]
                U_i.restoreColumn(j, col)
            V_i = SLEPc.BV().create(comm=PETSc.COMM_SELF)
            V_i.setSizes(N, n_svals)
            V_i.setType("vecs")
            v = vh.conj().T
            for j in range(n_svals):
                col = V_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = v[:, j]
                V_i.restoreColumn(j, col)
            U.append(U_i)
            V.append(V_i)

    temp_rhs.destroy()
    f.destroy()
    f_next.destroy()
    q_temp.destroy()

    return U, S, V
