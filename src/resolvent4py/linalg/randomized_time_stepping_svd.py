from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from math import ceil
import copy

from ..linalg import enforce_complex_conjugacy
from ..utils.mat_helpers import create_dense_matrix

def construct_dft_mats(n_omega, n_timesteps, n):
    dft_mat = create_dense_matrix(MPI.COMM_SELF, (n_omega, n_omega // 2))
    i_dft_mat = create_dense_matrix(MPI.COMM_SELF, (2 * n_timesteps, n_omega // 2))

    j = np.linspace(0, 2 * n_timesteps - 1, 2 * n_timesteps)
    
    alpha = -np.pi * 1j / n_timesteps 
    for i in range(n_omega // 2):
        col = i_dft_mat.getDenseColumnVec(i)
        col.array[:] = (alpha * i) * np.conj(np.exp(alpha * i * j)) / n_timesteps
        i_dft_mat.restoreDenseColumnVec(i, col)

    j = np.linspace(0, n_omega - 1, n_omega)
    
    alpha = -2 * np.pi * 1j / n_omega
    for i in range(n_omega // 2):
        col = dft_mat.getDenseColumnVec(i)
        col.array[:] = np.exp(alpha * i * j)
        dft_mat.restoreDenseColumnVec(i, col)

    return dft_mat, i_dft_mat

def randomized_time_stepping_svd(lin_op, lin_op_mass, lin_op_action, omega, n_periods, n_timesteps, n_rand, n_loops, n_svals):
    r"""
        Compute the SVD of the linear operator :math:`iwG - A`
        specified by :code:`lin_op` using a time-stepping randomized SVD algorithm

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply_mat` or
            :code:`lin_op.solve_mat`
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

    n_omega = len(omega)
    t_s = 2 * np.pi / np.min(np.abs(omega[omega != 0]))

    delta_t = t_s / n_omega
    dt = delta_t / ceil(delta_t / (t_s / n_timesteps))
    t_ratio = int(delta_t / dt)

    omega = omega[omega >= 0]

    if lin_op_action != lin_op.apply_mat and lin_op_action != lin_op.solve_mat:
        raise ValueError (
            f"lin_op_action must be lin_op.apply_mat or lin_op.solve_mat."
        )
    if lin_op_action == lin_op.apply_mat:
        lin_op_action_adj = lin_op.apply_hermitian_transpose_mat
    if lin_op_action == lin_op.solve_mat:
        lin_op_action_adj = lin_op.solve_hermitian_transpose_mat
    
    # Assemble random BV
    N = lin_op.get_dimensions()[0][1]
    print(f"N = {N}, k = {n_rand}")
    X = []
    for k in range(n_rand):
        X_k = SLEPc.BV().create(comm=lin_op._comm)
        X_k.setSizes(N, n_omega // 2)
        X_k.setType('mat')
        X_k.setRandomNormal()
        for j in range(n_omega // 2):
            xj = X_k.getColumn(j)
            if lin_op._real:
                row_offset = xj.getOwnershipRange()[0]
                rows = np.arange(N[0], dtype=np.int64) + row_offset
                array = xj.getArray()
                xj.setValues(rows, array.real)
                xj.assemble()
            if lin_op._block_cc:
                enforce_complex_conjugacy(lin_op._comm, xj, lin_op._nblocks)
            X_k.restoreColumn(j, xj)
        X_k.orthogonalize(None)
        X.append(X_k)

    # Assemble DFT matrices
    dft_mat, i_dft_mat = construct_dft_mats(n_omega, n_timesteps, N)

    # Forcing sampling function
    def sample_forcing(forcing_mat, idx):
        f = SLEPc.BV().create(comm=X[0].comm)
        f.setSizes(N, n_rand)
        f.setType('mat')

        for i in range(n_rand): 
            temp = forcing_mat[i].getMat()
            f_i = temp.getDenseArray() @ i_dft_mat[idx,:].transpose()
            
            f_col = f.getColumn(i)
            f_col.setArray(f_i)
            f.restoreColumn(i, f_col)
            
            forcing_mat[i].restoreMat(temp)
        return f

    # Timestepping functions
    def direct_action(forcing_mat):
        q_temp = SLEPc.BV().create(comm=lin_op._comm)
        q_temp.setSizes(N, n_rand)
        q_temp.setType('mat')
        q_temp.scale(0.0)

        Y_hat = []
        for i in range(n_omega):
            Y_hat_i = SLEPc.BV().create(comm=lin_op._comm)
            Y_hat_i.setSizes(N, n_rand)
            Y_hat_i.setType('mat')
            Y_hat_i.setFromOptions()
            Y_hat.append(Y_hat_i)

        lhs = copy.deepcopy(lin_op_mass)
        lhs.axpy(-dt/2, lin_op)
        
        rhs_1 = copy.deepcopy(lin_op_mass)
        rhs_1.axpy(dt/2, lin_op)
        
        ksp = PETSc.KSP().create(comm=lin_op._comm)
        ksp.setOperators(lhs.A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        ksp.setUp()

        temp_rhs = SLEPc.BV().create(comm=lin_op._comm)
        temp_rhs.setSizes(N, n_rand)
        temp_rhs.setType('mat')
        
        f_sum = SLEPc.BV().create(comm=lin_op._comm)
        f_sum.setSizes(N, n_rand)
        f_sum.setType('mat')

        idx = 0
        for period in range(n_periods):
            for i in range(n_timesteps):
                f = sample_forcing(forcing_mat, (2*(i-1)) % (2*n_timesteps))
                f_next = sample_forcing(forcing_mat, (2*(i-1)+2) % (2*n_timesteps))
                
                temp_rhs.scale(0.0)
                rhs_1.apply_mat(q_temp, temp_rhs)
                
                _, nc = f.getSizes()
                f_sum.scale(0.0)

                for k in range(nc):
                    v_f = f.getColumn(k)
                    v_fn = f_next.getColumn(k)
                    v_sum = f_sum.getColumn(k)

                    v_sum.axpy(1.0, v_f)
                    v_sum.axpy(1.0, v_fn)
                    v_sum.scale(dt / 2)

                    v_rhs = temp_rhs.getColumn(k)
                    v_rhs.axpy(1.0, v_sum)

                    f.restoreColumn(k, v_f)
                    f_next.restoreColumn(k, v_fn)
                    f_sum.restoreColumn(k, v_sum)
                    temp_rhs.restoreColumn(k, v_rhs)

                for j in range(n_rand):
                    rhs_col = temp_rhs.getColumn(j)
                    sol_col = q_temp.getColumn(j)
                    ksp.solve(rhs_col, sol_col)
                    temp_rhs.restoreColumn(j, rhs_col)
                    q_temp.restoreColumn(j, sol_col)
                
                if period == n_periods - 1 and i % t_ratio == 0:
                    q_temp.copy(Y_hat[idx])
                    idx += 1

                f.destroy()
                f_next.destroy()

        temp_rhs.destroy()
        f_sum.destroy()
        
        Y_temp = []
        for i in range(n_rand):
            Y_temp_i = SLEPc.BV().create(comm=lin_op._comm)
            Y_temp_i.setSizes(N, n_omega//2)
            Y_temp_i.setType('mat')

            local_cols = []
            for y in Y_hat:
                col = y.getColumn(i)
                local_cols.append(col.getArray().copy())
                y.restoreColumn(i, col)

            dft_coeffs_local = np.column_stack(local_cols) @ dft_mat.getDenseArray()

            for j in range(n_omega//2):
                col = Y_temp_i.getColumn(j)
                col.array[:] = dft_coeffs_local[:, j] * t_ratio
                Y_temp_i.restoreColumn(j, col)
            
            Y_temp.append(Y_temp_i)
            
            del local_cols
            del dft_coeffs_local
            del Y_temp_i

        lhs.destroy()
        rhs_1.destroy()
        ksp.destroy()
        q_temp.destroy()

        return Y_temp

    def adjoint_action(forcing_mat):
        z_temp = SLEPc.BV().create(comm=lin_op._comm)
        z_temp.setSizes(N, n_rand)
        z_temp.setType('mat')
        z_temp.scale(0.0)

        S_hat = []
        for i in range(n_omega):
            S_hat_i = SLEPc.BV().create(comm=lin_op._comm)
            S_hat_i.setSizes(N, n_rand)
            S_hat_i.setType('mat')
            S_hat.append(S_hat_i)
        
        lin_op_mass.hermitian_transpose()
        lin_op.hermitian_transpose()
        
        lhs = copy.deepcopy(lin_op_mass)
        lhs.axpy(-dt/2, lin_op)
        
        rhs_1 = copy.deepcopy(lin_op_mass)
        rhs_1.axpy(dt/2, lin_op)
        
        ksp = PETSc.KSP().create(comm=lin_op._comm)
        ksp.setOperators(lhs.A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        ksp.setUp()

        temp_rhs = SLEPc.BV().create(comm=lin_op._comm)
        temp_rhs.setSizes(N, n_rand)
        temp_rhs.setType('mat')
        
        s_sum = SLEPc.BV().create(comm=lin_op._comm)
        s_sum.setSizes(N, n_rand)
        s_sum.setType('mat')

        idx = n_omega - 1
        for period in range(n_periods):
            for i in range(n_timesteps):
                s = sample_forcing(forcing_mat, (2*(n_timesteps - i)) % (2*n_timesteps))
                s_next = sample_forcing(forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                
                temp_rhs.scale(0.0)
                rhs_1.apply_mat(z_temp, temp_rhs)

                _, nc = s.getSizes()
                s_sum.scale(0.0)
                
                for k in range(nc):
                    v_s = s.getColumn(k)
                    v_sn = s_next.getColumn(k)
                    v_sum = s_sum.getColumn(k)

                    v_sum.axpy(1.0, v_s)
                    v_sum.axpy(1.0, v_sn)
                    v_sum.scale(dt / 2)

                    v_rhs = temp_rhs.getColumn(k)
                    v_rhs.axpy(1.0, v_sum)

                    s.restoreColumn(k, v_s)
                    s_next.restoreColumn(k, v_sn)
                    s_sum.restoreColumn(k, v_sum)
                    temp_rhs.restoreColumn(k, v_rhs)

                for j in range(n_rand):
                    rhs_col = temp_rhs.getColumn(j)
                    sol_col = z_temp.getColumn(j)
                    ksp.solve(rhs_col, sol_col)
                    temp_rhs.restoreColumn(j, rhs_col)
                    z_temp.restoreColumn(j, sol_col)
                
                if period == n_periods - 1 and i % t_ratio == 0:
                    z_temp.copy(S_hat[idx])
                    idx -= 1

                s.destroy()
                s_next.destroy()

        temp_rhs.destroy()
        s_sum.destroy()
        
        S_temp = []
        for i in range(n_rand):
            S_temp_i = SLEPc.BV().create(comm=lin_op._comm)
            S_temp_i.setSizes(N, n_omega//2)
            S_temp_i.setType('mat')

            local_cols = []
            for s in S_hat:
                col = s.getColumn(i)
                local_cols.append(col.getArray().copy())
                s.restoreColumn(i, col)
            
            dft_coeffs_local = np.column_stack(local_cols) @ dft_mat.getDenseArray()

            for j in range(n_omega//2):
                col = S_temp_i.getColumn(j)
                col.array[:] = dft_coeffs_local[:, j] * t_ratio
                S_temp_i.restoreColumn(j, col)
            
            S_temp.append(S_temp_i)
            
            del local_cols
            del dft_coeffs_local
            del S_temp_i
        
        lhs.destroy()
        rhs_1.destroy()
        ksp.destroy()
        z_temp.destroy()

        return S_temp

    def orthogonalize_timestepped_mat(Z):
        for i in range(n_omega // 2):
            Z_i = Z[i]
            Z_i.orthogonalize(None)
            Z[i] = Z_i
        return Z

    Y_hat = orthogonalize_timestepped_mat(direct_action(X))

    for q in range(n_loops):
        S_hat = orthogonalize_timestepped_mat(adjoint_action(Y_hat))
        Y_hat = orthogonalize_timestepped_mat(direct_action(S_hat))

    S_hat = orthogonalize_timestepped_mat(adjoint_action(Y_hat))

    S = PETSc.Mat().createDense(n_omega // 2, n_svals, comm=MPI.COMM_SELF)
    U = []
    V = []
    for i in range(n_omega // 2):
        S_svd = S_hat[i].getMat().getDenseArray().T
        u_tilde, s, vh = sp.linalg.svd(S_svd, full_matrices=False)
        v = vh.conj().T

        S[i, :] = s[:n_svals]
        
        Y_hat_array = Y_hat[i].getMat().getDenseArray()
        factor = u_tilde[:, :n_svals] @ np.diag(s[:n_svals])
        U_array = Y_hat_array @ factor

        U_i = SLEPc.BV().create(comm=lin_op._comm)
        U_i.setSizes(N, n_svals)
        U_i.setType('mat')

        for j in range(n_svals):
            col = U_i.getColumn(j)
            col.array[:] = U_array[:, j]
            U_i.restoreColumn(j, col)

        V_i = SLEPc.BV().create(comm=lin_op._comm)
        V_i.setSizes(N, n_svals)
        V_i.setType('mat')
        
        for j in range(n_svals):
            col = V_i.getColumn(j)
            col.array[:] = v[:N, j]
            V_i.restoreColumn(j, col)
        
        U.append(U_i)
        V.append(V_i)
        

    return U, S, V