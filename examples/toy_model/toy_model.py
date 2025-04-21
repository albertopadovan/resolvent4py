import numpy as np
import scipy


# ----------------------------------------------------
# ----- Define the dynamics of the system ------------
# ----------------------------------------------------


def compute_third_order_tensor(params, n):
    _, alpha, beta = params[0], params[1], params[2]
    Q1 = np.random.randn(n, n**2)
    Q1 = Q1 / np.linalg.norm(Q1)
    Q2 = np.random.randn(n, n**2)
    Q2 = Q2 / np.linalg.norm(Q2)
    L = np.zeros_like(Q1)
    R = np.zeros((n**2, n**2))
    for k in range(n**2):
        x1, y1, _ = Q1[0, k], Q1[1, k], Q1[2, k]
        x2, y2, z2 = Q2[0, k], Q2[1, k], Q2[2, k]
        R[:, k] = np.reshape(np.einsum("i,j", Q1[:, k], Q2[:, k]), -1)
        L[:, k] = np.asarray(
            [
                -alpha * x1 * z2 - beta * x1 * y2,
                -alpha * y1 * z2 + beta * x1 * x2,
                alpha * (x1 * x2 + y1 * y2),
            ]
        )

    H = (L @ R.T @ scipy.linalg.inv(R @ R.T)).reshape((n, n, n))
    H[np.abs(H) < 1e-13] = 0.0
    validate_third_order_tensor(params, H, n)

    return H


def validate_third_order_tensor(params, H, n):
    _, alpha, beta = params[0], params[1], params[2]
    for k in range(10):
        Q1 = np.random.randn(n)
        Q2 = np.random.randn(n)

        x1, y1, _ = Q1[0], Q1[1], Q1[2]
        x2, y2, z2 = Q2[0], Q2[1], Q2[2]
        v1 = np.asarray(
            [
                -alpha * x1 * z2 - beta * x1 * y2,
                -alpha * y1 * z2 + beta * x1 * x2,
                alpha * (x1 * x2 + y1 * y2),
            ]
        )
        v2 = np.einsum("ijk,j,k", H, Q1, Q2)

        error = np.linalg.norm(v1 - v2)
        if error > 1e-10:
            raise ValueError(
                "Third-order tensor was not computed correctly. Error = %1.12e"
                % error
            )


def evaluate_rhs(t, q, A, H, B, wf):
    return A @ q + np.einsum("ijk,j,k", H, q, q) + B @ [np.sin(wf * t)]


# ----------------------------------------------------
# ----- Define forward and inverse fft ---------------
# ----------------------------------------------------


def fft(t, Q, n):
    T = t[-1] + t[1] - t[0]
    freqs = (2 * np.pi / T) * np.arange(-n, n + 1, 1)
    QHat = (1 / len(t)) * scipy.fft.rfft(Q, axis=-1)[:, : (n + 1)]
    QHat = np.concatenate((np.fliplr(QHat).conj(), QHat[:, 1:]), axis=-1)

    return freqs, QHat


def ifft(freqs, QHat, t, dt):
    Q = np.zeros((QHat.shape[0], len(t)), dtype=np.complex128)
    for k in range(len(freqs)):
        Q += QHat[:, k].reshape(-1, 1) * np.exp(1j * freqs[k] * (t + dt))

    return Q.real


# ----------------------------------------------------
# ----- Compute frequency domain matrices ------------
# ----------------------------------------------------


def compute_lifted_frequency_domain_matrices(tensors, freqs, QHat):
    A, H, B, C = tensors[0], tensors[1], tensors[2], tensors[3]

    n = A.shape[0]  # Size of the system
    m = B.shape[-1]  # Size of the input
    p = C.shape[0]  # Size of the output
    nf = len(freqs)  # Total number of frequencies (including negative freqs.)

    T = np.zeros((n * nf, n * nf), dtype=np.complex128)
    B_ = np.zeros((n * nf, m * nf), dtype=np.complex128)
    C_ = np.zeros((p * nf, n * nf), dtype=np.complex128)

    for i in range(-(nf // 2), nf // 2 + 1):
        i0 = (i + nf // 2) * n
        i1 = i0 + n
        T[i0:i1, i0:i1] = (
            -1j * freqs[i + nf // 2] * np.diag(np.ones(n))
            + A
            + np.einsum("ijk,j", H, QHat[:, nf // 2])
            + np.einsum("ijk,k", H, QHat[:, nf // 2])
        )

        j0 = (i + nf // 2) * m
        j1 = j0 + m
        B_[i0:i1, j0:j1] = B

        j0 = (i + nf // 2) * p
        j1 = j0 + p
        C_[j0:j1, i0:i1] = C

        for j in range(-(nf // 2), nf // 2 + 1):
            j0 = (j + nf // 2) * n
            j1 = j0 + n
            k = i - j

            if np.abs(k) <= nf // 2 and k != 0:
                T[i0:i1, j0:j1] = np.einsum(
                    "ijk,j", H, QHat[:, k + nf // 2]
                ) + np.einsum("ijk,k", H, QHat[:, k + nf // 2])

                # Since we are considering time-invariant matrices B and C,
                # we do not need to populate the off-diagonal entries of
                # the matrices B_ and C_

    return T, B_, C_


# ----------------------------------------------------
# ----- Compute base flow using Newton's method ------
# ----------------------------------------------------


def compute_frequency_domain_residual(t, tensors, freqs, QHat, wf):
    _, nf = QHat.shape
    A, H, B = tensors[0], tensors[1], tensors[2]

    Q = ifft(freqs, QHat, t, 0)
    res = np.zeros(Q.shape)
    for k in range(len(t)):
        res[:, k] = evaluate_rhs(t[k], Q[:, k], A, H, B, wf)

    _, reshat = fft(t, res, nf // 2)
    reshat -= QHat * (1j * freqs)

    return reshat.reshape(-1, order="F")


def newton_harmonic_balance(t, tensors, freqs, QHat, wf, tol):
    n, nf = QHat.shape
    rhat = compute_frequency_domain_residual(t, tensors, freqs, QHat, wf)
    error = np.linalg.norm(rhat)

    iter = 0
    print("Computing the base flow using Newton's method...")
    print("Iteration %d,\t Residual norm = %1.12e" % (iter, error))

    while error > tol and iter < 20:
        T, _, _ = compute_lifted_frequency_domain_matrices(
            tensors, freqs, QHat
        )
        dQHat = scipy.linalg.solve(-T, rhat).reshape((n, nf), order="F")
        QHat = QHat + dQHat

        rhat = compute_frequency_domain_residual(t, tensors, freqs, QHat, wf)
        error = np.linalg.norm(rhat)

        iter += 1
        print("Iteration %d,\t Residual norm = %1.12e" % (iter, error))

    print("Done.")

    return QHat


# --------------------------------------------------------------------------
# ----- Compute the Gramians using the frequency-domain lifted -------------
# ----- Lyapunov equations (these are obtained by harmonic     -------------
# ----- balancing the differntial Lyapunov equations)          -------------
# --------------------------------------------------------------------------


def compute_gramian_coefficients(freqs, A, M):
    nf = len(freqs)
    n = A.shape[0] // nf
    X = scipy.linalg.solve_continuous_lyapunov(A, -M @ (M.conj().T))
    X = X[n * (nf // 2) : n * (nf // 2 + 1), :]

    return X


def reconstruct_gramian(freqs, X, t):
    n = X.shape[0]
    nf = X.shape[-1] // n

    P = np.zeros((n, n))
    for k in range(nf // 2 + 1):
        k0 = k * n
        k1 = (k + 1) * n

        if k < nf // 2:
            P += 2 * (X[:, k0:k1] * np.exp(-1j * freqs[k] * t)).real
        else:
            P += X[:, k0:k1].real

    return P


# --------------------------------------------------------------------------
# ----- Compute the Gramians using their time-domain definition ------------
# --------------------------------------------------------------------------


def evaluate_linearized_rhs(t, q, A, H, fQ, Tp):
    tt = np.mod(t, Tp)
    return (
        A + np.einsum("ijk,j", H, fQ(tt)) + np.einsum("ijk,k", H, fQ(tt))
    ) @ q


def compute_gramian_time_domain(tensors, tbflow, Q, t, taus):
    A, H, B = tensors[0], tensors[1], tensors[2]
    dtau = taus[1] - taus[0]
    X = np.zeros((A.shape[0], len(taus) * B.shape[-1]))
    fQ = scipy.interpolate.interp1d(
        tbflow, Q, kind="linear", fill_value="extrapolate"
    )
    Tp = tbflow[-1] + tbflow[1] - tbflow[0]

    count = 0
    for j in range(B.shape[-1]):
        q0 = B[:, j]

        for i in range(len(taus)):
            print("Impulse %d/%d" % (i + 1, len(taus)))
            sol = scipy.integrate.solve_ivp(
                evaluate_linearized_rhs,
                [taus[i], t],
                q0,
                "RK45",
                t_eval=[t],
                args=(A, H, fQ, Tp),
            )
            X[:, count] = sol.y[:, -1]
            count += 1

    return np.sqrt(dtau) * X


# --------------------------------------------------------------------------
# ----- Compute the Gramians using the frequential factors -----------------
# --------------------------------------------------------------------------


def compute_matrix_X(gammas, freqs, A, M):
    nf = len(freqs)
    m = M.shape[-1] // nf

    Id = np.diag(np.ones(A.shape[0]))
    X = np.zeros((A.shape[0], len(gammas) * m), dtype=np.complex128)

    for i in range(len(gammas)):
        i0 = i * m
        i1 = (i + 1) * m

        X[:, i0:i1] = scipy.linalg.solve(
            1j * gammas[i] * Id - A, M[:, m * (nf // 2) : m * (nf // 2 + 1)]
        ).reshape(-1, m)

    return X


# This function implements Algorithm 1 in the JCP paper
def compute_matrix_X_alg1(gammas, m, freqs, A, M):
    nf = len(freqs)
    n = A.shape[0] // nf
    p = M.shape[-1] // nf
    ncols_X = ((len(gammas) - 2) * (2 * m + 1) + 2 * (m + 1)) * p

    omega = freqs[nf // 2 + 1]
    Id = np.diag(np.ones(A.shape[0]))
    X = np.zeros((A.shape[0], ncols_X), dtype=np.complex128)

    gvec = np.zeros(ncols_X // p)
    count = 0
    for i in range(len(gammas)):
        # Compute factorization (or preconditioner)
        lu, piv = scipy.linalg.lu_factor(1j * gammas[i] * Id - A)

        if i == 0 or i == len(gammas) - 1:
            Range = np.arange(0, m + 1, 1)
        else:
            Range = np.arange(-m, m + 1, 1)

        for j in range(len(Range)):
            idx0 = count * p
            idx1 = (count + 1) * p

            Mj = M[:, p * (nf // 2) : p * (nf // 2 + 1)].copy()
            Xj = np.zeros_like(Mj, dtype=np.complex128)
            for k in range(p):
                Mjk = Mj[:, k].reshape((n, nf), order="F")
                Mj[:, k] = np.roll(Mjk, Range[j]).reshape(-1, order="F")
                sol = scipy.linalg.lu_solve((lu, piv), Mj[:, k]).reshape(
                    (n, nf), order="F"
                )
                Xj[:, k] = np.roll(sol, -Range[j]).reshape(-1, order="F")

            X[:, idx0:idx1] = Xj

            gvec[count] = np.abs(gammas[i] + Range[j] * omega)
            count += 1

    return X, np.sort(np.abs(gvec))


def evaluate_gramian(gammas, freqs, X, t):
    nf = len(freqs)
    n = X.shape[0] // nf
    m = X.shape[-1] // len(gammas)
    dg = gammas[1] - gammas[0]

    Z = np.zeros((n, X.shape[-1]), dtype=np.complex128)

    for i in range(X.shape[-1]):
        if gammas[i // m] == 0:
            cxi = np.sqrt(0.5 * dg / np.pi)
        else:
            cxi = np.sqrt(dg / np.pi)

        Xi = X[:, i].reshape((n, nf), order="F")
        Z[:, i] = cxi * np.sum(Xi * np.exp(1j * freqs * t), axis=-1)

    Z = np.concatenate((Z.real, Z.imag), axis=-1)
    P = Z @ Z.T

    return P
