import numpy as np
import scipy
import matplotlib.pyplot as plt
import time as tlib

import toy_model as toy

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)


# -------------------------------------------------------------------
# --------- Define system -------------------------------------------
# -------------------------------------------------------------------

n = 3  # number of degrees of freedom
mu = 1 / 5
alpha = 1 / 5
beta = 1 / 5
params = [mu, alpha, beta]

omega = np.sqrt(
    1 - beta**2 * mu / alpha
)  # natural frequency of the unforced limit cycle
A = np.asarray([[mu, -1, 0], [1, mu, 0], [0, 0, -alpha]])
H = toy.compute_third_order_tensor(params, n)
B = 1e-1 * np.asarray([1, 1, 0]).reshape(-1, 1)
C = np.ones(n).reshape(1, -1)
tensors = [A, H, B, C]

# -------------------------------------------------------------------
# --------- Compute base flow ---------------------------------------
# -------------------------------------------------------------------

wf = 2 * omega  # Forcing frequency (twice the natural frequency)
Tw = 2 * np.pi / omega
dt = Tw / 300
time = np.arange(0, 100 * Tw, dt)
sol = scipy.integrate.solve_ivp(
    toy.evaluate_rhs,
    [0, time[-1]],
    np.zeros(n),
    "RK45",
    t_eval=time,
    args=(A, H, B, wf),
)
Q = sol.y

# Generate initial condition for Newton's method
idx0 = np.argmin(np.abs(time - 98 * Tw))
idxf = np.argmin(np.abs(time - 99 * Tw))
time = time[idx0:idxf] - time[idx0]
Q = Q[:, idx0:idxf]
freqs, QHat = toy.fft(time, Q, 10)

QHat = toy.newton_harmonic_balance(time, tensors, freqs, QHat, wf, 1e-9)
Q = toy.ifft(freqs, QHat, time, 0)


# Generate Figure 1(a) in the JCP manuscript
dQHat = QHat * (1j * freqs)
dQ = toy.ifft(freqs, dQHat, time, 0)

ax = plt.figure().add_subplot(projection="3d")
sc = ax.scatter(
    Q[0,], Q[1,], Q[2,], c=np.linalg.norm(dQ, axis=0), cmap="inferno", s=4
)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor("w")
ax.yaxis.pane.set_edgecolor("w")
ax.zaxis.pane.set_edgecolor("w")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_zlim(0.9, 1.1)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([0.9, 1.0, 1.1])

plt.colorbar(sc, location="right", shrink=0.8, pad=0.15)
plt.tight_layout()
# plt.savefig("./Figures/limit_cycle.eps",format='eps')


# Generate Figure 1(b) in the JCP manuscript
fQ = scipy.interpolate.interp1d(
    time, Q, kind="linear", fill_value="extrapolate"
)
Tp = Tw

tf = 50 * Tw
dtau = Tw / 10
taus = np.arange(0, tf, dtau)

sol = scipy.integrate.solve_ivp(
    toy.evaluate_linearized_rhs,
    [taus[0], tf],
    B[:, 0],
    "RK45",
    t_eval=taus,
    args=(A, H, fQ, Tp),
)


ylabs = [r"$x^\prime$", r"$y^\prime$", r"$z^\prime$"]

fig, ax = plt.subplots(nrows=3, ncols=1)
for k in range(3):
    ax[k].plot(taus, sol.y[k,], "k")
    ax[k].set_ylabel(ylabs[k])

    if k < 2:
        ax[k].set_xticks([])
    else:
        ax[k].set_xticks(Tw * np.arange(0, 51, 10))
        ax[k].set_xticklabels(
            [r"$0$", r"$10\,T$", r"$20\,T$", r"$30\,T$", r"$40\,T$", r"$50\,T$"]
        )
        ax[k].set_xlabel("Time $t$")

plt.tight_layout()
# plt.savefig("./Figures/response.eps",format='eps')

# %%
# -------------------------------------------------------------------
# --------- Compute the Gramians using different methods ------------
# -------------------------------------------------------------------


T, B_, C_ = toy.compute_lifted_frequency_domain_matrices(tensors, freqs, QHat)

t0 = tlib.time()
X = toy.compute_gramian_coefficients(freqs, T, B_)
Pl = toy.reconstruct_gramian(freqs, X, np.mod(tf, Tw))
t1 = tlib.time()

print("Execution time using the lyapunov equation = %1.03e" % (t1 - t0))


tf = 30 * Tw
dtau = Tw / 10
taus = np.arange(0, tf, dtau)

t0 = tlib.time()
X = toy.compute_gramian_time_domain(tensors, time, Q, tf, taus)
Pt = X @ X.T
t1 = tlib.time()

E = Pt - Pl
error = np.trace(E.T @ E) / np.trace(Pl.T @ Pl)

print(
    "Execution time in the time domain = %1.03e,\t Error = %1.03e"
    % (t1 - t0, error)
)

m = 10
gammas = np.linspace(0, omega / 2, num=30)
t0 = tlib.time()
X, gs = toy.compute_matrix_X_alg1(gammas, m, freqs, T, B_)
Pe = toy.evaluate_gramian(gs, freqs, X, np.mod(tf, Tw))
t1 = tlib.time()

E = Pe - Pl
error = np.trace(E.T @ E) / np.trace(Pl.T @ Pl)

print(
    "Execution time using the efficient alg. in the "
    + "frequency domain = %1.03e,\t Error = %1.03e" % (t1 - t0, error)
)

t0 = tlib.time()
X = toy.compute_matrix_X(gs, freqs, T, B_)
Pi = toy.evaluate_gramian(gs, freqs, X, np.mod(tf, Tw))
t1 = tlib.time()

E = Pi - Pl
error = np.trace(E.T @ E) / np.trace(Pl.T @ Pl)

print(
    "Execution time using the inefficient alg. in the "
    + "frequency domain = %1.03e,\t Error = %1.03e" % (t1 - t0, error)
)
