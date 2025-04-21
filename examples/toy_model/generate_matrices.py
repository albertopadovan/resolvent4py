import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

import toy_model as toy
import resolvent4py as res4py
from petsc4py import PETSc
from mpi4py import MPI

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
nf = 10
freqs, QHat = toy.fft(time, Q, nf)

QHat = toy.newton_harmonic_balance(time, tensors, freqs, QHat, wf, 1e-9)
T, _, _ = toy.compute_lifted_frequency_domain_matrices(tensors, freqs, QHat)

Q = toy.ifft(freqs, QHat, time, 0)
fQ = scipy.interpolate.interp1d(
    time, Q, kind="linear", fill_value="extrapolate"
)
As = np.zeros((9, len(time)), dtype=np.float64)
for j in range(len(time)):
    tj = time[j]
    Aj = A + np.einsum("ijk,j", H, fQ(tj)) + np.einsum("ijk,k", H, fQ(tj))
    As[:, j] = Aj.reshape(-1)

Ahat = (1 / len(time)) * np.fft.rfft(As, axis=-1)[:, : len(freqs) // 2 + 1]

save_path = "data/"
os.makedirs(save_path) if not os.path.exists(save_path) else None



for j in range(Ahat.shape[-1]):
    Aj = Ahat[:, j].reshape((3, 3))
    Aj = scipy.sparse.coo_matrix(Aj)
    rows = Aj.row
    cols = Aj.col
    data = Aj.data

    idces = np.argwhere(np.abs(data) >= 1e-16)
    rows = rows[idces]
    cols = cols[idces]
    data = data[idces]

    arrays = [rows, cols, data]
    fnames = ["rows_%02d.dat" % j, "cols_%02d.dat" % j, "vals_%02d.dat" % j]
    for i, array in enumerate(arrays):
        fname = save_path + fnames[i]
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_WORLD, fname, vec)

perts_freqs = freqs.copy()
bflow_freqs = freqs[len(freqs) // 2 :]

np.save(save_path + 'bflow_freqs.npy', bflow_freqs)
np.save(save_path + 'perts_freqs.npy', perts_freqs)
