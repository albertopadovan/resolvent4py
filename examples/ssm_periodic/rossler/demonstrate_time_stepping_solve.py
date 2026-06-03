"""
Compare the algebraic and time-stepping
:meth:`RosslerPeriodic.solve_linear_system` paths on a single solve.

The algebraic path forms the harmonic-balanced matrix
:math:`s I - L_{\\mathrm{HB}}` and inverts it with MUMPS.  The
time-stepping path integrates the equivalent linear ODE

.. math::

    \\dot{q} = (A(t) - s I)\\, q + b(t)

to its :math:`T`-periodic steady state with
:func:`compute_post_transient_solution` and FFTs the response.  Both
return the same HB long vector when ``Re(s) > 0`` so the shifted
system is stable.

Run with::

    mpiexec -n <nprocs> python demonstrate_time_stepping_solve.py
"""

import sys
import numpy as np
from scipy.signal import resample

from petsc4py import PETSc
import resolvent4py as res4py

from rossler_differential_equation import RosslerPeriodic


comm = PETSc.COMM_WORLD


# ── Parameters (small ones for a fast comparison) ───────────────────────────
nf = 20
nfb = 10
shift = 0.1     # Re(s) > 0 → shifted Floquet system is stable
ts_dt = 1e-2           # let RosslerPeriodic pick T/200
gmres_rtol = 1e-10


# ── Load Rössler periodic orbit ─────────────────────────────────────────────
data = np.load("data/periodic_orbit.npz")
c = float(data["c"])
T = float(data["T"])
C_periodic_full = data["C"][:, :-1]    # drop repeated endpoint

n_time = 2 * (nf + nfb) + 1
C_periodic = resample(C_periodic_full, n_time, axis=1)
time_orbit = np.linspace(0, T, n_time, endpoint=False)

res4py.petscprint(
    comm,
    f"Rossler: c={c}, T={T:.6f}, nf={nf}, nfb={nfb}, shift={shift}",
)


# ── Build both equations ────────────────────────────────────────────────────
eq_alg = RosslerPeriodic(
    c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit
)
eq_ts = RosslerPeriodic(
    c=c, nf=nf, nfb=nfb, c_star=C_periodic, time=time_orbit,
    use_time_stepping=True,
    ts_dt=ts_dt,
    gmres_rtol=gmres_rtol,
    ts_verbose=1,
)


# ── Conjugate-symmetric complex HB rhs ──────────────────────────────────────
# b_{-k} = conj(b_k) so the rhs corresponds to a *real* time-domain forcing.
# Combined with a real ``shift``, both x_alg and x_ts inherit the conjugate
# symmetry, and the per-mode errors at ±k should be conjugates (modulus-equal).
N_hb = eq_alg.n * (2 * nf + 1)
rng = np.random.default_rng(42)
b_pos = (
    rng.standard_normal((nf + 1, eq_alg.n))
    + 1j * rng.standard_normal((nf + 1, eq_alg.n))
)
b_pos[0] = b_pos[0].real                       # DC block must be real
b_full = np.zeros((2 * nf + 1, eq_alg.n), dtype=complex)
b_full[nf:] = b_pos                            # HB blocks nf..2nf → modes 0..+nf
b_full[:nf] = b_pos[1:][::-1].conj()           # HB blocks 0..nf-1 → modes -nf..-1
rhs_np = b_full.ravel()

rhs = PETSc.Vec().create(comm=comm)
rhs.setSizes((res4py.compute_local_size(N_hb), N_hb))
rhs.setUp()
r0, r1 = rhs.getOwnershipRange()
rhs.setValues(np.arange(r0, r1, dtype=PETSc.IntType), rhs_np[r0:r1])
rhs.assemble()


# ── Solve with both paths ───────────────────────────────────────────────────
res4py.petscprint(comm, "Algebraic solve …")
x_alg = eq_alg.solve_linear_system(shift, rhs)

res4py.petscprint(comm, "Time-stepping solve …")
x_ts = eq_ts.solve_linear_system(shift, rhs)


# ── Compare ─────────────────────────────────────────────────────────────────
x_alg_seq = res4py.distributed_to_sequential_vector(x_alg)
x_ts_seq = res4py.distributed_to_sequential_vector(x_ts)
x_alg_np = x_alg_seq.getArray().copy()
x_ts_np = x_ts_seq.getArray().copy()
x_alg_seq.destroy()
x_ts_seq.destroy()

abs_err = np.linalg.norm(x_alg_np - x_ts_np)
rel_err = abs_err / np.linalg.norm(x_alg_np)

res4py.petscprint(
    comm,
    f"||x_ts - x_alg||      = {abs_err:.3e}\n"
    f"||x_ts - x_alg|| / ||x_alg|| = {rel_err:.3e}",
)

# Per-harmonic breakdown so we can spot bandwidth-edge errors if any.
x_alg_blocks = x_alg_np.reshape(2 * nf + 1, eq_alg.n)
x_ts_blocks = x_ts_np.reshape(2 * nf + 1, eq_ts.n)
res4py.petscprint(comm, "Per-harmonic relative error (HB index → mode):")
for k in range(2 * nf + 1):
    mode = k - nf
    ref = np.linalg.norm(x_alg_blocks[k])
    err = (
        np.linalg.norm(x_alg_blocks[k] - x_ts_blocks[k]) / ref
        if ref > 1e-14
        else 0.0
    )
    res4py.petscprint(comm, f"  HB[{k:3d}]  mode {mode:+3d}  rel err = {err:.2e}")

sys.exit()