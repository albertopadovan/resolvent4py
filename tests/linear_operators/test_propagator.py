import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


# ── helpers for the time-periodic A(t) tests ────────────────────────────────


def _make_periodic_A_coeffs(N, seed):
    r"""One-sided coefficients of a real T-periodic
    :math:`A(t) = A_0 + 2 \mathrm{Re}[A_1 e^{i\omega t} + A_2 e^{i 2\omega t}]`
    with ``omega = 1.2``.

    To keep the forward propagator stable over the full period, the
    eigenvalues of the random :math:`A_0` are shifted into the left
    half plane so that every eigenvalue has real part :math:`\le -0.5`
    (a uniform shift if needed; no shift if :math:`A_0` is already
    stable), and the AC coefficients :math:`A_1, A_2` are multiplied
    by :math:`\varepsilon = 10^{-2}` so the periodic perturbation is
    small relative to the decay rate.
    """
    rng = np.random.default_rng(seed)
    freqs = np.array([0.0, 1.2, 2.4])

    A0 = rng.standard_normal((N, N))
    evals, V = np.linalg.eig(A0)
    shift = max(0.0, evals.real.max() + 0.5)
    new_evals = evals - shift
    A0 = (V @ np.diag(new_evals) @ np.linalg.inv(V)).real.astype(np.complex128)

    eps = 1e-1
    A1 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    A2 = eps * (
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    )
    return [A0, A1, A2], freqs


def _A_at(t, Alst_np, freqs):
    r"""Build the real :math:`A(t)` as a dense numpy matrix at a single
    time, reconstructing :math:`A_{-k} = \overline{A_k}` for ``k > 0``."""
    n = Alst_np[0].shape[0]
    A = np.zeros((n, n), dtype=np.complex128)
    for Ak, omega_k in zip(Alst_np, freqs):
        A += Ak * np.exp(1j * omega_k * t)
        if omega_k > 0.0:
            A += np.conj(Ak) * np.exp(-1j * omega_k * t)
    return A


def _build_time_periodic_op(comm, Alst_np, freqs, time):
    r"""Wrap the numpy Fourier coefficients into a
    :class:`TimePeriodicMatrixLinearOperator`."""
    Apetsc_lst = [pytest_utils.numpy_to_petsc(comm, Ak) for Ak in Alst_np]
    linop_lst = [
        res4py.linear_operators.MatrixLinearOperator(A) for A in Apetsc_lst
    ]
    return res4py.linear_operators.TimePeriodicMatrixLinearOperator(
        linop_lst, freqs, time
    )


def test_propagator_on_vectors(comm, square_stable_random_matrix):
    r"""Test PropagatorLinearOperator on vectors"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.PropagatorLinearOperator(
        Alop, 0.0, tf, dt, method="RK3"
    )
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply, linop.apply_hermitian_transpose]

    y = linop.create_left_vector()
    error_vec = [
        pytest_utils.compute_error_vector(
            comm, actions_petsc[i], x, y, actions_python[i], xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    Alop.destroy()
    assert error < 1e-11


def test_propagator_on_bvs(comm, square_stable_random_matrix):
    r"""Test PropagatorLinearOperator on BVs"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.PropagatorLinearOperator(
        Alop, 0.0, tf, dt, method="RK3"
    )
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 3))

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply_mat, linop.apply_hermitian_transpose_mat]

    Y = linop.create_left_bv(X.getSizes()[-1])
    error_vec = [
        pytest_utils.compute_error_bv(
            comm, actions_petsc[i], X, Y, actions_python[i], Xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    X.destroy()
    Y.destroy()
    linop.destroy()
    Alop.destroy()
    assert error < 1e-11


def test_propagator_rk2_on_vectors(comm, square_stable_random_matrix):
    r"""Test PropagatorLinearOperator with RK2 on vectors"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.PropagatorLinearOperator(
        Alop, 0.0, tf, dt, method="RK2"
    )
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply, linop.apply_hermitian_transpose]

    y = linop.create_left_vector()
    error_vec = [
        pytest_utils.compute_error_vector(
            comm, actions_petsc[i], x, y, actions_python[i], xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    Alop.destroy()
    assert error < 1e-7


def test_propagator_rk2_on_bvs(comm, square_stable_random_matrix):
    r"""Test PropagatorLinearOperator with RK2 on BVs"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.PropagatorLinearOperator(
        Alop, 0.0, tf, dt, method="RK2"
    )
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 3))

    ExpPython = sp.linalg.expm(Apython * tf)
    actions_python = [ExpPython.dot, ExpPython.conj().T.dot]
    actions_petsc = [linop.apply_mat, linop.apply_hermitian_transpose_mat]

    Y = linop.create_left_bv(X.getSizes()[-1])
    error_vec = [
        pytest_utils.compute_error_bv(
            comm, actions_petsc[i], X, Y, actions_python[i], Xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    X.destroy()
    Y.destroy()
    linop.destroy()
    Alop.destroy()
    assert error < 1e-7


def test_propagator_y_none(comm, square_stable_random_matrix):
    r"""Test PropagatorLinearOperator with Y=None"""
    tf = 2.1
    dt = 1e-5
    Apetsc, Apython = square_stable_random_matrix
    Alop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    linop = res4py.linear_operators.PropagatorLinearOperator(
        Alop, 0.0, tf, dt, method="RK3"
    )

    ExpPython = sp.linalg.expm(Apython * tf)

    # Vec Y=None
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])
    y = linop.apply(x)
    ys = res4py.distributed_to_sequential_vector(y)
    ypython = ExpPython.dot(xpython)
    error_vec = np.linalg.norm(ypython - ys.getArray()) / np.linalg.norm(
        ypython
    )
    ys.destroy()
    x.destroy()
    y.destroy()

    # BV Y=None
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 3))
    Y = linop.apply_mat(X)
    Ym = Y.getMat()
    Yms = res4py.distributed_to_sequential_matrix(Ym)
    Y.restoreMat(Ym)
    Ypython = ExpPython.dot(Xpython)
    error_bv = np.linalg.norm(Ypython - Yms.getDenseArray()) / np.linalg.norm(
        Ypython
    )
    Yms.destroy()
    X.destroy()
    Y.destroy()

    error = np.linalg.norm([error_vec, error_bv])
    linop.destroy()
    Alop.destroy()
    assert error < 1e-11


# ── time-periodic A(t): forward and adjoint propagators ────────────────────


def test_propagator_time_periodic_forward(comm):
    r"""Forward propagator with time-periodic :math:`A(t)`.

    Integrate :math:`dx/dt = A(t)\, x,\; x(0) = x_0` from 0 to one
    period :math:`T = 2\pi/\omega` with the resolvent4py
    :class:`PropagatorLinearOperator` (RK3) and compare against
    :func:`scipy.integrate.solve_ivp` (RK45) on the dense numpy
    reference.
    """
    N = 8
    omega = 1.2
    T = 2.0 * np.pi / omega
    tf = T
    dt = 1e-4
    Alst_np, freqs = _make_periodic_A_coeffs(N, seed=101)

    Atop = _build_time_periodic_op(comm, Alst_np, freqs, time=0.0)
    Prop = res4py.linear_operators.PropagatorLinearOperator(
        Atop, 0.0, tf, dt, method="RK3"
    )

    x, xp = pytest_utils.generate_random_vector(comm, N, complex=False)
    y = Prop.apply(x)
    ys = res4py.distributed_to_sequential_vector(y).getArray().copy()

    def rhs(t, v):
        return _A_at(t, Alst_np, freqs) @ v

    sol = sp.integrate.solve_ivp(
        rhs,
        (0.0, tf),
        xp.astype(np.complex128),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    y_ref = sol.y[:, -1]

    error = np.linalg.norm(ys - y_ref) / np.linalg.norm(y_ref)

    x.destroy()
    y.destroy()
    Prop.destroy()
    for Aop in Atop.Alst:
        Aop.A.destroy()
        Aop.destroy()
    Atop.destroy()
    assert error < 1e-6, f"forward propagator error = {error:.2e}"


def test_propagator_time_periodic_adjoint(comm):
    r"""Adjoint propagator with time-periodic :math:`A(t)`.

    Integrate :math:`-dx/dt = A^{\ast}(t)\, x,\; x(t_f) = x_0`
    backward in time from :math:`t_f = T` to 0 with the resolvent4py
    :class:`PropagatorLinearOperator.apply_hermitian_transpose` (RK3)
    and compare against :func:`scipy.integrate.solve_ivp` on the dense
    numpy reference, integrating backward via ``t_span=(tf, 0)`` with
    RHS :math:`-A^{\ast}(t)\, x`.
    """
    N = 8
    omega = 1.2
    T = 2.0 * np.pi / omega
    tf = T
    dt = 1e-4
    Alst_np, freqs = _make_periodic_A_coeffs(N, seed=103)

    Atop = _build_time_periodic_op(comm, Alst_np, freqs, time=0.0)
    Prop = res4py.linear_operators.PropagatorLinearOperator(
        Atop, 0.0, tf, dt, method="RK3"
    )

    x, xp = pytest_utils.generate_random_vector(comm, N, complex=False)
    y = Prop.apply_hermitian_transpose(x)
    ys = res4py.distributed_to_sequential_vector(y).getArray().copy()

    def rhs_adj(t, v):
        return -_A_at(t, Alst_np, freqs).conj().T @ v

    sol = sp.integrate.solve_ivp(
        rhs_adj,
        (tf, 0.0),
        xp.astype(np.complex128),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    y_ref = sol.y[:, -1]

    error = np.linalg.norm(ys - y_ref) / np.linalg.norm(y_ref)

    x.destroy()
    y.destroy()
    Prop.destroy()
    for Aop in Atop.Alst:
        Aop.A.destroy()
        Aop.destroy()
    Atop.destroy()
    assert error < 1e-6, f"adjoint propagator error = {error:.2e}"
