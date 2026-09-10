import importlib

import numpy as np
import pytest
import scipy as sp
from petsc4py import PETSc

import resolvent4py as res4py
from .. import pytest_utils

owns_mod = importlib.import_module("resolvent4py.linalg.resolvent_analysis_owns")


def _petsc_matrix(array):
    array = np.asarray(array, dtype=complex)
    rows, cols = array.shape
    mat = PETSc.Mat().createAIJ(((rows, rows), (cols, cols)), comm=PETSc.COMM_SELF)
    mat.setUp()
    nz_rows, nz_cols = np.nonzero(array)
    for row, col in zip(nz_rows, nz_cols):
        mat.setValue(int(row), int(col), array[row, col])
    mat.assemble()
    return mat


class ProjectedOWNSCase:
    """Small projected OWNS-P problem."""

    def __init__(self, L, x_grid):
        self.comm = PETSc.COMM_SELF
        self.x_grid = x_grid
        self.dx = np.diff(x_grid)
        self.n_x = len(x_grid)

        n = L.shape[0]
        identity = np.eye(n, dtype=complex)

        recursion = res4py.linalg.OWNSRecursionParameters(
            beta_plus=np.asarray([0.25 + 0.10j]),
            beta_minus=np.asarray([-0.30 - 0.15j]),
        )
        station = res4py.linalg.OWNSStation(
            A=_petsc_matrix(np.diag([1.0, -1.0])),
            L=_petsc_matrix(L),
            input_map=_petsc_matrix(identity),
            output_map=_petsc_matrix(identity),
            input_weight=_petsc_matrix(identity),
            output_weight=_petsc_matrix(identity),
            recursion=recursion,
        )
        self.stations = tuple(station for _ in range(self.n_x))

    def get_station(self, x_idx):
        return self.stations[x_idx]

    def destroy(self):
        for matrix in (
            self.stations[0].A,
            self.stations[0].L,
            self.stations[0].input_map,
            self.stations[0].output_map,
            self.stations[0].input_weight,
            self.stations[0].output_weight,
        ):
            matrix.destroy()


class OWNSTestLinearOperator(res4py.linear_operators.LinearOperator):
    """LinearOperator wrapper for the test OWNS march."""

    def __init__(self, owns_ops):
        self.owns_ops = owns_ops
        self.marcher = owns_mod._Marcher(owns_ops, verbose=0)
        station = owns_ops.get_station(0)
        self.n_input = station.n_input
        self.n_output = station.n_output
        dimensions = ((self.n_output * owns_ops.n_x, self.n_output * owns_ops.n_x), (self.n_input * owns_ops.n_x, self.n_input * owns_ops.n_x))
        super().__init__(PETSc.COMM_SELF, "OWNSTestLinearOperator", dimensions)

    def check_if_real_valued(self):
        return False

    def apply(self, x, y=None):
        y = self.create_left_vector() if y is None else y
        forcing = x.getArray(readonly=True).reshape((self.n_input, self.owns_ops.n_x, 1), order="F")
        response = self.marcher.direct(forcing, 1, 1).reshape(-1, order="F")
        y.setValues(np.arange(response.size, dtype=PETSc.IntType), response)
        y.assemble()
        return y

    def apply_mat(self, X, Y=None):
        Y = self.create_left_bv(X.getSizes()[-1]) if Y is None else Y
        for column in range(X.getSizes()[-1]):
            x = X.getColumn(column)
            y = Y.getColumn(column)
            try:
                self.apply(x, y)
            finally:
                Y.restoreColumn(column, y)
                X.restoreColumn(column, x)
        return Y

    def destroy(self):
        pass


def _exact_resolvent(owns_ops):
    n_input = owns_ops.get_station(0).n_input
    n_output = owns_ops.get_station(0).n_output
    n_x = owns_ops.n_x

    total_input = n_input * n_x
    total_output = n_output * n_x

    operator = OWNSTestLinearOperator(owns_ops)
    resolvent = np.zeros((total_output, total_input), dtype=complex)
    x = operator.create_right_vector()
    y = operator.create_left_vector()
    try:
        for column in range(total_input):
            x.set(0.0)
            x.setValue(column, 1.0)
            x.assemble()
            operator.apply(x, y)
            resolvent[:, column] = y.getArray(readonly=True)
    finally:
        y.destroy()
        x.destroy()
        operator.destroy()

    return resolvent


def test_owns_resolvent_analysis_matches_dense_svd(comm, square_matrix_size):
    """Test projected OWNS gain against a dense SVD."""
    _, L = pytest_utils.generate_stable_random_matrix(comm, (2, 2))
    L = 0.1 * L - 0.2 * np.eye(2)
    owns_ops = ProjectedOWNSCase(L, np.linspace(0.0, 1.0, 6))

    try:
        _, gains, _ = res4py.linalg.resolvent_analysis_owns(
            owns_ops,
            n_modes=1,
            max_iter=80,
            tol=1e-12,
            verbose=0,
        )
        _, singular_values, _ = sp.linalg.svd(
            _exact_resolvent(owns_ops),
            full_matrices=False,
        )
    finally:
        owns_ops.destroy()

    error = 100 * np.max(np.abs((gains - singular_values[:1]) / singular_values[:1]))
    assert error < 5e-1


# ──────────────────────────────────────────────────────────────────────────────
# Unit coverage for the OWNS internals
#
# The composite test above runs entirely on COMM_SELF, so it exercises the
# algorithm but not the distributed paths.  The linear-algebra helpers below
# are therefore driven on COMM_WORLD, where they must agree with numpy on
# every rank count.
# ──────────────────────────────────────────────────────────────────────────────


def _owns_operator(comm, A_np):
    r"""Wrap a numpy matrix as the module does, with a direct solver."""
    A = pytest_utils.numpy_to_petsc(comm, A_np)
    return owns_mod._matrix_operator(A, solve=True)


# --- pure helpers -------------------------------------------------------------


def test_bdf_coefficients_are_consistent():
    r"""BDF1 on the first step, BDF2 thereafter.  A linear multistep method
    is consistent only if its coefficients sum to zero -- that is what makes
    it exact on constants."""
    assert owns_mod._bdf_coefficients(1) == (1.0, -1.0)
    for index in (2, 3, 7):
        assert owns_mod._bdf_coefficients(index) == (1.5, -2.0, 0.5)
    for index in (1, 2, 3, 7):
        coefficients = owns_mod._bdf_coefficients(index)
        assert abs(sum(coefficients)) < 1e-15, (
            f"index {index}: coefficients {coefficients} are inconsistent"
        )


def test_natural_characteristic_order_defaults_to_identity():
    r"""With no point count the ordering is the identity."""
    assert owns_mod._natural_characteristic_order(5, None) == [0, 1, 2, 3, 4]


def test_natural_characteristic_order_is_a_permutation():
    r"""Whatever the ordering, it must be a bijection onto range(n_state) --
    a repeated or missing index would silently corrupt every subblock
    extraction downstream."""
    for n_state, n_points in ((6, 3), (6, 2), (12, 4), (8, 8), (9, 1)):
        order = owns_mod._natural_characteristic_order(n_state, n_points)
        assert len(order) == n_state
        assert sorted(order) == list(range(n_state)), (
            f"n_state={n_state} n_points={n_points} is not a permutation: {order}"
        )


def test_natural_characteristic_order_rejects_bad_point_counts():
    r"""n_points must be positive and divide the state dimension."""
    for bad in (0, -2, 4, 5):
        with pytest.raises(ValueError):
            owns_mod._natural_characteristic_order(6, bad)


def test_recursion_parameters_order():
    r"""The recursion order is the number of beta_plus parameters."""
    params = res4py.linalg.OWNSRecursionParameters(
        beta_plus=np.asarray([0.1 + 0.2j, 0.3 - 0.1j, 0.5j]),
        beta_minus=np.asarray([-0.1, -0.2, -0.3]),
    )
    assert params.order == 3


def test_owns_station_infers_dimensions():
    r"""n_state / n_input / n_output come from the map shapes, not from
    redundant user-supplied sizes."""
    n_state, n_input, n_output = 4, 2, 3
    station = res4py.linalg.OWNSStation(
        A=_petsc_matrix(np.eye(n_state)),
        L=_petsc_matrix(np.eye(n_state)),
        input_map=_petsc_matrix(np.zeros((n_state, n_input))),
        output_map=_petsc_matrix(np.zeros((n_output, n_state))),
        input_weight=_petsc_matrix(np.eye(n_input)),
        output_weight=_petsc_matrix(np.eye(n_output)),
        recursion=res4py.linalg.OWNSRecursionParameters(
            beta_plus=np.asarray([0.1]), beta_minus=np.asarray([-0.1])
        ),
    )
    assert station.n_state == n_state
    assert station.n_input == n_input
    assert station.n_output == n_output
    for matrix in (
        station.A,
        station.L,
        station.input_map,
        station.output_map,
        station.input_weight,
        station.output_weight,
    ):
        matrix.destroy()


# --- linear-algebra helpers, distributed --------------------------------------


def test_apply_array_matches_numpy(comm):
    r"""_apply_array must reproduce A @ V, and its adjoint A^H @ V."""
    N = 8
    rng = np.random.default_rng(21)
    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    V_np = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
    A_np = comm.tompi4py().bcast(A_np, root=0)
    V_np = comm.tompi4py().bcast(V_np, root=0)

    operator = _owns_operator(comm, A_np)
    try:
        forward = owns_mod._apply_array(operator, V_np)
        adjoint = owns_mod._apply_array(operator, V_np, adjoint=True)
    finally:
        owns_mod._destroy_owned_operator(operator)

    fwd_err = np.linalg.norm(forward - A_np @ V_np) / np.linalg.norm(A_np @ V_np)
    adj_ref = A_np.conj().T @ V_np
    adj_err = np.linalg.norm(adjoint - adj_ref) / np.linalg.norm(adj_ref)
    assert fwd_err < 1e-10, f"_apply_array forward error {fwd_err:.2e}"
    assert adj_err < 1e-10, f"_apply_array adjoint error {adj_err:.2e}"


def test_apply_array_accepts_a_1d_vector(comm):
    r"""A 1-D input is treated as a single column, not broadcast."""
    N = 6
    rng = np.random.default_rng(22)
    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    v_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    A_np = comm.tompi4py().bcast(A_np, root=0)
    v_np = comm.tompi4py().bcast(v_np, root=0)

    operator = _owns_operator(comm, A_np)
    try:
        got = owns_mod._apply_array(operator, v_np)
    finally:
        owns_mod._destroy_owned_operator(operator)

    assert got.shape == (N, 1)
    error = np.linalg.norm(got[:, 0] - A_np @ v_np) / np.linalg.norm(A_np @ v_np)
    assert error < 1e-10, f"1-D _apply_array error {error:.2e}"


def test_solve_array_matches_numpy(comm):
    r"""_solve_array must reproduce A^-1 @ V and A^-H @ V."""
    N = 8
    rng = np.random.default_rng(23)
    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A_np = A_np + N * np.eye(N)  # keep it well conditioned
    V_np = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    A_np = comm.tompi4py().bcast(A_np, root=0)
    V_np = comm.tompi4py().bcast(V_np, root=0)

    operator = _owns_operator(comm, A_np)
    try:
        forward = owns_mod._solve_array(operator, V_np)
        adjoint = owns_mod._solve_array(operator, V_np, adjoint=True)
    finally:
        owns_mod._destroy_owned_operator(operator)

    fwd_ref = np.linalg.solve(A_np, V_np)
    adj_ref = np.linalg.solve(A_np.conj().T, V_np)
    fwd_err = np.linalg.norm(forward - fwd_ref) / np.linalg.norm(fwd_ref)
    adj_err = np.linalg.norm(adjoint - adj_ref) / np.linalg.norm(adj_ref)
    assert fwd_err < 1e-8, f"_solve_array forward error {fwd_err:.2e}"
    assert adj_err < 1e-8, f"_solve_array adjoint error {adj_err:.2e}"


def test_solve_matrix_satisfies_the_system(comm):
    r"""_solve_matrix returns X with A @ X = B -- checked by residual rather
    than against an inverse, so it holds regardless of the solver used."""
    N, ncols = 7, 3
    rng = np.random.default_rng(24)
    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A_np = A_np + N * np.eye(N)
    B_np = rng.standard_normal((N, ncols)) + 1j * rng.standard_normal((N, ncols))
    A_np = comm.tompi4py().bcast(A_np, root=0)
    B_np = comm.tompi4py().bcast(B_np, root=0)

    operator = _owns_operator(comm, A_np)
    B = pytest_utils.numpy_to_petsc(comm, B_np)
    try:
        X = owns_mod._solve_matrix(operator, B)
        try:
            X_np = np.column_stack(
                [
                    res4py.distributed_to_sequential_vector(
                        X.getColumnVector(j)
                    ).getArray().copy()
                    for j in range(ncols)
                ]
            )
        finally:
            X.destroy()
    finally:
        B.destroy()
        owns_mod._destroy_owned_operator(operator)

    residual = np.linalg.norm(A_np @ X_np - B_np) / np.linalg.norm(B_np)
    assert residual < 1e-8, f"_solve_matrix residual {residual:.2e}"


def test_q_matrix(comm):
    r"""_q_matrix builds Q = L - i*beta*A."""
    N = 6
    beta = 0.4 - 0.7j
    rng = np.random.default_rng(25)
    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    L_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A_np = comm.tompi4py().bcast(A_np, root=0)
    L_np = comm.tompi4py().bcast(L_np, root=0)

    A = pytest_utils.numpy_to_petsc(comm, A_np)
    L = pytest_utils.numpy_to_petsc(comm, L_np)
    Q = owns_mod._q_matrix(A, L, beta)

    x, x_np = pytest_utils.generate_random_vector(comm, N)
    y = Q.createVecLeft()
    Q.mult(x, y)
    y_seq = res4py.distributed_to_sequential_vector(y)
    expected = (L_np - 1j * beta * A_np) @ x_np
    error = np.linalg.norm(y_seq.getArray() - expected) / np.linalg.norm(expected)

    for obj in (y_seq, y, x, Q, L, A):
        obj.destroy()
    assert error < 1e-10, f"_q_matrix error {error:.2e}"


# --- behaviour of the public entry point --------------------------------------


def test_owns_multiple_modes_are_ordered_and_match_dense_svd(comm):
    r"""Asking for two modes must return two gains, positive and in
    descending order, matching the leading dense singular values."""
    _, L = pytest_utils.generate_stable_random_matrix(comm, (2, 2))
    L = 0.1 * L - 0.2 * np.eye(2)
    owns_ops = ProjectedOWNSCase(L, np.linspace(0.0, 1.0, 6))

    try:
        _, gains, _ = res4py.linalg.resolvent_analysis_owns(
            owns_ops, n_modes=2, max_iter=80, tol=1e-12, verbose=0
        )
        _, singular_values, _ = sp.linalg.svd(
            _exact_resolvent(owns_ops), full_matrices=False
        )
    finally:
        owns_ops.destroy()

    assert gains.shape == (2,)
    assert np.all(gains > 0), f"gains must be positive, got {gains}"
    assert gains[0] >= gains[1], f"gains must be descending, got {gains}"
    error = 100 * np.max(
        np.abs((gains - singular_values[:2]) / singular_values[:2])
    )
    assert error < 5e-1, f"two-mode gain error {error:.2e}%"


def test_owns_returns_consistent_mode_shapes(comm):
    r"""Response and forcing modes must come back shaped
    (n_output, n_x, n_modes) and (n_input, n_x, n_modes)."""
    _, L = pytest_utils.generate_stable_random_matrix(comm, (2, 2))
    L = 0.1 * L - 0.2 * np.eye(2)
    owns_ops = ProjectedOWNSCase(L, np.linspace(0.0, 1.0, 6))
    n_modes = 2

    try:
        station = owns_ops.get_station(0)
        n_input, n_output, n_x = station.n_input, station.n_output, owns_ops.n_x
        response, gains, forcing = res4py.linalg.resolvent_analysis_owns(
            owns_ops, n_modes=n_modes, max_iter=40, tol=1e-10, verbose=0
        )
    finally:
        owns_ops.destroy()

    assert response.shape == (n_output, n_x, n_modes)
    assert forcing.shape == (n_input, n_x, n_modes)
    assert gains.shape == (n_modes,)
    assert np.all(np.isfinite(response))
    assert np.all(np.isfinite(forcing))


def test_owns_is_deterministic(comm):
    r"""The march must be reproducible: identical inputs, identical gains.
    A dependence on uninitialised memory or on an unseeded random start
    would show up here."""
    _, L = pytest_utils.generate_stable_random_matrix(comm, (2, 2))
    L = 0.1 * L - 0.2 * np.eye(2)

    results = []
    for _ in range(2):
        owns_ops = ProjectedOWNSCase(L, np.linspace(0.0, 1.0, 6))
        try:
            _, gains, _ = res4py.linalg.resolvent_analysis_owns(
                owns_ops, n_modes=1, max_iter=60, tol=1e-12, verbose=0
            )
        finally:
            owns_ops.destroy()
        results.append(gains)

    assert np.allclose(results[0], results[1], rtol=0, atol=0), (
        f"non-deterministic gains: {results[0]} vs {results[1]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# The same OWNS problem built on COMM_WORLD
#
# ProjectedOWNSCase above puts every operator on COMM_SELF, so running it
# under mpiexec gives N identical serial solves and the distributed code
# paths inside the pipeline -- matrix_subblock, _ownership_slice, the MUMPS
# factorizations -- never see more than one rank.  The case below is the
# same mathematics on COMM_WORLD, so those paths are exercised for real.
# ──────────────────────────────────────────────────────────────────────────────


def _matrix_for(comm, array):
    r"""Build `array` on `comm`: replicated for COMM_SELF, distributed
    otherwise."""
    if comm == PETSc.COMM_SELF:
        return _petsc_matrix(array)
    return pytest_utils.numpy_to_petsc(comm, np.asarray(array, dtype=complex))


class OWNSCaseOnComm:
    r"""ProjectedOWNSCase generalised over the communicator and the
    characteristic diagonal."""

    def __init__(self, comm, L, x_grid, a_diagonal):
        self.comm = comm
        self.x_grid = x_grid
        self.dx = np.diff(x_grid)
        self.n_x = len(x_grid)

        n = L.shape[0]
        identity = np.eye(n, dtype=complex)
        recursion = res4py.linalg.OWNSRecursionParameters(
            beta_plus=np.asarray([0.25 + 0.10j]),
            beta_minus=np.asarray([-0.30 - 0.15j]),
        )
        station = res4py.linalg.OWNSStation(
            A=_matrix_for(comm, np.diag(np.asarray(a_diagonal, dtype=complex))),
            L=_matrix_for(comm, L),
            input_map=_matrix_for(comm, identity),
            output_map=_matrix_for(comm, identity),
            input_weight=_matrix_for(comm, identity),
            output_weight=_matrix_for(comm, identity),
            recursion=recursion,
        )
        self.stations = tuple(station for _ in range(self.n_x))

    def get_station(self, x_idx):
        return self.stations[x_idx]

    def destroy(self):
        station = self.stations[0]
        for matrix in (
            station.A,
            station.L,
            station.input_map,
            station.output_map,
            station.input_weight,
            station.output_weight,
        ):
            matrix.destroy()


def _owns_gains(case, n_modes=1):
    _, gains, _ = res4py.linalg.resolvent_analysis_owns(
        case, n_modes=n_modes, max_iter=80, tol=1e-12, verbose=0
    )
    return gains


def test_owns_on_comm_world_matches_the_serial_answer(comm):
    r"""The whole OWNS pipeline, run distributed on COMM_WORLD, must produce
    the same gains as the identical problem solved on COMM_SELF.

    This is the only test that drives matrix_subblock / _ownership_slice
    inside the OWNS pipeline with more than one rank in the communicator, so
    a parallel-layout bug there shows up here and nowhere else.  The 4x4
    characteristic diagonal (two downstream, two upstream waves) is chosen so
    the state distributes unevenly at 3 ranks.
    """
    n = 4
    a_diagonal = [2.0, 1.0, -1.0, -2.0]

    rng = np.random.default_rng(31)
    L = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    L = comm.tompi4py().bcast(L, root=0)
    L = 0.1 * L - 0.2 * np.eye(n)
    x_grid = np.linspace(0.0, 1.0, 6)

    serial_case = OWNSCaseOnComm(PETSc.COMM_SELF, L, x_grid, a_diagonal)
    try:
        serial_gains = _owns_gains(serial_case)
    finally:
        serial_case.destroy()

    parallel_case = OWNSCaseOnComm(comm, L, x_grid, a_diagonal)
    try:
        parallel_gains = _owns_gains(parallel_case)
    finally:
        parallel_case.destroy()

    assert parallel_gains.shape == serial_gains.shape
    error = np.max(
        np.abs(parallel_gains - serial_gains) / np.abs(serial_gains)
    )
    assert error < 1e-8, (
        f"COMM_WORLD gains {parallel_gains} disagree with COMM_SELF gains "
        f"{serial_gains}: relative error {error:.2e}"
    )


def test_owns_on_comm_world_agrees_across_ranks(comm):
    r"""Because the marcher replicates its state on every rank rather than
    distributing it, every rank must independently arrive at the same gains.
    A divergence here means the replicated marchers have drifted apart."""
    n = 4
    rng = np.random.default_rng(32)
    L = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    L = comm.tompi4py().bcast(L, root=0)
    L = 0.1 * L - 0.2 * np.eye(n)

    case = OWNSCaseOnComm(comm, L, np.linspace(0.0, 1.0, 6), [2.0, 1.0, -1.0, -2.0])
    try:
        gains = _owns_gains(case, n_modes=2)
    finally:
        case.destroy()

    everyones = comm.tompi4py().allgather(gains)
    for rank, other in enumerate(everyones):
        assert np.allclose(other, everyones[0], rtol=1e-12, atol=0), (
            f"rank {rank} got {other}, rank 0 got {everyones[0]}"
        )
