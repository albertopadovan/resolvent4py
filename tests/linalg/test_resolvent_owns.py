import importlib

import numpy as np
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
