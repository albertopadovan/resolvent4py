from __future__ import annotations

__all__ = ["OWNSRecursionParameters", "OWNSStation", "resolvent_analysis_owns"]


import typing
from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc

from ..linear_operators.matrix import MatrixLinearOperator
from ..utils.ksp import create_direct_solver
from ..utils.matrix import add_identity_block, add_matrix_block, create_aij_matrix, left_diagonal_solve, matrix_diagonal_array, matrix_subblock
from ..utils.miscellaneous import petscprint
from ..utils.vector import array_from_petsc_vector


@dataclass(frozen=True)
class OWNSRecursionParameters:
    r"""
    Store OWNS projection recursion parameters.

    :param beta_plus: Downstream recursion parameters
    :type beta_plus: np.ndarray
    :param beta_minus: Upstream recursion parameters
    :type beta_minus: np.ndarray
    """

    beta_plus: np.ndarray
    beta_minus: np.ndarray

    @property
    def order(self) -> int:
        r"""
        Return the recursion order.

        :return: Recursion order
        :rtype: int
        """
        return len(self.beta_plus)


@dataclass(frozen=True)
class OWNSStation:
    r"""
    Store OWNS-P station inputs.

    The station state dimension is ``N``. The input and output dimensions are
    inferred from ``input_map`` and ``output_map``.

    :param A: Characteristic spatial operator, shape ``(N, N)``
    :type A: PETSc.Mat
    :param L: Characteristic state operator in ``A phi_x = L phi + f``, shape ``(N, N)``
    :type L: PETSc.Mat
    :param input_map: Map from external forcing to characteristic forcing, shape ``(N, n_input)``
    :type input_map: PETSc.Mat
    :param output_map: Map from characteristic response to external output, shape ``(n_output, N)``
    :type output_map: PETSc.Mat
    :param input_weight: External forcing inner-product weight, shape ``(n_input, n_input)``
    :type input_weight: PETSc.Mat
    :param output_weight: External output inner-product weight, shape ``(n_output, n_output)``
    :type output_weight: PETSc.Mat
    :param recursion: OWNS recursion parameters
    :type recursion: OWNSRecursionParameters
    :param sort_indices: Optional zero-based characteristic sorting indices
    :type sort_indices: np.ndarray | None
    :param n_points: Optional transverse grid point count
    :type n_points: int | None
    """

    A: PETSc.Mat
    L: PETSc.Mat
    input_map: PETSc.Mat
    output_map: PETSc.Mat
    input_weight: PETSc.Mat
    output_weight: PETSc.Mat
    recursion: OWNSRecursionParameters
    sort_indices: np.ndarray | None = None
    n_points: int | None = None

    @property
    def n_state(self) -> int:
        r"""
        Return the characteristic state dimension.

        :return: State dimension
        :rtype: int
        """
        return self.A.getSize()[0]

    @property
    def n_input(self) -> int:
        r"""
        Return the external forcing dimension.

        :return: Input dimension
        :rtype: int
        """
        return self.input_map.getSize()[1]

    @property
    def n_output(self) -> int:
        r"""
        Return the external output dimension.

        :return: Output dimension
        :rtype: int
        """
        return self.output_map.getSize()[0]


class OWNSOperator(typing.Protocol):
    r"""
    Defines a station-wise OWNS provider.

    Users supply an object with these attributes and a :meth:`get_station`
    method that returns an :class:`OWNSStation` for each streamwise station.

    :ivar comm: PETSc communicator
    :vartype comm: PETSc.Comm
    :ivar n_x: Number of streamwise stations
    :vartype n_x: int
    :ivar dx: Streamwise spacing
    :vartype dx: np.ndarray
    """

    comm: PETSc.Comm
    n_x: int
    dx: np.ndarray

    def get_station(self, x_idx: int) -> OWNSStation:
        r"""
        Return one OWNS station.

        :param x_idx: Station index
        :type x_idx: int

        :return: Station data
        :rtype: OWNSStation
        """


@dataclass(frozen=True)
class _Split:
    sorted: np.ndarray
    n_plus: int
    n_minus: int
    n_zero: int

    @property
    def n_pm(self) -> int:
        return self.n_plus + self.n_minus

    @property
    def n(self) -> int:
        return self.n_pm + self.n_zero


@dataclass
class _StationSystem:
    split: _Split
    dx: float
    forward: MatrixLinearOperator
    Apm: MatrixLinearOperator
    input: MatrixLinearOperator
    response: MatrixLinearOperator
    P1_full: MatrixLinearOperator
    P2: MatrixLinearOperator
    P3: MatrixLinearOperator
    P3z: MatrixLinearOperator | None
    input_weight: MatrixLinearOperator
    output_weight: MatrixLinearOperator

    def destroy(self) -> None:
        # input_weight/output_weight wrap the caller's station matrices, so
        # only the KSP and internal transpose are ours to free.  Everything
        # else was built here and is released in full.
        _destroy_borrowed_operator(self.input_weight)
        _destroy_borrowed_operator(self.output_weight)
        _destroy_owned_operator(self.P3z)
        _destroy_owned_operator(self.P3)
        _destroy_owned_operator(self.P2)
        _destroy_owned_operator(self.P1_full)
        _destroy_owned_operator(self.response)
        _destroy_owned_operator(self.input)
        _destroy_owned_operator(self.Apm)
        _destroy_owned_operator(self.forward)


class _Marcher:
    def __init__(self, operators: OWNSOperator, verbose: typing.Optional[int] = 0) -> None:
        self.operators = operators
        self.verbose = 0 if verbose is None else verbose
        first = operators.get_station(0)
        self.n_input = first.n_input
        self.n_output = first.n_output

    def run(
        self,
        n_modes: int,
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        forcing = self.normalize(rng.standard_normal((self.n_input, self.operators.n_x, n_modes)) + 1j * rng.standard_normal((self.n_input, self.operators.n_x, n_modes)))
        gains = np.zeros(n_modes)
        previous_gains = np.zeros(n_modes)

        for iteration in range(1, max_iter + 1):
            if self.verbose > 0:
                petscprint(self.operators.comm, "Loop %d/%d, OWNS march (forward action)" % (iteration, max_iter))

            response = self.direct(forcing, iteration, max_iter)
            output_energy, input_energy = self.energies(response, forcing)
            gains = np.sqrt(output_energy / np.maximum(input_energy, 1e-300))

            if self.verbose > 0:
                for mode, gain in enumerate(gains, start=1):
                    petscprint(self.operators.comm, "Gain for mode %d = %1.15e" % (mode, gain))

            if iteration > 1:
                change = np.max(np.abs(gains - previous_gains) / np.maximum(previous_gains, 1e-14))

                if self.verbose > 0:
                    petscprint(self.operators.comm, "Maximum relative gain change = %1.15e" % change)

                if change < tol:
                    break

            previous_gains = gains.copy()
            if iteration < max_iter:
                if self.verbose > 0:
                    petscprint(self.operators.comm, "Loop %d/%d, OWNS march (adjoint action)" % (iteration, max_iter))

                forcing = self.normalize(self.adjoint(response, iteration, max_iter))

        return response, gains, forcing

    def direct(
        self,
        forcing: np.ndarray,
        iteration: int,
        max_iter: int,
    ) -> np.ndarray:
        n_modes = forcing.shape[2]
        response = np.zeros((self.n_output, self.operators.n_x, n_modes), dtype=complex)
        states: list[np.ndarray | None] = [None] * self.operators.n_x

        for index in range(1, self.operators.n_x):
            if self.verbose > 1:
                petscprint(self.operators.comm, "Loop %d/%d, station %d/%d (forward action)" % (iteration, max_iter, index, self.operators.n_x - 1))
            station = self.operators.get_station(index)
            station_system = _build_station_system(station, self.operators.dx[index - 1], _bdf_coefficients(index)[0])

            try:
                g = _apply_array(station_system.input, forcing[:, index, :])
                rhs = _direct_rhs(station_system, states, g, index)
                raw_state = _solve_array(station_system.forward, rhs)
                state = _project_state(station_system, raw_state)
                states[index] = np.zeros_like(state)
                states[index][station_system.split.sorted, :] = state
                response[:, index, :] = _apply_array(station_system.response, state)
            finally:
                station_system.destroy()

        return response

    def adjoint(
        self,
        response: np.ndarray,
        iteration: int,
        max_iter: int,
    ) -> np.ndarray:
        n_modes = response.shape[2]
        state_adjoint: list[np.ndarray | None] = [None] * self.operators.n_x
        forcing_adjoint = np.zeros((self.n_input, self.operators.n_x, n_modes), dtype=complex)
        forcing = np.zeros_like(forcing_adjoint)

        for index in range(self.operators.n_x - 1, 0, -1):
            if self.verbose > 1:
                petscprint(self.operators.comm, "Loop %d/%d, station %d/%d (adjoint action)" % (iteration, max_iter, index, self.operators.n_x - 1))
            station = self.operators.get_station(index)
            station_system = _build_station_system(station, self.operators.dx[index - 1], _bdf_coefficients(index)[0])

            try:
                state_rhs = _adjoint_response_source(station_system, response[:, index, :])

                if state_adjoint[index] is not None:
                    state_rhs += state_adjoint[index][station_system.split.sorted, :]

                raw_rhs = _project_state_adjoint(station_system, state_rhs)
                rhs_adjoint = _solve_array(station_system.forward, raw_rhs, adjoint=True)

                g_adjoint = _direct_rhs_adjoint(station_system, state_adjoint, rhs_adjoint, index)
                forcing_adjoint[:, index, :] = _apply_array(station_system.input, g_adjoint, adjoint=True)
                forcing[:, index, :] = _solve_array(station_system.input_weight, forcing_adjoint[:, index, :])
            finally:
                station_system.destroy()

        return forcing

    def normalize(self, modes: np.ndarray) -> np.ndarray:
        out = modes.copy()
        for mode in range(out.shape[2]):
            for previous in range(mode):
                out[:, :, mode] -= self._weighted_inner(out[:, :, previous], out[:, :, mode], "input_weight") * out[:, :, previous]

            norm = np.sqrt(max(np.real(self._weighted_inner(out[:, :, mode], out[:, :, mode], "input_weight")), 0.0))
            if norm <= 1e-14:
                raise ValueError("OWNS forcing mode has zero norm.")

            out[:, :, mode] /= norm
        return out

    def _weighted_inner(
        self,
        left: np.ndarray,
        right: np.ndarray,
        weight_name: str,
    ) -> complex:
        value = 0.0 + 0.0j
        for index in range(self.operators.n_x):
            station = self.operators.get_station(index)
            weight = _matrix_operator(getattr(station, weight_name))
            value += np.vdot(left[:, index], _apply_array(weight, right[:, index]))
        return value

    def energies(
        self,
        response: np.ndarray,
        forcing: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        output_energy = np.zeros(response.shape[2])
        input_energy = np.zeros(forcing.shape[2])
        for mode in range(response.shape[2]):
            output_energy[mode] = np.real(self._weighted_inner(response[:, :, mode], response[:, :, mode], "output_weight"))
            input_energy[mode] = np.real(self._weighted_inner(forcing[:, :, mode], forcing[:, :, mode], "input_weight"))
        return output_energy, input_energy


def _matrix_operator(matrix: PETSc.Mat, solve: bool = False) -> MatrixLinearOperator:
    ksp = create_direct_solver(matrix) if solve else None
    return MatrixLinearOperator(matrix, ksp, real_valued=False)


# As of v2.0, MatrixLinearOperator.destroy() frees only the Hermitian
# transpose it builds internally: the matrix and the KSP handed to
# __init__ belong to the caller.  Every operator here is built by
# _matrix_operator(), so this module is that caller and must release the
# KSP itself -- and, where it also created the matrix, the matrix.


def _destroy_owned_operator(operator: MatrixLinearOperator | None) -> None:
    r"""Tear down an operator whose matrix this module created."""
    if operator is None:
        return
    matrix = operator.A
    ksp = operator.ksp
    operator.destroy()
    if ksp is not None:
        ksp.destroy()
    matrix.destroy()


def _destroy_borrowed_operator(operator: MatrixLinearOperator | None) -> None:
    r"""Tear down an operator wrapping a matrix owned by someone else: free
    the KSP created here and the internal transpose, leave the matrix."""
    if operator is None:
        return
    ksp = operator.ksp
    operator.destroy()
    if ksp is not None:
        ksp.destroy()


def _set_vector(vector: PETSc.Vec, values: np.ndarray) -> None:
    start, end = vector.getOwnershipRange()
    vector.setValues(np.arange(start, end, dtype=PETSc.IntType), values[start:end], addv=PETSc.InsertMode.INSERT_VALUES)
    vector.assemble()


def _apply_array(operator: MatrixLinearOperator, values: np.ndarray, adjoint: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    values = values.reshape(-1, 1) if values.ndim == 1 else values
    out = []
    x = operator.create_left_vector() if adjoint else operator.create_right_vector()
    try:
        for column in range(values.shape[1]):
            _set_vector(x, values[:, column])
            y = operator.apply_hermitian_transpose(x) if adjoint else operator.apply(x)
            try:
                out.append(array_from_petsc_vector(y))
            finally:
                y.destroy()
    finally:
        x.destroy()
    return np.column_stack(out)


def _solve_array(operator: MatrixLinearOperator, values: np.ndarray, adjoint: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    values = values.reshape(-1, 1) if values.ndim == 1 else values
    out = []
    x = operator.create_right_vector() if adjoint else operator.create_left_vector()
    try:
        for column in range(values.shape[1]):
            _set_vector(x, values[:, column])
            y = operator.solve_hermitian_transpose(x) if adjoint else operator.solve(x)
            try:
                solved = array_from_petsc_vector(y)
                if operator.ksp is not None and (operator.ksp.getConvergedReason() < 0 or not np.all(np.isfinite(solved))):
                    raise RuntimeError("PETSc solve failed in OWNS with KSP reason %s and PC reason %s." % (operator.ksp.getConvergedReason(), operator.ksp.getPC().getFailedReason()))
                out.append(solved)
            finally:
                y.destroy()
    finally:
        x.destroy()
    return np.column_stack(out)


def _solve_matrix(operator: MatrixLinearOperator, rhs: PETSc.Mat) -> PETSc.Mat:
    rhs_dense = rhs.copy()
    rhs_dense.convert(PETSc.Mat.Type.DENSE)
    nrows, ncols = rhs.getSize()
    out = create_aij_matrix(rhs.getComm(), nrows, ncols)
    try:
        for column in range(ncols):
            b = rhs_dense.getColumnVector(column)
            y = operator.solve(b)
            try:
                start, end = y.getOwnershipRange()
                rows = np.arange(start, end, dtype=PETSc.IntType)
                out.setValues(rows, np.asarray([column], dtype=PETSc.IntType), y.getArray(readonly=True).reshape(-1, 1))
            finally:
                y.destroy()
                b.destroy()
    finally:
        rhs_dense.destroy()
    out.assemble()
    return out


def _build_station_system(
    station: OWNSStation,
    dx: float,
    lhs_coefficient: float,
) -> _StationSystem:
    split = _characteristic_split(station)
    temporary: list[PETSc.Mat] = []

    try:
        A_sort = matrix_subblock(station.A, split.sorted, split.sorted)
        L_sort = matrix_subblock(station.L, split.sorted, split.sorted)
        input_sort = matrix_subblock(station.input_map, split.sorted, np.arange(station.n_input, dtype=PETSc.IntType))
        output_sort = matrix_subblock(station.output_map, np.arange(station.n_output, dtype=PETSc.IntType), split.sorted)
        temporary.extend([A_sort, L_sort, input_sort, output_sort])

        A_normalized, L_normalized, normalization_diagonal = _normalize_characteristic_operators(A_sort, L_sort, split)
        input_normalized = left_diagonal_solve(normalization_diagonal, input_sort)
        projection = _build_projection(A_normalized, L_normalized, split, station.recursion)
        temporary.extend([A_normalized, L_normalized])

        pm = np.arange(0, split.n_pm, dtype=PETSc.IntType)
        Apm = matrix_subblock(A_normalized, pm, pm)
        forward = _forward_matrix(Apm, L_normalized, split, dx, lhs_coefficient)
        response = output_sort.copy()
        response.assemble()
    finally:
        for item in reversed(temporary):
            item.destroy()

    return _StationSystem(split=split, dx=dx, forward=_matrix_operator(forward, solve=True), Apm=_matrix_operator(Apm), input=_matrix_operator(input_normalized), response=_matrix_operator(response), P1_full=projection[0], P2=projection[1], P3=projection[2], P3z=projection[3], input_weight=_matrix_operator(station.input_weight, solve=True), output_weight=_matrix_operator(station.output_weight))


def _characteristic_split(station: OWNSStation) -> _Split:
    diagonal = matrix_diagonal_array(station.A)
    max_diagonal = np.max(np.abs(diagonal))
    tolerance = max(1e-5, 1e-7 * max_diagonal)

    if np.max(np.abs(np.imag(diagonal))) > 1e-10:
        raise ValueError("OWNS characteristic spatial operator must have real diagonal.")

    if station.sort_indices is None:
        order = _natural_characteristic_order(station.n_state, station.n_points)
        plus = [idx for idx in order if np.real(diagonal[idx]) > tolerance]
        minus = [idx for idx in order if np.real(diagonal[idx]) < -tolerance]
        zero = [idx for idx in order if abs(diagonal[idx]) <= tolerance]
        sorted_indices = np.asarray(plus + minus + zero, dtype=PETSc.IntType)
    else:
        sorted_indices = np.asarray(station.sort_indices, dtype=PETSc.IntType).reshape(-1)
        sorted_diagonal = diagonal[sorted_indices]
        plus = np.flatnonzero(np.real(sorted_diagonal) > tolerance)
        minus = np.flatnonzero(np.real(sorted_diagonal) < -tolerance)
        zero = np.flatnonzero(np.abs(sorted_diagonal) <= tolerance)
        sorted_indices = sorted_indices[np.concatenate([plus, minus, zero])]

    sorted_diagonal = diagonal[sorted_indices]
    n_plus = int(np.sum(np.real(sorted_diagonal) > tolerance))
    n_minus = int(np.sum(np.real(sorted_diagonal) < -tolerance))
    n_zero = int(len(sorted_diagonal) - n_plus - n_minus)

    if n_plus == 0 or n_minus == 0:
        raise ValueError("OWNS characteristic split must include plus and minus waves.")

    return _Split(sorted_indices, n_plus, n_minus, n_zero)


def _natural_characteristic_order(n_state: int, n_points: int | None) -> list[int]:
    if n_points is None:
        return list(range(n_state))

    if n_points <= 0 or n_state % n_points != 0:
        raise ValueError("n_points must divide the station state dimension.")

    n_variables = n_state // n_points
    return [variable * n_points + point for variable in range(n_variables) for point in range(n_points)]


def _normalize_characteristic_operators(
    A_sort: PETSc.Mat,
    L_sort: PETSc.Mat,
    split: _Split,
) -> tuple[PETSc.Mat, PETSc.Mat, np.ndarray]:
    diagonal = matrix_diagonal_array(A_sort)

    if np.any(np.abs(diagonal[: split.n_pm]) == 0):
        raise ValueError("OWNS A-normalization has a zero diagonal entry.")

    diagonal[split.n_pm :] = 1.0
    A_normalized = left_diagonal_solve(diagonal, A_sort)
    L_normalized = left_diagonal_solve(diagonal, L_sort)
    return A_normalized, L_normalized, diagonal


def _build_projection(
    A_sort: PETSc.Mat,
    L_sort: PETSc.Mat,
    split: _Split,
    recursion: OWNSRecursionParameters,
) -> tuple[MatrixLinearOperator, MatrixLinearOperator, MatrixLinearOperator, MatrixLinearOperator | None]:
    comm = A_sort.getComm()
    n = split.n
    npm = split.n_pm
    n_zero = split.n_zero
    nb = recursion.order
    nv = 2 * nb * n + n_zero
    i_ref = nb * n
    j_ref = (nb - 1) * n + n_zero + split.n_minus
    all_idx = np.arange(n, dtype=PETSc.IntType)
    zero = np.arange(npm, n, dtype=PETSc.IntType)
    minus_zero = np.arange(split.n_plus, n, dtype=PETSc.IntType)
    plus_zero = np.concatenate([np.arange(0, split.n_plus, dtype=PETSc.IntType), zero])

    S = _algebraic_completion(L_sort, split)
    P1 = create_aij_matrix(comm, nv, npm)
    P2 = create_aij_matrix(comm, nv, nv)
    P3 = create_aij_matrix(comm, npm, nv)
    P3z = create_aij_matrix(comm, n_zero, nv) if n_zero > 0 else None

    try:
        Qm0 = _q_matrix(A_sort, L_sort, recursion.beta_minus[0])
        Qm0S = Qm0.matMult(S)

        try:
            add_matrix_block(P1, i_ref - n, 0, 1.0, Qm0S)
        finally:
            Qm0S.destroy()
            Qm0.destroy()

        for index in range(nb):
            Qp = _q_matrix(A_sort, L_sort, recursion.beta_plus[index])
            Qm = _q_matrix(A_sort, L_sort, recursion.beta_minus[index])

            try:
                if index < nb - 1:
                    add_matrix_block(P2, i_ref - (index + 1) * n, j_ref - (index + 1) * n, -1.0, Qp)
                    add_matrix_block(P2, i_ref - (index + 1) * n, j_ref - index * n, 1.0, Qm)
                    add_matrix_block(P2, i_ref + index * n + n_zero, j_ref + index * n, -1.0, Qp)
                    add_matrix_block(P2, i_ref + index * n + n_zero, j_ref + (index + 1) * n, 1.0, Qm)
                else:
                    Qp_mz = matrix_subblock(Qp, all_idx, minus_zero)
                    Qm_pz = matrix_subblock(Qm, all_idx, plus_zero)

                    try:
                        add_matrix_block(P2, 0, 0, -1.0, Qp_mz)
                        add_matrix_block(P2, 0, len(minus_zero), 1.0, Qm)
                        add_matrix_block(P2, i_ref + index * n + n_zero, j_ref + index * n, -1.0, Qp)
                        add_matrix_block(P2, i_ref + index * n + n_zero, j_ref + index * n + n, 1.0, Qm_pz)
                    finally:
                        Qm_pz.destroy()
                        Qp_mz.destroy()
            finally:
                Qm.destroy()
                Qp.destroy()

        if n_zero > 0:
            Bz = matrix_subblock(L_sort, zero, all_idx)
            try:
                add_matrix_block(P2, i_ref, j_ref, 1.0, Bz)
            finally:
                Bz.destroy()

        add_identity_block(P3, 0, j_ref, npm)
        if P3z is not None:
            add_identity_block(P3z, 0, j_ref + npm, n_zero)

        P1.assemble()
        P2.assemble()
        P3.assemble()
        if P3z is not None:
            P3z.assemble()

        P1_full = create_aij_matrix(comm, nv, n)
        add_matrix_block(P1_full, 0, 0, 1.0, P1)
        P1_full.assemble()
    finally:
        if P3z is not None:
            P3z.assemble()
        P3.assemble()
        P2.assemble()
        P1.assemble()
        P1.destroy()
        S.destroy()

    return _matrix_operator(P1_full), _matrix_operator(P2, solve=True), _matrix_operator(P3), _matrix_operator(P3z) if P3z is not None else None


def _algebraic_completion(L_sort: PETSc.Mat, split: _Split) -> PETSc.Mat:
    comm = L_sort.getComm()

    S = create_aij_matrix(comm, split.n, split.n_pm)
    add_identity_block(S, 0, 0, split.n_pm)

    if split.n_zero > 0:
        pm = np.arange(0, split.n_pm, dtype=PETSc.IntType)
        zero = np.arange(split.n_pm, split.n, dtype=PETSc.IntType)
        Lzz = matrix_subblock(L_sort, zero, zero)
        Lzpm = matrix_subblock(L_sort, zero, pm)

        try:
            Lzz_operator = _matrix_operator(Lzz, solve=True)
            try:
                zpm = _solve_matrix(Lzz_operator, Lzpm)
            finally:
                _destroy_borrowed_operator(Lzz_operator)

            try:
                add_matrix_block(S, split.n_pm, 0, -1.0, zpm)
            finally:
                zpm.destroy()
        finally:
            Lzpm.destroy()
            Lzz.destroy()

    S.assemble()
    return S


def _q_matrix(A_sort: PETSc.Mat, L_sort: PETSc.Mat, beta: complex) -> PETSc.Mat:
    Q = L_sort.copy()
    Q.axpy(-1j * beta, A_sort)
    Q.assemble()
    return Q


def _forward_matrix(
    Apm: PETSc.Mat,
    L_sort: PETSc.Mat,
    split: _Split,
    dx: float,
    lhs_coefficient: float,
) -> PETSc.Mat:
    comm = L_sort.getComm()
    n = split.n
    npm = split.n_pm
    pm = np.arange(0, npm, dtype=PETSc.IntType)
    zero = np.arange(npm, n, dtype=PETSc.IntType)

    Lpm_pm = matrix_subblock(L_sort, pm, pm)
    system = create_aij_matrix(comm, n, n)
    temporary = [Lpm_pm]

    try:
        add_matrix_block(system, 0, 0, lhs_coefficient, Apm)
        add_matrix_block(system, 0, 0, -dx, Lpm_pm)

        if split.n_zero > 0:
            Lpm_z = matrix_subblock(L_sort, pm, zero)
            Lz_pm = matrix_subblock(L_sort, zero, pm)
            Lz_z = matrix_subblock(L_sort, zero, zero)
            temporary.extend([Lpm_z, Lz_pm, Lz_z])

            add_matrix_block(system, 0, npm, -dx, Lpm_z)
            add_matrix_block(system, npm, 0, 1.0, Lz_pm)
            add_matrix_block(system, npm, npm, 1.0, Lz_z)
        system.assemble()
    finally:
        for matrix in reversed(temporary):
            matrix.destroy()

    return system


def _direct_rhs(
    station_system: _StationSystem,
    states: list[np.ndarray | None],
    forcing: np.ndarray,
    index: int,
) -> np.ndarray:
    split = station_system.split
    rhs = np.zeros((split.n, forcing.shape[1]), dtype=complex)

    coefficients = _bdf_coefficients(index)
    for offset, coefficient in enumerate(coefficients[1:], start=1):
        previous = states[index - offset]
        if previous is None:
            continue

        previous_sorted = previous[split.sorted, :]
        rhs[: split.n_pm, :] -= coefficient * _apply_array(station_system.Apm, previous_sorted[: split.n_pm, :])

    rhs[: split.n_pm, :] += station_system.dx * forcing[: split.n_pm, :]
    if split.n_zero > 0:
        rhs[split.n_pm :, :] += forcing[split.n_pm :, :]

    return rhs


def _project_state(
    station_system: _StationSystem,
    state: np.ndarray,
) -> np.ndarray:
    split = station_system.split

    aux_rhs = _apply_array(station_system.P1_full, state)
    aux = _solve_array(station_system.P2, aux_rhs)

    projected = np.zeros_like(state)
    projected[: split.n_pm, :] = _apply_array(station_system.P3, aux)
    if split.n_zero > 0 and station_system.P3z is not None:
        projected[split.n_pm :, :] = _apply_array(station_system.P3z, aux)

    return projected


def _project_state_adjoint(
    station_system: _StationSystem,
    state: np.ndarray,
) -> np.ndarray:
    split = station_system.split

    aux_rhs = _apply_array(station_system.P3, state[: split.n_pm, :], adjoint=True)

    if split.n_zero > 0 and station_system.P3z is not None:
        aux_rhs += _apply_array(station_system.P3z, state[split.n_pm :, :], adjoint=True)

    aux = _solve_array(station_system.P2, aux_rhs, adjoint=True)
    return _apply_array(station_system.P1_full, aux, adjoint=True)


def _direct_rhs_adjoint(
    station_system: _StationSystem,
    states: list[np.ndarray | None],
    rhs_adjoint: np.ndarray,
    index: int,
) -> np.ndarray:
    split = station_system.split
    forcing_adjoint = np.zeros_like(rhs_adjoint)
    forcing_adjoint[: split.n_pm, :] += station_system.dx * rhs_adjoint[: split.n_pm, :]

    if split.n_zero > 0:
        forcing_adjoint[split.n_pm :, :] += rhs_adjoint[split.n_pm :, :]

    coefficients = _bdf_coefficients(index)
    for offset, coefficient in enumerate(coefficients[1:], start=1):
        previous = index - offset
        if previous < 0:
            continue

        state_sorted = np.zeros_like(rhs_adjoint)
        state_sorted[: split.n_pm, :] -= coefficient * _apply_array(station_system.Apm, rhs_adjoint[: split.n_pm, :], adjoint=True)

        contribution = np.zeros_like(state_sorted)
        contribution[split.sorted, :] = state_sorted
        if states[previous] is None:
            states[previous] = contribution
        else:
            states[previous] += contribution

    return forcing_adjoint


def _adjoint_response_source(
    station_system: _StationSystem,
    response: np.ndarray,
) -> np.ndarray:
    weighted = _apply_array(station_system.output_weight, response)
    return _apply_array(station_system.response, weighted, adjoint=True)


def _bdf_coefficients(index: int) -> tuple[float, ...]:
    return (1.0, -1.0) if index == 1 else (1.5, -2.0, 0.5)


def resolvent_analysis_owns(
    owns_ops: OWNSOperator,
    n_modes: int,
    max_iter: int,
    tol: float = 1e-4,
    verbose: typing.Optional[int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute resolvent modes with the projected one-way Navier-Stokes
    (OWNS-P) equations.

    The supplied :code:`owns_ops` object must provide ``comm``, ``n_x``,
    ``dx``, and ``get_station(j)``. Each station must provide characteristic
    operators, forcing and response maps, weights, and recursion parameters
    through :class:`OWNSStation`. The leading singular values are computed by
    alternating direct and adjoint OWNS-P marches.

    :param owns_ops: OWNS operator provider
    :type owns_ops: OWNSOperator
    :param n_modes: Number of modes
    :type n_modes: int
    :param max_iter: Maximum iteration count
    :type max_iter: int
    :param tol: Relative gain-change tolerance
    :type tol: float
    :param verbose: defines verbosity of output to terminal. = 0 no printout to terminal, = 1 monitor OWNS iterations, = 2 monitor OWNS iterations and station marches
    :type verbose: Optional[int], default is 0

    :return: Response modes with shape ``(n_output, n_x, n_modes)``, gains
        with shape ``(n_modes,)``, and forcing modes with shape
        ``(n_input, n_x, n_modes)``
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    return _Marcher(owns_ops, verbose=verbose).run(n_modes, max_iter, tol)
