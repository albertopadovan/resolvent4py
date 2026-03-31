import abc
from typing import Optional, Tuple

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np


class DifferentialEquation(metaclass=abc.ABCMeta):
    r"""
    Abstract base class for dynamical systems of the form

    .. math::

        \dot{q} = A\,q + B(q, q),

    where :math:`A` is a linear operator and :math:`B` is a symmetric
    bilinear form representing the quadratic nonlinearity.

    Subclasses must implement :meth:`evaluate_linear_term`,
    :meth:`evaluate_quadratic_term`, and :meth:`solve_linear_system`.
    """

    def __init__(
        self, comm: PETSc.Comm, name: str,
        state_dim: Tuple[int, int], poly_deg: int,
    ) -> None:
        r"""
        :param comm: MPI communicator
        :type comm: PETSc.Comm
        :param name: identifier for this system
        :type name: str
        :param state_dim: state-space dimensions as
            ``(local_size, global_size)``
        :type state_dim: Tuple[int, int]
        :param poly_deg: degree of the polynomial nonlinearity
        :type poly_deg: int
        """
        self._comm = comm
        self._name = name
        self._state_dim = state_dim
        self._poly_deg = poly_deg

    def get_name(self) -> str:
        r"""Return the system name."""
        return self._name

    def get_comm(self) -> PETSc.Comm:
        r"""Return the MPI communicator."""
        return self._comm

    def get_state_dimension(self) -> Tuple[int, int]:
        r"""Return ``(local_size, global_size)`` of the state space."""
        return self._state_dim

    def get_poly_degree(self) -> int:
        r"""Return the degree of the polynomial nonlinearity."""
        return self._poly_deg
    

    @abc.abstractmethod
    def evaluate_linear_term(
        self, q: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the linear part of the RHS: :math:`A q`.

        :param q: state vector
        :type q: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`A q`
        :rtype: PETSc.Vec
        """
        ...

    @abc.abstractmethod
    def evaluate_quadratic_term(
        self, q1: PETSc.Vec, q2: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the quadratic bilinear term :math:`B(q_1, q_2)`.
        Must satisfy :math:`B(q_1, q_2) = B(q_2, q_1)`.

        :param q1: first state vector
        :type q1: PETSc.Vec
        :param q2: second state vector
        :type q2: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`B(q_1, q_2)`
        :rtype: PETSc.Vec
        """
        ...

    @abc.abstractmethod
    def solve_linear_system(
        self, s: complex, b: PETSc.Vec, x: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Solve :math:`(s I - A)\,x = b`.

        :param s: shift parameter
        :type s: complex
        :param b: right-hand side vector
        :type b: PETSc.Vec
        :param x: optional output vector (reused if provided)
        :type x: Optional[PETSc.Vec]

        :return: solution :math:`x`
        :rtype: PETSc.Vec
        """
        ...

    def evaluate_dynamics(
        self, t: float, q: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the full RHS: :math:`A q + B(q, q)`.
        Override if the system has additional terms.

        :param t: time
        :type t: float
        :param q: state vector
        :type q: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`A q + B(q, q)`
        :rtype: PETSc.Vec
        """
        y = self.evaluate_linear_term(q, y)
        Bqq = self.evaluate_quadratic_term(q, q)
        y.axpy(1.0, Bqq)
        Bqq.destroy()
        return y
