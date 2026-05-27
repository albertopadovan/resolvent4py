from petsc4py import PETSc
from .linear_operator import LinearOperator


class _PCContext:
    """PETSc Python shell PC context. Used internally by
    :meth:`PetscPythonLinearOperator.create_shell_pc`."""

    def __init__(self, L: LinearOperator, action=None) -> None:
        self.L = L
        if action is None or action == L.solve:
            self._action = L.solve
            self._action_t = L.solve_hermitian_transpose
        elif action == L.apply:
            self._action = L.apply
            self._action_t = L.apply_hermitian_transpose
        else:
            raise ValueError(
                f"action must be L.apply or L.solve, got {action}."
            )

    def setUp(self, pc: PETSc.PC) -> None:
        pass

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self._action(x, y)

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        # PETSc calls this for P^T, but our _action_t is P^H.
        # P^T x = conj(P^H conj(x))
        x.conjugate()
        self._action_t(x, y)
        y.conjugate()
        x.conjugate()


class PetscPythonLinearOperator:
    r"""
    Class for a `PETSc Python Linear Operator <https://petsc.org/release/petsc4py/petsc_python_types.html>`_.
    This class allows for compatibility between `resolvent4py` linear operators
    and PETSc-wide functionality (e.g., Kyrlov-based solvers for matrix-free
    linear operators).

    .. note::

        Unlike all other linear operators defined within `resolvent4py`,
        this is not a child class of the :class:`.LinearOperator` class

    :param L: linear operator
    :type L: LinearOperator
    :param action: callable used for ``mult``.
        Must be ``L.apply`` or ``L.solve``.  Defaults to ``L.apply``.
    :type action: Optional[Callable]
    """

    def __init__(
        self: "PetscPythonLinearOperator",
        L: LinearOperator,
        action=None,
    ) -> None:
        self.L = L
        if action is None or action == L.apply:
            self._action = L.apply
            self._action_ht = L.apply_hermitian_transpose
        elif action == L.solve:
            self._action = L.solve
            self._action_ht = L.solve_hermitian_transpose
        else:
            raise ValueError(
                f"action must be L.apply or L.solve, got {action}."
            )

    def mult(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""Compute :math:`y = L x`"""
        self._action(x, y)

    def multHermitian(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""Compute :math:`y = L^* x`"""
        self._action_ht(x, y)

    @classmethod
    def create_shell(
        cls: type["PetscPythonLinearOperator"],
        L: LinearOperator,
        action=None,
    ) -> PETSc.Mat:
        """
        Create a PETSc shell matrix wrapping this operator.

        :param L: `resolvent4py` linear operator
        :type L: LinearOperator
        :param action: callable for ``mult``.  Must be ``L.apply``
            or ``L.solve``.  Defaults to ``L.apply``.
        :type action: Optional[Callable]

        :rtype: PETSc.Mat of type "python"
        """
        A = PETSc.Mat().create(L.get_comm())
        A.setSizes(L.get_dimensions())
        A.setType("python")
        A.setPythonContext(cls(L, action))
        A.setUp()
        return A
