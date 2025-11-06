from petsc4py import PETSc
from .linear_operator import LinearOperator


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
    """

    def __init__(self: "PetscPythonLinearOperator", L: LinearOperator) -> None:
        self.L = L

    def mult(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""Compute :math:`y = L x`"""
        self.L.apply(x, y)

    def multHermitian(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""Comput :math:`y = L^* x`"""
        self.L.apply_hermitian_transpose(x, y)

    @classmethod
    def create_shell(
        cls: type["PetscPythonLinearOperator"], L: LinearOperator
    ) -> PETSc.Mat:
        """
        Create a PETSc shell matrix wrapping this operator.

        :param L: `resolvent4py` linear operator
        :type L: LinearOperator

        :rtype: PETSc.Mat of type "python"
        """
        A = PETSc.Mat().create(L.get_comm())
        A.setSizes(L.get_dimensions())
        A.setType("python")
        A.setPythonContext(cls(L))
        A.setUp()
        return A
