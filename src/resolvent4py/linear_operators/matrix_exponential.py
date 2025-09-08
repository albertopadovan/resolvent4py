import typing
from .linear_operator import LinearOperator
from ..utils.time_stepping import solve_ivp

class MatrixExponentialLinearOperator(LinearOperator):
    r"""
    Class for a linear operator of the form

    .. math::

        L = e^{A t_f}

    where :math:`A` is a resolvent4py Linear operator
    (see :class:`.LinearOperator`).
    The action of this operator on vectors and matrices is computed
    via time-stepping with a second-order accurate explicit stepper.

    :param A: square linear operator
    :type A: LinearOperator
    :param tf: time at which to evaluate :math:`e^{A t_f}`.
    :type tf: float
    :param dt: time step for numerical integration
    :type dt: float
    :param nblocks: number of blocks (if the operator has block structure)
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "MatrixExponentialLinearOperator",
        A: LinearOperator,
        tf: float,
        dt: float,
        method: typing.Optional[str] = "RK2",
        nblocks: typing.Optional[int] = None,
    ) -> None:
        comm = A.get_comm()
        dimensions = A.get_dimensions()
        nrows, ncols = dimensions[0][-1], dimensions[-1][-1]
        if dimensions[0][-1] != dimensions[1][-1]:
            raise ValueError(
                f"A should be a square linear operator. "
                f"Currently nrows = {nrows} and ncols = {ncols}."
            )
        self.tf = tf
        self.nsteps = int(self.tf // dt)
        self.method = method
        self.A = A
        super().__init__(
            comm, "MatrixExponentialLinearOperator", dimensions, nblocks
        )

    def apply(self, x, y=None):
        y = x.duplicate() if y == None else y
        sol = solve_ivp(
            x, self.A.apply, 0.0, self.tf, self.nsteps, self.method
        )
        sol.copy(y)
        sol.destroy()
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = x.duplicate() if y == None else y
        sol = solve_ivp(
            x,
            self.A.apply_hermitian_transpose,
            0.0,
            self.tf,
            self.nsteps,
            self.method,
            adjoint=True,
        )
        sol.copy(y)
        sol.destroy()
        return y

    def apply_mat(self, X, Y=None):
        Y = X.copy() if Y == None else Y
        for j in range(Y.getSizes()[-1]):
            x = X.getColumn(j)
            y = Y.getColumn(j)
            y = self.apply(x, y)
            Y.restoreColumn(j, y)
            X.restoreColumn(j, x)
        return Y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        Y = X.copy() if Y == None else Y
        for j in range(Y.getSizes()[-1]):
            x = X.getColumn(j)
            y = Y.getColumn(j)
            y = self.apply_hermitian_transpose(x, y)
            Y.restoreColumn(j, y)
            X.restoreColumn(j, x)
        return Y

    def destroy(self):
        pass
