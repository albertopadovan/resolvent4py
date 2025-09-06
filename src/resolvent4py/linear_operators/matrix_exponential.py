import typing


from petsc4py import PETSc
from .linear_operator import LinearOperator


def _integrate(
    Aaction: typing.Callable[[PETSc.Vec, PETSc.Vec], PETSc.Vec],
    tf: float,
    dt: float,
    x0: PETSc.Vec,
    x: typing.Optional[PETSc.Vec] = None,
):
    x = x0.duplicate() if x == None else x
    x0.copy(x)
    k1 = x.duplicate()
    k2 = x.duplicate()
    x_temp = x.duplicate()
    # Integrate using Heun's method (2nd order in time)
    for _ in range(int(tf // dt)):
        k1 = Aaction(x, k1) 
        x.copy(x_temp)
        x_temp.axpy(dt, k1)
        k2 = Aaction(x_temp, k2)
        x.axpy(dt/2, k1)
        x.axpy(dt/2, k2)

    vecs = [k1, k2, x_temp]
    for v in vecs:
        v.destroy()

    return x


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
        self.dt = self.tf / (
            self.tf // dt
        )  # Adjust dt so that tf = dt * m where m is an integer
        self.A = A
        super().__init__(
            comm, "MatrixExponentialLinearOperator", dimensions, nblocks
        )

    def apply(self, x, y=None):
        return _integrate(self.A.apply, self.tf, self.dt, x, y)

    def apply_hermitian_transpose(self, x, y=None):
        return _integrate(
            self.A.apply_hermitian_transpose, self.tf, self.dt, x, y
        )

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
