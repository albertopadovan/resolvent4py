import typing
from .linear_operator import LinearOperator
from ..utils.time_stepping import solve_ivp


class PropagatorLinearOperator(LinearOperator):
    r"""
    Class for the propagator-style linear operator

    .. math::

        L\, x_0 \;=\; x(t_f), \qquad
        \frac{d x}{d t} = A(t)\, x, \quad x(t_0) = x_0,

    integrated from :math:`t_0` to :math:`t_f` with an explicit Runge-Kutta
    stepper.  When :math:`A` is time-invariant this collapses to the matrix
    exponential :math:`L = e^{A (t_f - t_0)}`; when :math:`A` is a
    :class:`~resolvent4py.linear_operators.TimePeriodicMatrixLinearOperator`
    (or any operator that overrides :meth:`set_evaluation_time`) the
    integrator forwards the current time at every RK stage and you get
    the (time-dependent) propagator :math:`\Phi(t_f, t_0)` — for
    :math:`t_f - t_0 = T` periodic, this is the monodromy matrix.

    To avoid the expensive base-class probes that would otherwise call
    :meth:`apply` (= a full time integration) on a random vector, the
    real-valued and complex-conjugate-structure flags are inherited
    directly from :code:`A` via overridden :meth:`check_if_real_valued`
    and :meth:`check_if_complex_conjugate_structure` methods.

    :param A: square linear operator (possibly time-dependent)
    :type A: LinearOperator
    :param t0: initial integration time.  Matters only when :code:`A`
        is time-dependent — for a time-periodic :code:`A` the
        propagator depends on the starting phase :math:`t_0`.
    :type t0: float
    :param tf: final integration time
    :type tf: float
    :param dt: time step for numerical integration
    :type dt: float
    :param method: RK integrator passed to
        :func:`~resolvent4py.utils.time_stepping.solve_ivp`
    :type method: Optional[str], default is ``"RK2"``
    :param nblocks: number of blocks (if the operator has block
        structure).  If ``None``, inherit from :code:`A`.
    :type nblocks: Optional[Union[int, None]], default is None
    """

    def __init__(
        self: "PropagatorLinearOperator",
        A: LinearOperator,
        t0: float,
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
        self.A = A
        self.t0 = t0
        self.tf = tf
        self.nsteps = int((self.tf - self.t0) // dt)
        self.method = method
        nblocks = A.get_nblocks() if nblocks is None else nblocks
        super().__init__(
            comm, "PropagatorLinearOperator", dimensions, nblocks
        )

    def check_if_real_valued(self) -> bool:
        r"""Inherit the real-valued flag from :code:`A` — for a real
        :math:`A(t)` the propagator :math:`\Phi(t_f, t_0)` is real."""
        return self.A.get_real_flag()

    def check_if_complex_conjugate_structure(self) -> bool:
        r"""Inherit the block-conjugate-symmetry flag from :code:`A` —
        the harmonic-balanced cc structure (if any) is preserved by
        time integration."""
        return self.A.get_block_cc_flag()

    def apply(self, x, y=None):
        y = x.duplicate() if y == None else y
        sol = solve_ivp(
            x, self.A, self.t0, self.tf, self.nsteps, self.method
        )
        sol.copy(y)
        sol.destroy()
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = x.duplicate() if y == None else y
        sol = solve_ivp(
            x,
            self.A,
            self.t0,
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
