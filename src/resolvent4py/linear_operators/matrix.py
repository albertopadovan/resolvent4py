from .linear_operator import LinearOperator
from ..utils.mat_helpers import mat_solve_hermitian_transpose
from .. import SLEPc
from .. import PETSc
from .. import MPI
from .. import typing


class MatrixLinearOperator(LinearOperator):
    r"""
    Class for a linear operator :math:`L = A`, where :math:`A` is a matrix.
    In general, :code:`A` can be any matrix (rectangular or square,
    invertible or non-invertible). For an invertible :math:`A`, the user
    may also provide a PETSc KSP object :code:`ksp` to act with :math:`A^{-1}`
    on vectors and matrices.

    :param comm: MPI communicator :code:`MPI.COMM_WORLD`
    :type comm: MPI.Comm
    :param A: a PETSc matrix
    :type A: PETSc.Mat
    :param ksp: a PETSc KSP object to enable the :code:`solve()`
        and :code:`solve_hermitian_transpose()` methods
    :type ksp: Optional[Union[PETSc.KSP, None]], defaults to None
    :param nblocks: number of blocks (if the linear operator has block
        structure). This must be an odd number.
    :type nblocks: Optional[Union[int, None]], defaults to None
    """

    def __init__(
        self: "MatrixLinearOperator",
        comm: MPI.Comm,
        A: PETSc.Mat,
        ksp: typing.Optional[typing.Union[PETSc.KSP, None]] = None,
        nblocks: typing.Optional[typing.Union[int, None]] = None,
    ) -> None:
        self.A = A
        self.ksp = ksp
        super().__init__(comm, "MatrixLinearOperator", A.getSizes(), nblocks)

    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        self.A.mult(x, y)
        return y
    
    def hermitian_transpose(self):
        self.A.hermitianTranspose()

    def axpy(self, alpha, X, Y=None):
        Xm = X.A
        if Y != None:
            Ym = Y.A
            Ym.axpy(alpha, Xm)
            Y.restoreMat(Ym)
        else:
            Ym = Xm.duplicate()
            Ym.axpy(alpha, Xm)
            Y = SLEPc.BV().createFromMat(Ym)
            Y.setType('mat')
            Ym.destroy()
        return Y

    def apply_mat(self, X, Y=None):
        Xm = X.getMat()
        if Y != None:
            Ym = Y.getMat()
            Ym = self.A.matMult(Xm, Ym)
            Y.restoreMat(Ym)
        else:
            Ym = Xm.duplicate()
            Ym = self.A.matMult(Xm, Ym)
            Y = SLEPc.BV().createFromMat(Ym)
            Y.setType("mat")
            Ym.destroy()
        X.restoreMat(Xm)
        return Y

    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        self.A.multHermitian(x, y)
        return y

    def apply_hermitian_transpose_mat(self, X, Y=None):
        self.A.hermitianTranspose()
        Xm = X.getMat()
        if Y != None:
            Ym = Y.getMat()
            Ym = self.A.matMult(Xm, Ym)
            Y.restoreMat(Ym)
        else:
            Ym = Xm.duplicate()
            Ym = self.A.matMult(Xm, Ym)
            Y = SLEPc.BV().createFromMat(Ym)
            Y.setType("mat")
            Ym.destroy()
        X.restoreMat(Xm)
        self.A.hermitianTranspose()
        return Y

    def solve(self, x, y=None):
        if self.ksp != None:
            y = self.create_left_vector() if y == None else y
            self.ksp.solve(x, y)
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )

    def solve_mat(self, X, Y=None):
        if self.ksp != None:
            Xm = X.getMat()
            if Y != None:
                Ym = Y.getMat()
                self.ksp.matSolve(Xm, Ym)
                Y.restoreMat(Ym)
            else:
                Ym = Xm.duplicate()
                self.ksp.matSolve(Xm, Ym)
                Y = SLEPc.BV().createFromMat(Ym)
                Y.setType("mat")
                Ym.destroy()
            X.restoreMat(Xm)
            return Y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )

    def solve_hermitian_transpose(self, x, y=None):
        if self.ksp != None:
            y = self.create_right_vector() if y == None else y
            x.conjugate()
            self.ksp.solveTranspose(x, y)
            x.conjugate()
            y.conjugate()
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve_hermitian_transpose(). "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve_hermitian_transpose() method."
            )

    def solve_hermitian_transpose_mat(self, X, Y=None):
        if self.ksp != None:
            Xm = X.getMat()
            if Y != None:
                Ym = Y.getMat()
                Ym = mat_solve_hermitian_transpose(self.ksp, Xm, Ym)
                Y.restoreMat(Ym)
            else:
                Ym = Xm.duplicate()
                Ym = mat_solve_hermitian_transpose(self.ksp, Xm, Ym)
                Y = SLEPc.BV().createFromMat(Ym)
                Y.setType("mat")
                Ym.destroy()
            X.restoreMat(Xm)
            return Y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )

    def destroy_matrix(self: "MatrixLinearOperator"):
        self.A.destroy()

    def destroy_ksp(self: "MatrixLinearOperator"):
        self.ksp.destroy() if self.ksp is not None else None

    def destroy(self):
        self.destroy_matrix()
        self.destroy_ksp()
