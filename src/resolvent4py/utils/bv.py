__all__ = [
    "bv_add",
    "bv_conj",
    "bv_real",
    "bv_imag",
    "bv_slice",
    "reshape_bv_into_harmonic_balanced_vector",
]

import typing

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

from .matrix import convert_coo_to_csr


def bv_add(alpha: float, X: SLEPc.BV, Y: SLEPc.BV) -> None:
    r"""
    Compute in-place addition :math:`X \leftarrow X + \alpha Y`

    :type alpha: float
    :type X: SLEPc.BV
    :type Y: SLEPc.BV

    :rtype: SLEPc.BV
    """
    Xm = X.getMat()
    Ym = Y.getMat()
    Xm.axpy(alpha, Ym)
    X.restoreMat(Xm)
    Y.restoreMat(Ym)
    return X


def bv_conj(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Returns the complex conjugate :math:`\overline{X}` of the BV structure

    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.conjugate()
    Y.restoreMat(Ym)
    return Y


def bv_real(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Returns the real part :math:`\text{Re}(X)` of the BV structure

    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.realPart()
    Y.restoreMat(Ym)
    return Y


def bv_imag(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Returns the imaginary part :math:`\text{Im}(X)` of the BV structure

    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.imagPart()
    Y.restoreMat(Ym)
    return Y


def bv_slice(
    X: SLEPc.BV,
    columns: np.array,
    Y: typing.Optional[SLEPc.BV] = None,
):
    r"""
    Extract a subset of columns from X and store into Y

    :type X: SLEPc.BV
    :param columns: array of columns to extract
    :type columns: np.array
    :type Y: Optional[SLEPc.BV], default is None

    :rtype: SLEPc.BV
    """
    if Y == None:
        Y = SLEPc.BV().create(X.getComm())
        Y.setSizes(X.getSizes()[0], len(columns))
        Y.setType("mat")
    Q = np.zeros((X.getSizes()[-1], len(columns)))
    for i in range(len(columns)):
        Q[columns[i], i] = 1.0
    Q = PETSc.Mat().createDense(Q.shape, None, Q, PETSc.COMM_SELF)
    Y.mult(1.0, 0.0, X, Q)
    Q.destroy()
    return Y


def reshape_bv_into_harmonic_balanced_vector(
    bv: SLEPc.BV,
    vec: typing.Optional[PETSc.Vec] = None,
) -> PETSc.Vec:
    r"""
    Reshape an :math:`n \times (2m+1)` SLEPc BV into a harmonic-balanced
    vector of size :math:`n \cdot (2m+1)`.

    This is the inverse of
    :func:`~resolvent4py.utils.vector.reshape_harmonic_balanced_vector_into_bv`.
    Column :math:`j` of the BV becomes the :math:`j`-th block of size
    :math:`n` in the output vector, i.e.
    ``vec[j*n + i] = bvMat[i, j]``.

    :param bv: SLEPc BV of shape :math:`n \times (2m+1)`
    :type bv: SLEPc.BV
    :param vec: optional pre-allocated PETSc vector of size
        :math:`n \cdot (2m+1)`.  If ``None``, a new vector is created.
    :type vec: Optional[PETSc.Vec]

    :return: the reshaped harmonic-balanced vector
    :rtype: PETSc.Vec
    """
    comm = bv.getComm()
    rowsize, nblocks = bv.getSizes()
    n = rowsize[-1]
    N = n * nblocks

    if vec is None:
        from .comms import compute_local_size
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes((compute_local_size(N), N))
        vec.setUp()

    bvMat = bv.getMat()
    r0, r1 = bvMat.getOwnershipRange()
    bvArr = bvMat.getDenseArray()  # (r1 - r0) x nblocks

    # For each local row i (global index r0+i) and column j,
    # the target in the stacked vector is g = j * n + (r0 + i).
    nloc = r1 - r0
    local_rows = np.arange(r0, r1, dtype=PETSc.IntType)
    vec_rows = np.empty(nloc * nblocks, dtype=PETSc.IntType)
    vec_vals = np.empty(nloc * nblocks, dtype=PETSc.ScalarType)
    for j in range(nblocks):
        s = j * nloc
        vec_rows[s : s + nloc] = j * n + local_rows
        vec_vals[s : s + nloc] = bvArr[:, j]

    bv.restoreMat(bvMat)

    vec.setValues(vec_rows, vec_vals)
    vec.assemble()
    return vec


def bv_roll(
    X: SLEPc.BV,
    roll: int,
    axis: int,
    in_place: typing.Optional[bool] = False,
):
    r"""
    If :code:`axis=0` roll the rows of X by amount :code:`roll`, if
    :code:`axis=-1` roll the columns of X by amount :code:`roll`. This operation
    can be done in place if :code:`in_place == True` (default is :code:`False`).

    :type X: SLEPc.BV
    :type roll: int
    :type axis: int
    :type in_place: Optional[bool], default is :code:`False`

    :rtype: SLEPc.BV
    """
    Y = X.copy() if not in_place else X

    if axis == -1:
        Q = np.diag(np.ones(Y.getSizes()[-1]))
        Q = np.roll(Q, roll, axis=-1)
        Q = PETSc.Mat().createDense(Q.shape, None, Q, PETSc.COMM_SELF)
        Y.multInPlace(Q, 0, Y.getSizes()[-1])
        Q.destroy()
    else:
        # from ..linear_operators import MatrixLinearOperator
        Ym = Y.getMat()
        r0, r1 = Ym.getOwnershipRange()
        cols = np.arange(r1 - r0) + r0
        rows = np.mod(cols.copy() + roll, Y.getSizes()[0][-1])
        vals = np.ones(len(rows))
        sizes = (Y.getSizes()[0], Y.getSizes()[0])
        row_ptr, col, val = convert_coo_to_csr([rows, cols, vals], sizes)
        M = PETSc.Mat().createAIJ(sizes, comm=X.getComm())
        M.setPreallocationCSR((row_ptr, col))
        M.setValuesCSR(row_ptr, col, val, True)
        M.assemble(False)
        raise ValueError(
            "This error is coming from utils.bv.bv_roll(). "
            + "Need to fix this function because it was causing a"
            + "circular import."
        )
        # Mlop = MatrixLinearOperator(comm, M)
        # Y = Mlop.apply_mat(X, Y)
        # Mlop.destroy()
    return Y
