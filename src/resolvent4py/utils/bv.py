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
from petsc4py import PETSc
from slepc4py import SLEPc


def bv_add(alpha: float, X: SLEPc.BV, Y: SLEPc.BV) -> SLEPc.BV:
    r"""
    Compute in-place addition :math:`X \leftarrow X + \alpha Y`.

    :param alpha: scalar multiplier
    :type alpha: float
    :param X: BV to update in place
    :type X: SLEPc.BV
    :param Y: BV to add (unchanged on exit)
    :type Y: SLEPc.BV

    :return: the updated :code:`X`
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
    Return the complex conjugate :math:`\overline{X}` of the BV.

    :param X: input BV
    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :return: :math:`\overline{X}`
    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.conjugate()
    Y.restoreMat(Ym)
    return Y


def bv_real(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Return the real part :math:`\text{Re}(X)` of the BV.

    :param X: input BV
    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :return: :math:`\text{Re}(X)`
    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.realPart()
    Y.restoreMat(Ym)
    return Y


def bv_imag(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Return the imaginary part :math:`\text{Im}(X)` of the BV.

    :param X: input BV
    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :return: :math:`\text{Im}(X)`
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
) -> SLEPc.BV:
    r"""
    Extract a subset of columns from :code:`X` and store into :code:`Y`.

    :param X: source BV
    :type X: SLEPc.BV
    :param columns: array of column indices to extract
    :type columns: np.array
    :param Y: destination BV.  If :code:`None`, a new BV is created with
        the same row layout as :code:`X` and :code:`len(columns)` columns
    :type Y: Optional[SLEPc.BV], default is None

    :return: the destination BV holding the selected columns
    :rtype: SLEPc.BV
    """
    if Y is None:
        Y = SLEPc.BV().create(X.getComm())
        Y.setSizes(X.getSizes()[0], len(columns))
        Y.setType("mat")
    Q_data = np.zeros((X.getSizes()[-1], len(columns)))
    for i in range(len(columns)):
        Q_data[columns[i], i] = 1.0
    Q = PETSc.Mat().createDense(Q_data.shape, None, Q_data, PETSc.COMM_SELF)
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
    in_place: typing.Optional[bool] = False,
) -> SLEPc.BV:
    r"""
    Roll the columns of :code:`X` by amount :code:`roll`.

    :param X: BV whose columns will be rolled
    :type X: SLEPc.BV
    :param roll: shift amount (same convention as :func:`numpy.roll`)
    :type roll: int
    :param in_place: if :code:`True` modify :code:`X` in place; otherwise
        operate on a copy
    :type in_place: Optional[bool], default is :code:`False`

    :return: rolled BV (same object as :code:`X` when :code:`in_place`)
    :rtype: SLEPc.BV
    """
    Y = X.copy() if not in_place else X

    Q_data = np.diag(np.ones(Y.getSizes()[-1]))
    Q_data = np.roll(Q_data, roll, axis=-1)
    Q = PETSc.Mat().createDense(Q_data.shape, None, Q_data, PETSc.COMM_SELF)
    Y.multInPlace(Q, 0, Y.getSizes()[-1])
    Q.destroy()
    return Y
