from .. import SLEPc
from .. import MPI
from .. import typing
from .. import np


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


def assemble_harmonic_balanced_bv(
        bvs_lst: typing.List[SLEPc.BV],
        bflow_freqs: np.array,
        pertb_freqs: np.array,
        sizes: typing.Tuple[typing.Tuple[int, int], int]
) -> SLEPc.BV:
    
    if len(bflow_freqs) != len(bvs_lst):
        raise ValueError (
            f"Error in assemble_harmonic_balanced_bv(). bvs_lst "
            f"should have the same length as bflow_freqs."
        )
    
    put_back = False
    if np.min(bflow_freqs) == 0.0:
        put_back = True
        for i in range (1, len(bflow_freqs)):
            idx_lst = i - 1 - nfp
            bvs_lst.insert(0, bv_conj(bvs_lst[idx_lst], False))
        bflow_freqs = np.concatenate((-np.flipud(bflow_freqs[1:]), bflow_freqs))
    
    # Create the harmonic-balanced BV
    BV = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    BV.setSizes(sizes)
    BV.setType('mat')
    BV_mat = BV.getMat()
    m = bvs_lst[0].getMat()
    r0, _ = m.getOwnershipRange()
    bvs_lst[0].restoreMat(m)
    
    nfb = (len(bflow_freqs) - 1)//2     # Number of perturbation frequencies
    nfp = (len(pertb_freqs) - 1)//2     # Number of baseflow frequencies
    bv_sizes = bvs_lst[0].getSizes()
    nrows_loc, nrows = bv_sizes[0]
    ncols = bv_sizes[-1]
    bvdata = np.zeros((nrows_loc, ncols*(2*nfp + 1)), dtype=np.complex128)
    
    for i in range (2*nfp + 1):
        cols = []
        rows = i*nrows + np.arange(nrows_loc, dtype=np.int32) + r0
        for j in range (2*nfb + 1):
            k = i - j + nfb
            if k >= 0:
                m = bvs_lst[k].getMat()
                bvdata[:, j*ncols:(j+1)*ncols] = m.getDenseArray()
                bvs_lst[k].restoreMat(m)
                cols.extend(np.arange(ncols, dtype=np.int32) + j*ncols)
        cols = np.asarray(cols, dtype=np.int32)
        BV_mat.setValues(rows, cols, bvdata, None)
    BV_mat.assemble(None)
    BV.restoreMat(BV_mat)

    if put_back:
        bflow_freqs = bflow_freqs[nfb:]
        for i in range (nfb):
            bvs_lst[i].destroy()
        bvs_lst[nfb:]
    
    return BV


        

