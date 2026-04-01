__all__ = [
    "enforce_complex_conjugacy",
    "check_complex_conjugacy",
    "vec_real",
    "vec_imag",
]

import numpy as np
import typing
from petsc4py import PETSc
from slepc4py import SLEPc


def vec_real(
    x: PETSc.Vec, inplace: typing.Optional[bool] = False
) -> PETSc.Vec:
    r"""
    Returns the real part :math:`\text{Re}(x)` of the PETSc Vec.

    :type x: PETSc.Vec
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new PETSc.Vec
    :type inplace: Optional[bool], default is False

    :rtype: PETSc.Vec
    """
    y = x if inplace else x.copy()
    ya = y.getArray()
    ya[:] = ya.real
    return y


def vec_imag(
    x: PETSc.Vec, inplace: typing.Optional[bool] = False
) -> PETSc.Vec:
    r"""
    Returns the imaginary part :math:`\text{Im}(x)` of the PETSc Vec.

    :type x: PETSc.Vec
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new PETSc.Vec
    :type inplace: Optional[bool], default is False

    :rtype: PETSc.Vec
    """
    y = x if inplace else x.copy()
    ya = y.getArray()
    ya[:] = ya.imag
    return y


def enforce_complex_conjugacy(
    comm: PETSc.Comm, vec: PETSc.Vec, nblocks: int
) -> None:
    r"""
    Suppose we have a vector

    .. math::
        v = \left(\ldots,v_{-1},v_{0},v_{1},\ldots\right)

    where :math:`v_i` are complex vectors. This function enforces
    :math:`v_{-i} = \overline{v_{i}}` for all :math:`i` (this implies that
    :math:`v_0` will be purely real).

    :param vec: vector :math:`v` described above
    :type vec: PETSc.Vec
    :param nblocks: number of vectors :math:`v_i` in :math:`v`. This must
        be an odd number.
    :type nblocks: int

    :rtype: None
    """
    if np.mod(nblocks, 2) == 0:
        raise ValueError(
            "The number of blocks must be an odd number. "
            "Currently you set {nblocks} blocks."
        )
    scatter, vec_seq = PETSc.Scatter().toZero(vec)
    scatter.begin(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    scatter.end(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    if comm.getRank() == 0:
        array = vec_seq.getArray()
        block_size = len(array) // nblocks
        for i in range(nblocks // 2):
            j = nblocks - 1 - i
            i0, i1 = i * block_size, (i + 1) * block_size
            j0, j1 = j * block_size, (j + 1) * block_size
            array[i0:i1] = array[j0:j1].conj()
        i = nblocks // 2
        i0, i1 = i * block_size, (i + 1) * block_size
        array[i0:i1] = array[i0:i1].real
        vec_seq.setValues(np.arange(len(array), dtype=PETSc.IntType), array)
        vec_seq.assemble()
    scatter.begin(
        vec_seq,
        vec,
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.REVERSE,
    )
    scatter.end(
        vec_seq,
        vec,
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.REVERSE,
    )
    scatter.destroy()
    vec_seq.destroy()


def check_complex_conjugacy(
    comm: PETSc.Comm, vec: PETSc.Vec, nblocks: int
) -> bool:
    r"""
    Verify whether the components :math:`v_i` of the vector

    .. math::
        v = \left(\ldots,v_{-1},v_{0},v_{1},\ldots\right)

    satisfy :math:`v_{-i} = \overline{v_{i}}` for all :math:`i`.

    :param vec: vector :math:`v` described above
    :type vec: PETSc.Vec
    :param nblocks: number of vectors :math:`v_i` in :math:`v`. This must
        be an odd number.
    :type nblocks: int

    :return: :code:`True` if the components are complex-conjugates of each
        other and :code:`False` otherwise
    :rtype: Bool
    """
    if np.mod(nblocks, 2) == 0:
        raise ValueError(
            "The number of blocks must be an odd number. "
            "Currently you set {nblocks} blocks."
        )
    scatter, vec_seq = PETSc.Scatter().toZero(vec)
    scatter.begin(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    scatter.end(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    cc = None
    if comm.getRank() == 0:
        array = vec_seq.getArray()
        block_size = len(array) // nblocks
        array_block = np.zeros(block_size, dtype=np.complex128)
        for i in range(nblocks):
            i0, i1 = i * block_size, (i + 1) * block_size
            array_block += array[i0:i1]
        array_block /= np.linalg.norm(array_block)
        cc = True if np.linalg.norm(array_block.imag) <= 1e-13 else False
    scatter.destroy()
    vec_seq.destroy()
    cc = comm.tompi4py().bcast(cc, root=0)
    return cc


def assemble_harmonic_balanced_vector(
    vec_lst: typing.List[PETSc.Vec],
    bflow_freqs: np.array,
    pertb_freqs: np.array,
    sizes: typing.Tuple[int, int],
) -> PETSc.Vec:
    r"""
    Assemble a harmonic-balanced vector from its Fourier coefficient
    vectors.

    Given :math:`\{v_{-n_{fb}}, \ldots, v_{n_{fb}}\}`, assemble

    .. math::

        \hat{v} = \begin{bmatrix}
            \vdots \\ v_{-1} \\ v_0 \\ v_{1} \\ \vdots
        \end{bmatrix}

    of total size :math:`(2 n_{fp} + 1) N`, where :math:`n_{fp}` is the
    number of perturbation frequencies and :math:`N` is the size of each
    :math:`v_j`.  Blocks outside the baseflow bandwidth
    (:math:`|j| > n_{fb}`) are left as zero.

    If ``bflow_freqs`` starts at zero (i.e.\ only non-negative
    frequencies are provided), the negative-frequency vectors are
    generated automatically as complex conjugates:
    :math:`v_{-j} = \overline{v_j}`.

    :param vec_lst: list of PETSc vectors :math:`v_j`.
        If ``bflow_freqs`` starts at 0, the list should contain
        :math:`\{v_0, v_1, \ldots, v_{n_{fb}}\}` (length
        :math:`n_{fb} + 1`).
        Otherwise it should contain
        :math:`\{v_{-n_{fb}}, \ldots, v_{n_{fb}}\}` (length
        :math:`2 n_{fb} + 1`).
    :type vec_lst: List[PETSc.Vec]
    :param bflow_freqs: baseflow frequency array
    :type bflow_freqs: np.array
    :param pertb_freqs: perturbation frequency array
        :math:`\omega (-n_{fp}, \ldots, -1, 0, 1, \ldots, n_{fp})`
    :type pertb_freqs: np.array
    :param sizes: ``(local_size, global_size)`` of the assembled vector
    :type sizes: Tuple[int, int]

    :return: the assembled harmonic-balanced vector :math:`\hat{v}`
    :rtype: PETSc.Vec
    """
    if len(bflow_freqs) != len(vec_lst):
        raise ValueError(
            f"Error in assemble_harmonic_balanced_vector(). vec_lst "
            f"should have the same length as bflow_freqs."
        )

    put_back = False
    if np.min(bflow_freqs) == 0.0:
        put_back = True
        for i in range(1, len(bflow_freqs)):
            idx_lst = i - 1 - nfp
            vecconj = vec_lst[idx_lst].copy()
            vecconj.conjugate()
            vec_lst.insert(0, vecconj)
        bflow_freqs = np.concatenate(
            (-np.flipud(bflow_freqs[1:]), bflow_freqs)
        )

    # Create the harmonic-balanced BV
    Vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    Vec.setSizes(sizes)
    Vec.setUp()
    r0, _ = Vec.getOwnershipRange()
    nfb = (len(bflow_freqs) - 1) // 2  # Number of baseflow frequencies
    nfp = (len(pertb_freqs) - 1) // 2  # Number of perturbation frequencies
    vec_sizes = vec_lst[0].getSizes()
    nrows_loc, nrows = vec_sizes[0]
    for i in range(2 * nfb + 1):
        j = i + (nfp - nfb)
        rows = j * nrows + np.arange(nrows_loc, dtype=PETSc.IntType) + r0
        Vec.setValues(rows, vec_lst[j].getArray(), False)
    Vec.assemble()

    if put_back:
        bflow_freqs = bflow_freqs[nfb:]
        for i in range(nfb):
            vec_lst[i].destroy()
        vec_lst[nfb:]

    return Vec


def reshape_harmonic_balanced_vector_into_bv(
    vec: PETSc.Vec,
    nblocks: int,
    bv: SLEPc.BV = None,
) -> SLEPc.BV:
    r"""
    Reshape a harmonic-balanced vector into a SLEPc BV (basis vectors)
    matrix.

    Given :math:`\hat{v} = (v_{-m}, \ldots, v_m)^T` of total size
    :math:`N = n \cdot (2m+1)`, reshape it into an :math:`n \times (2m+1)`
    dense matrix (stored as a SLEPc BV) where column :math:`j` holds
    :math:`v_{j-m}`.

    The vector :math:`\hat{v}` and the BV matrix may have different
    parallel row distributions, so values are exchanged across ranks
    via PETSc's COO assembly (``setPreallocationCOO`` /
    ``setValuesCOO``).

    :param vec: distributed PETSc vector of size :math:`n \cdot (2m+1)`
    :type vec: PETSc.Vec
    :param nblocks: number of harmonic blocks :math:`2m+1`
    :type nblocks: int
    :param bv: optional pre-allocated SLEPc BV of size
        :math:`n \times (2m+1)`.  If ``None``, a new BV is created.
    :type bv: Optional[SLEPc.BV]

    :return: the reshaped BV
    :rtype: SLEPc.BV
    """
    comm = vec.getComm()
    N = vec.getSizes()[-1]
    n = N // nblocks

    if bv is None:
        from .comms import compute_local_size
        bv = SLEPc.BV().create(comm=comm)
        bv.setSizes((compute_local_size(n), n), nblocks)
        bv.setType("mat")

    rowsize, ncols = bv.getSizes()
    if ncols != nblocks:
        raise ValueError(
            f"The number of columns in the provided BV should be "
            f"equal to nblocks. Currently, ncols = {ncols} and "
            f"nblocks = {nblocks}."
        )
    if rowsize[-1] != n:
        raise ValueError(
            f"The row size of the provided BV should be equal to "
            f"N / nblocks = {n}. Currently, rowsize = {rowsize[-1]}."
        )

    # For each local entry vec[g] (global index g in the stacked vector),
    # compute the target (row, col) in the n x nblocks matrix:
    #   block index j = g // n   (which column, 0-based)
    #   local index i = g % n    (which row in the block)
    # So bvMat[i, j] = vec[g].
    local_globals = np.arange(*vec.getOwnershipRange(), dtype=PETSc.IntType)
    bv_rows = (local_globals % n).astype(PETSc.IntType)
    bv_cols = (local_globals // n).astype(PETSc.IntType)
    bv_vals = np.asarray(vec.getArray(readonly=True), dtype=PETSc.ScalarType)

    # Use CSR assembly via convert_coo_to_csr: PETSc handles
    # cross-rank redistribution during the COO-to-CSR conversion.
    from .matrix import convert_coo_to_csr_v2 as convert_coo_to_csr
    bvMat = bv.getMat()
    rows_ptr, cols_csr, vals_csr = convert_coo_to_csr(
        [bv_rows, bv_cols, bv_vals], bvMat.getSizes()
    )
    bvMat.setPreallocationCSR((rows_ptr, cols_csr))
    bvMat.setValuesCSR(rows_ptr, cols_csr, vals_csr, True)
    bvMat.assemble()

    bv.restoreMat(bvMat)
    return bv
