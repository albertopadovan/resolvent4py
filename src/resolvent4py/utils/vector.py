__all__ = [
    "array_from_petsc_vector",
    "enforce_complex_conjugacy",
    "check_complex_conjugacy",
    "vec_real",
    "vec_imag",
    "assemble_harmonic_balanced_vector",
    "reshape_harmonic_balanced_vector_into_bv",
    "embed_into_2T_vec",
]

import typing

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from .bv import reshape_bv_into_harmonic_balanced_vector


def array_from_petsc_vector(vector: PETSc.Vec) -> np.ndarray:
    r"""
    Gather a PETSc vector into a NumPy array.

    :param vector: PETSc vector
    :type vector: PETSc.Vec

    :return: Global vector values
    :rtype: np.ndarray
    """
    try:
        if vector.getComm().getSize() == 1:
            return vector.getArray(readonly=True).copy()
    except AttributeError:
        return vector.getArray(readonly=True).copy()

    scatter, all_vector = PETSc.Scatter.toAll(vector)
    try:
        scatter.scatter(vector, all_vector, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return all_vector.getArray(readonly=True).copy()
    finally:
        all_vector.destroy()
        scatter.destroy()


def vec_real(
    x: PETSc.Vec, inplace: typing.Optional[bool] = False
) -> PETSc.Vec:
    r"""
    Return the real part :math:`\text{Re}(x)` of a PETSc vector.

    :param x: input vector
    :type x: PETSc.Vec
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new PETSc.Vec
    :type inplace: Optional[bool], default is False

    :return: :math:`\text{Re}(x)`
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
    Return the imaginary part :math:`\text{Im}(x)` of a PETSc vector.

    :param x: input vector
    :type x: PETSc.Vec
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new PETSc.Vec
    :type inplace: Optional[bool], default is False

    :return: :math:`\text{Im}(x)`
    :rtype: PETSc.Vec
    """
    y = x if inplace else x.copy()
    ya = y.getArray()
    ya[:] = ya.imag
    return y


def enforce_complex_conjugacy(vec: PETSc.Vec, nblocks: int) -> None:
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
    if vec.getComm().getRank() == 0:
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


def check_complex_conjugacy(vec: PETSc.Vec, nblocks: int) -> bool:
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
    :rtype: bool
    """
    if np.mod(nblocks, 2) == 0:
        raise ValueError(
            "The number of blocks must be an odd number. "
            "Currently you set {nblocks} blocks."
        )
    comm = vec.getComm()
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
    vec_lst: list[PETSc.Vec],
    bflow_freqs: np.array,
    pertb_freqs: np.array,
    sizes: tuple[int, int],
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
            "Error in assemble_harmonic_balanced_vector(). vec_lst "
            "should have the same length as bflow_freqs."
        )

    put_back = False
    if np.min(bflow_freqs) == 0.0:
        put_back = True
        n_original = len(vec_lst)
        for i in range(1, n_original):
            idx_lst = i - n_original
            vecconj = vec_lst[idx_lst].copy()
            vecconj.conjugate()
            vec_lst.insert(0, vecconj)
        bflow_freqs = np.concatenate(
            (-np.flipud(bflow_freqs[1:]), bflow_freqs)
        )

    # Create the harmonic-balanced vector.  Same assembly recipe as
    # :func:`~resolvent4py.utils.io.read_harmonic_balanced_vector`: each
    # rank writes its own local slice of every input vector at the right
    # block offset, and PETSc routes cross-rank rows during assemble().
    Vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    Vec.setSizes(sizes)
    Vec.setUp()
    r0, _ = vec_lst[0].getOwnershipRange()
    nfb = (len(bflow_freqs) - 1) // 2  # Number of baseflow frequencies
    nfp = (len(pertb_freqs) - 1) // 2  # Number of perturbation frequencies
    # PETSc.Vec.getSizes() returns a single (local, global) tuple.
    nrows_loc, nrows = vec_lst[0].getSizes()
    for i in range(2 * nfb + 1):
        j = i + (nfp - nfb)
        rows = j * nrows + np.arange(nrows_loc, dtype=PETSc.IntType) + r0
        Vec.setValues(rows, vec_lst[i].getArray(), False)
    Vec.assemble()

    if put_back:
        # Destroy the conjugate temporaries we prepended and remove them
        # from the caller's list so it holds no dangling references.
        for i in range(nfb):
            vec_lst[i].destroy()
        del vec_lst[:nfb]

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
    from .matrix import convert_coo_to_csr

    bvMat = bv.getMat()
    rows_ptr, cols_csr, vals_csr = convert_coo_to_csr(
        [bv_rows, bv_cols, bv_vals], bvMat.getSizes()
    )
    # The BV's underlying Mat is reused across calls — overwrite, don't
    # accumulate.  In petsc4py the 4th argument is the ``addv`` mode:
    # True = ADD_VALUES, False = INSERT_VALUES.
    bvMat.zeroEntries()
    bvMat.setPreallocationCSR((rows_ptr, cols_csr))
    bvMat.setValuesCSR(rows_ptr, cols_csr, vals_csr, False)
    bvMat.assemble()

    bv.restoreMat(bvMat)
    return bv


def embed_into_2T_vec(
    freqsT: np.ndarray,
    vecT: PETSc.Vec,
    freqs2T: np.ndarray,
    vec2T: typing.Optional[PETSc.Vec] = None,
    enforce_cc: bool = False,
) -> PETSc.Vec:
    r"""
    Embed a T-periodic harmonic-balanced vector into a 2T-periodic one.

    Given the T-periodic vector :math:`\hat{v}^T` with Fourier
    coefficients at the frequencies listed in ``freqsT``, lay them out
    into the (larger) 2T-periodic vector :math:`\hat{v}^{2T}` whose
    Fourier coefficients live at ``freqs2T``.  Each ``freqsT[i]`` is
    matched to the closest entry in ``freqs2T`` and the corresponding
    block is copied across; entries of ``freqs2T`` with no match in
    ``freqsT`` are left at zero.

    Typical use: ``freqsT = omega * arange(-nf, nf+1)`` and
    ``freqs2T = (omega/2) * arange(-2*nf, 2*nf+1)``.  The T harmonics
    then land on the even (integer-:math:`\omega`) indices of the 2T
    basis and the odd (half-integer-:math:`\omega`) indices stay zero
    — exactly the structure of a period-doubling lift.

    :param freqsT: frequency array of the T-periodic vector, length
        :math:`2 n_f^T + 1`.
    :type freqsT: np.ndarray
    :param vecT: T-periodic harmonic-balanced vector, size
        ``n * len(freqsT)``.
    :type vecT: PETSc.Vec
    :param freqs2T: frequency array of the 2T-periodic vector, length
        :math:`2 n_f^{2T} + 1`.  Must contain every entry of ``freqsT``
        to within numerical tolerance.
    :type freqs2T: np.ndarray
    :param vec2T: optional pre-allocated 2T-periodic vector to write
        into.  If ``None``, a new vector is created.
    :type vec2T: Optional[PETSc.Vec]
    :param enforce_cc: if ``True``, call
        :func:`enforce_complex_conjugacy` on the result to make
        :math:`v_{-i} = \overline{v_i}`.
    :type enforce_cc: bool

    :return: the assembled 2T-periodic harmonic-balanced vector.
    :rtype: PETSc.Vec
    """
    bvT = reshape_harmonic_balanced_vector_into_bv(vecT, len(freqsT))

    bv2T = SLEPc.BV().create(comm=vecT.getComm())
    bv2T.setSizes(bvT.getSizes()[0], len(freqs2T))
    bv2T.setType("mat")

    for iT, fT in enumerate(freqsT):
        i2T = np.argmin(np.abs(freqs2T - fT))
        if np.abs(fT - freqs2T[i2T]) > 1e-10:
            raise ValueError(
                "The array freqsT should be a subset of the the "
                "array freqs2T. Embedding otherwise makes no sense."
            )
        vT = bvT.getColumn(iT)
        bv2T.insertVec(i2T, vT)
        bvT.restoreColumn(iT, vT)

    vec2T = reshape_bv_into_harmonic_balanced_vector(bv2T, vec2T)
    enforce_complex_conjugacy(vec2T, len(freqs2T)) if enforce_cc else None

    bvT.destroy()
    bv2T.destroy()
    return vec2T
