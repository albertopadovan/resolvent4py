__all__ = [
    "enforce_complex_conjugacy",
    "check_complex_conjugacy",
    "vec_real",
    "vec_imag",
]

import numpy as np
import typing
from petsc4py import PETSc


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
