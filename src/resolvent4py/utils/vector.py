__all__ = ["enforce_complex_conjugacy", "check_complex_conjugacy"]

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def enforce_complex_conjugacy(
    comm: MPI.Comm, vec: PETSc.Vec, nblocks: int
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
    if comm.Get_rank() == 0:
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
        vec_seq.setValues(np.arange(len(array)), array)
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
    comm: MPI.Comm, vec: PETSc.Vec, nblocks: int
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
    if comm.Get_rank() == 0:
        array = vec_seq.getArray()
        block_size = len(array) // nblocks
        array_block = np.zeros(block_size, dtype=np.complex128)
        for i in range(nblocks):
            i0, i1 = i * block_size, (i + 1) * block_size
            array_block += array[i0:i1]
        cc = True if np.linalg.norm(array_block.imag) <= 1e-14 else False
    scatter.destroy()
    vec_seq.destroy()
    cc = comm.bcast(cc, root=0)
    return cc
