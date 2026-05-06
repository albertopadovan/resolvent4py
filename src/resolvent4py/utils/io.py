__all__ = [
    "read_vector",
    "read_coo_matrix",
    "read_harmonic_balanced_matrix",
    "read_dense_matrix",
    "read_bv",
    "read_harmonic_balanced_bv",
    "read_harmonic_balanced_vector",
    "write_to_file",
]

import typing

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from .bv import bv_conj
from .comms import compute_local_size
from .matrix import convert_coo_to_csr


def read_vector(
    filename: str,
    sizes: typing.Optional[typing.Tuple[int, int]] = None,
) -> PETSc.Vec:
    r"""
    Read PETSc vector from file

    :param filename: name of the file that holds the vector
    :type filename: str
    :param sizes: :code:`(local size, global size)`
    :type sizes: Optional[Tuple[int, int]]

    :rtype: PETSc.Vec
    """
    comm = PETSc.COMM_WORLD
    viewer = PETSc.Viewer().createMPIIO(filename, "r", comm=comm)
    vec = PETSc.Vec().create(comm)
    vec.setSizes(sizes) if sizes != None else None
    vec.load(viewer)
    viewer.destroy()
    return vec


def read_coo_matrix(
    filenames: typing.Tuple[str, str, str],
    sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
) -> PETSc.Mat:
    r"""
    Read COO matrix from file

    :param filenames: names of the files that hold the rows, columns and values
        of the sparse matrix (e.g., :code:`(rows, cols, values)`)
    :type filenames: Tuple[str, str, str]
    :param sizes: :code:`((local rows, global rows), (local cols, global cols))`
    :type sizes: Tuple[Tuple[int, int], Tuple[int, int]]

    :rtype: PETSc.Mat
    """
    # Read COO vectors
    fname_rows, fname_cols, fname_vals = filenames
    rowsvec = read_vector(fname_rows)
    colsvec = read_vector(fname_cols)
    valsvec = read_vector(fname_vals)
    rows = np.asarray(rowsvec.getArray().real, dtype=PETSc.IntType)
    cols = np.asarray(colsvec.getArray().real, dtype=PETSc.IntType)
    vals = valsvec.getArray()
    # Delete zeros for efficiency
    idces = np.argwhere(np.abs(vals) <= 1e-16)
    rows = np.delete(rows, idces)
    cols = np.delete(cols, idces)
    vals = np.delete(vals, idces)
    # Convert COO to CSR and create the sparse matrix
    rows_ptr, cols, vals = convert_coo_to_csr([rows, cols, vals], sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=PETSc.COMM_WORLD)
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble(False)
    rowsvec.destroy()
    colsvec.destroy()
    valsvec.destroy()
    return M


def read_harmonic_balanced_matrix(
    filenames_lst: typing.List[typing.Tuple[str, str, str]],
    real_bflow: bool,
    block_sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
    full_sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
) -> PETSc.Mat:
    r"""
    Given :math:`\{\ldots, A_{-1}, A_{0}, A_{1},\ldots\}`, where :math:`A_j` is
    the :math:`j` th Fourier coefficient of a time-periodic matrix :math:`A(t)`,
    assemble the harmonic-balanced matrix

    .. math::

        M = \begin{bmatrix}
        \ddots & \ddots & \ddots & \ddots \\
        \ddots & A_{0} & A_{-1} & A_{-2} & \ddots \\
        \ddots & A_{1} & A_{0} & A_{-1} & A_{-2} & \ddots \\
        \ddots & A_{2} & A_{1} & A_{0} & A_{-1} & A_{-2} &\ddots \\
        & \ddots & A_{2} & A_{1} & A_{0} & A_{-1} &\ddots \\
        &  & \ddots & A_{2} & A_{1} & A_{0} & \ddots\\
        & & & \ddots & \ddots & \ddots & \ddots \\
        \end{bmatrix}
    

    :param filenames_lst: list of tuples, with each tuple of the form
        :code:`(rows_j.dat, cols_j.dat, vals_j.dat)` containing the COO arrays
        of the matrix :math:`A_j`
    :type filenames_lst: List[Tuple[str, str, str]]
    :param real_bflow: set to :code:`True` if :code:`filenames_lst` 
        contains the names of the COO arrays of the positive Fourier 
        coefficients of :math:`A(t)` (i.e., :math:`\{A_0, A_1, A_2, \ldots\}`). 
        The negative frequencies are assumed the complex-conjugates of the 
        positive ones. Set to :code:`False` otherwise
        (i.e., :math:`\{\ldots, A_{-2}, A_{-1}, A_0, A_1, A_2, \ldots\}`).
    :type real_bflow: bool
    :param block_sizes: sizes :code:`((local rows, global rows), 
        (local cols, global cols))` of :math:`A_j`
    :type block_sizes: Tuple[Tuple[int, int], Tuple[int, int]]
    :param full_sizes: sizes :code:`((local rows, global rows), 
        (local cols, global cols))` of :math:`M`
    :type full_sizes: Tuple[Tuple[int, int], Tuple[int, int]]

    :return: Sparse PETSc matrix :math:`M`
    :rtype: PETSc.Mat
    """
    # Read list of COO vectors
    rows_lst, cols_lst, vals_lst = [], [], []
    for filenames in filenames_lst:
        fname_rows, fname_cols, fname_vals = filenames
        rowsvec = read_vector(fname_rows)
        colsvec = read_vector(fname_cols)
        valsvec = read_vector(fname_vals)
        rowsvec_arr = np.asarray(
            rowsvec.getArray().real, dtype=PETSc.IntType
        ).copy()
        colsvec_arr = np.asarray(
            colsvec.getArray().real, dtype=PETSc.IntType
        ).copy()
        valsvec_arr = valsvec.getArray().copy()
        rows_lst.append(rowsvec_arr)
        cols_lst.append(colsvec_arr)
        vals_lst.append(valsvec_arr)
        rowsvec.destroy()
        colsvec.destroy()
        valsvec.destroy()

    if real_bflow:
        l = len(rows_lst)
        for i in range(1, l):
            idx_lst = i - l
            rows_lst.insert(0, rows_lst[idx_lst])
            cols_lst.insert(0, cols_lst[idx_lst])
            vals_lst.insert(0, np.conj(vals_lst[idx_lst]))

    rows, cols, vals = [], [], []
    Nrb = block_sizes[0][-1]  # Number of rows for each block
    Ncb = block_sizes[-1][-1]  # Number of columns for each block
    nblocks = full_sizes[0][-1] // Nrb  # Number of blocks
    nfb = (len(rows_lst) - 1) // 2  # Number of baseflow frequencies
    nfp = (nblocks - 1) // 2  # Number of perturbation frequencies
    if nfp < nfb:
        raise ValueError(
            f"The number of blocks must be larger than the number of Fourier "
            f"coefficients of A(t). (See function description.)"
        )
    for i in range(2 * nfp + 1):
        for j in range(2 * nfp + 1):
            k = i - j + nfb
            if k >= 0 and k < (2 * nfb + 1):
                rows.extend(rows_lst[k] + i * Nrb)
                cols.extend(cols_lst[k] + j * Ncb)
                vals.extend(vals_lst[k])
    rows = np.asarray(rows, dtype=PETSc.IntType)
    cols = np.asarray(cols, dtype=PETSc.IntType)
    vals = np.asarray(vals)

    # Delete zeros for efficiency
    idces = np.argwhere(np.abs(vals) <= 1e-16)
    rows = np.delete(rows, idces)
    cols = np.delete(cols, idces)
    vals = np.delete(vals, idces)
    # Convert COO to CSR and create the sparse matrix
    rows_ptr, cols, vals = convert_coo_to_csr([rows, cols, vals], full_sizes)
    M = PETSc.Mat().createAIJ(full_sizes, comm=PETSc.COMM_WORLD)
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble(False)

    return M


def read_dense_matrix(
    filename: str,
    sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
) -> PETSc.Mat:
    r"""
    Read dense PETSc matrix from file.

    :param filename: name of the file that holds the matrix
    :type filename: str
    :param sizes: :code:`((local rows, global rows), (local cols, global cols))`
    :type sizes: Tuple[Tuple[int, int], Tuple[int, int]]

    :rtype: PETSc.Mat
    """
    comm = PETSc.COMM_WORLD
    viewer = PETSc.Viewer().createMPIIO(filename, "r", comm=comm)
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.load(viewer)
    viewer.destroy()
    return M


def read_bv(
    filename: str,
    sizes: typing.Tuple[typing.Tuple[int, int], int],
) -> SLEPc.BV:
    r"""
    Read dense matrix from file and store as a SLEPc BV

    :param filename: name of the file that holds the matrix
    :type filename: str
    :param sizes: :code:`((local rows, global rows), global columns)`
    :type sizes: Tuple[Tuple[int, int], int]

    :rtype: SLEPc.BV
    """
    ncols = sizes[-1]
    sizes_mat = (sizes[0], (compute_local_size(ncols), ncols))
    Mm = read_dense_matrix(filename, sizes_mat)
    M = SLEPc.BV().createFromMat(Mm)
    M.setType("mat")
    Mm.destroy()
    return M


def read_harmonic_balanced_bv(
    filenames_lst: typing.List[str],
    real_bflow: bool,
    block_sizes: typing.Tuple[typing.Tuple[int, int], int],
    full_sizes: typing.Tuple[typing.Tuple[int, int], int],
) -> SLEPc.BV:
    r"""
    Given :math:`\{\ldots, A_{-1}, A_{0}, A_{1},\ldots\}`, where :math:`A_j` is
    the :math:`j` th Fourier coefficient of a time-periodic matrix :math:`A(t)`,
    assemble the harmonic-balanced matrix

    .. math::

        M = \begin{bmatrix}
        \ddots & \ddots & \ddots & \ddots \\
        \ddots & A_{0} & A_{-1} & A_{-2} & \ddots \\
        \ddots & A_{1} & A_{0} & A_{-1} & A_{-2} & \ddots \\
        \ddots & A_{2} & A_{1} & A_{0} & A_{-1} & A_{-2} &\ddots \\
        & \ddots & A_{2} & A_{1} & A_{0} & A_{-1} &\ddots \\
        &  & \ddots & A_{2} & A_{1} & A_{0} & \ddots\\
        & & & \ddots & \ddots & \ddots & \ddots \\
        \end{bmatrix}
    
    :param comm: MPI communicator (PETSc.COMM_WORLD)
    :type comm: PETSc.Comm
    :param filenames_lst: list of names of files where the Fourier modes 
        :math:`A_j` are stored as SLEPc BVs
    :type filenames_lst: List[str]
    :param real_bflow: set to :code:`True` if :code:`filenames_lst` 
        contains the names of the COO arrays of the positive Fourier 
        coefficients of :math:`A(t)` (i.e., :math:`\{A_0, A_1, A_2, \ldots\}`). 
        The negative frequencies are assumed the complex-conjugates of the 
        positive ones. Set to :code:`False` otherwise
        (i.e., :math:`\{\ldots, A_{-2}, A_{-1}, A_0, A_1, A_2, \ldots\}`).
    :type real_bflow: bool
    :param block_sizes: sizes :code:`((local rows, global rows), global_cols)` 
        of :math:`A_j`
    :type block_sizes: Tuple[Tuple[int, int], int]
    :param full_sizes: sizes :code:`((local rows, global rows), global_cols)` 
        of :math:`M`
    :type full_sizes: Tuple[Tuple[int, int], int]

    :return: SLEPc BV :math:`M`
    :rtype: SLEPc.BV
    """
    # Read list of BVs
    bvs_lst = []
    for filename in filenames_lst:
        bv = read_bv(filename, block_sizes)
        bvs_lst.append(bv.copy())
        bv.destroy()

    if real_bflow:
        l = len(bvs_lst)
        for i in range(1, l):
            idx_lst = i - l
            bvs_lst.insert(0, bv_conj(bvs_lst[idx_lst]))

    Nrb = block_sizes[0][-1]  # Number of rows for each block
    Ncb = block_sizes[-1]  # Number of cols for each block
    nblocks = full_sizes[0][-1] // Nrb  # Number of blocks
    nfb = (len(bvs_lst) - 1) // 2  # Number of baseflow frequencies
    nfp = (nblocks - 1) // 2  # Number of perturbation frequencies
    if nfp < nfb:
        raise ValueError(
            f"The number of blocks must be larger than the number of Fourier "
            f"coefficients of A(t). (See function description.)"
        )

    bv_mat = bvs_lst[0].getMat()
    r0, r1 = bv_mat.getOwnershipRange()
    bvs_lst[0].restoreMat(bv_mat)
    rows = np.arange(r0, r1, dtype=PETSc.IntType)
    cols = np.arange(Ncb, dtype=PETSc.IntType)
    M = SLEPc.BV().create(PETSc.COMM_WORLD)
    M.setSizes(full_sizes[0], full_sizes[-1])
    M.setType("mat")
    Mmat = M.getMat()
    for i in range(2 * nfp + 1):
        for j in range(2 * nfp + 1):
            k = i - j + nfb
            if k >= 0 and k < (2 * nfb + 1):
                rows_k = rows + i * Nrb
                cols_k = cols + j * Ncb
                bv_k_mat = bvs_lst[k].getMat()
                array = bv_k_mat.getDenseArray().reshape(-1)
                Mmat.setValues(rows_k, cols_k, array, None)
                bvs_lst[k].restoreMat(bv_k_mat)
    Mmat.assemble(None)
    M.restoreMat(Mmat)
    for bv in bvs_lst:
        bv.destroy()
    return M


def read_harmonic_balanced_vector(
    filenames_lst: typing.List[str],
    real_bflow: bool,
    block_sizes: typing.Tuple[int, int],
    full_sizes: typing.Tuple[int, int],
) -> PETSc.Vec:
    r"""
    Given :math:`\{\ldots, v_{-1}, v_{0}, v_{1},\ldots\}`, where :math:`v_j` is
    the :math:`j` th Fourier coefficient of a time-periodic vector :math:`v(t)`,
    assemble

    .. math::

        \hat{v} = \begin{bmatrix}
            \vdots \\ v_{-1} \\ v_0 \\ v_{1} \\ \vdots
        \end{bmatrix}.

    :param filenames_lst: list of names of files where the Fourier modes
        :math:`v_j` are stored as PETSc Vec
    :type filenames_lst: List[str]
    :param real_bflow: set to :code:`True` if :code:`filenames_lst`
        contains the names of the COO arrays of the positive Fourier
        coefficients of :math:`v(t)` (i.e., :math:`\{v_0, v_1, v_2, \ldots\}`).
        The negative frequencies are assumed the complex-conjugates of the
        positive ones. Set to :code:`False` otherwise
        (i.e., :math:`\{\ldots, v_{-2}, v_{-1}, v_0, v_1, v_2, \ldots\}`).
    :type real_bflow: bool
    :param block_sizes: sizes :code:`(local rows, global rows)` of :math:`v_j`
    :type block_sizes: Tuple[int, int]
    :param full_sizes: sizes :code:`(local rows, global rows)` of :math:`\hat{v}`
    :type full_sizes: Tuple[int, int]

    :return: vector :math:`\hat{v}`
    :rtype: PETSc.Vec
    """
    vec_lst = []
    for filename in filenames_lst:
        vec = read_vector(filename, block_sizes)
        vec_lst.append(vec.copy())
        vec.destroy()

    if real_bflow:
        l = len(vec_lst)
        for i in range(1, l):
            idx_lst = i - l
            vecconj = vec_lst[idx_lst].copy()
            vecconj.conjugate()
            vec_lst.insert(0, vecconj.copy())
            vecconj.destroy()

    Vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    Vec.setSizes(full_sizes)
    Vec.setUp()
    r0, _ = vec_lst[0].getOwnershipRange()
    nfb = (len(vec_lst) - 1) // 2  # Number of baseflow frequencies
    nfp = (
        full_sizes[-1] // block_sizes[-1] - 1
    ) // 2  # Number of perturbation frequencies
    for i in range(2 * nfb + 1):
        j = i + (nfp - nfb)
        rows = (
            j * block_sizes[-1]
            + np.arange(block_sizes[0], dtype=PETSc.IntType)
            + r0
        )
        Vec.setValues(rows, vec_lst[i].getArray(), False)
    Vec.assemble()

    for vec in vec_lst:
        vec.destroy()

    return Vec


def write_to_file(
    filename: str,
    object: typing.Union[PETSc.Mat, PETSc.Vec, SLEPc.BV],
) -> None:
    r"""
    Write PETSc/SLEPc object (PETSc.Mat, PETSc.Vec or SLEPc.BV) to file.

    :param filename: name of the file to store the object
    :type filename: str
    :param object: any PETSc matrix or vector, or SLEPc BV
    :type object: Union[PETSc.Mat, PETSc.Vec, SLEPc.BV]
    """
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=object.getComm())
    if isinstance(object, SLEPc.BV):
        mat = object.getMat()
        mat.view(viewer)
        object.restoreMat(mat)
    else:
        object.view(viewer)
    viewer.destroy()
