import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from resolvent4py.linear_operators.superoptimal_block_circulant import _extract_toeplitz_blocks
from resolvent4py.utils.comms import scatter_array_from_root_to_all
from resolvent4py.utils.matrix import convert_coo_to_csr_v2


def _build_block_toeplitz_petsc(comm, blocks, nblocks):
    r"""
    Build a block-Toeplitz PETSc matrix Ahat from a list of N x N numpy
    blocks [A_{-nf}, ..., A_{nf}].

    Ahat_{k,j} = A_{k-j}  for |k - j| <= nf, else 0.

    blocks is indexed 0..2*nf, corresponding to A_{-nf}..A_{nf}.
    """
    nf = (len(blocks) - 1) // 2
    N = blocks[0].shape[0]
    nN = nblocks * N

    # Build COO arrays on rank 0 only, then scatter
    rows_coo, cols_coo, vals_coo = None, None, None
    Ahat_np = np.zeros((nN, nN), dtype=np.complex128)
    if comm.getRank() == 0:
        for k in range(nblocks):
            for j in range(nblocks):
                m = k - j
                if abs(m) <= nf:
                    Ahat_np[
                        k * N : (k + 1) * N, j * N : (j + 1) * N
                    ] = blocks[m + nf]

        r, c = np.nonzero(Ahat_np)
        rows_coo = np.asarray(r, dtype=PETSc.IntType)
        cols_coo = np.asarray(c, dtype=PETSc.IntType)
        vals_coo = np.asarray(Ahat_np[r, c], dtype=PETSc.ScalarType)

    # Broadcast Ahat_np to all ranks for verification
    Ahat_np = comm.tompi4py().bcast(Ahat_np, root=0)
    
    rows = scatter_array_from_root_to_all(rows_coo)
    cols = scatter_array_from_root_to_all(cols_coo)
    vals = scatter_array_from_root_to_all(vals_coo)

    Nl = res4py.compute_local_size(nN)
    sizes = ((Nl, nN), (Nl, nN))
    rows_ptr, cols_csr, vals_csr = convert_coo_to_csr_v2(
        [rows, cols, vals], sizes
    )
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols_csr))
    M.setValuesCSR(rows_ptr, cols_csr, vals_csr, True)
    M.assemble()
    return M, Ahat_np


def test_extract_toeplitz_blocks_recovers_blocks(comm):
    r"""Test that _extract_toeplitz_blocks recovers the original A_j blocks
    from a block-Toeplitz matrix."""
    N = 6
    nf = 2
    nblocks = 2 * nf + 3  # np > nf, so some zero blocks exist

    # Generate random N x N Fourier coefficient blocks A_{-nf}..A_{nf}
    rng = np.random.default_rng(42)
    blocks = [
        rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        for _ in range(2 * nf + 1)
    ]

    Ahat_petsc, Ahat_np = _build_block_toeplitz_petsc(
        comm, blocks, nblocks
    )

    # Extract blocks
    Alst = _extract_toeplitz_blocks(Ahat_petsc, nblocks)

    np_ = (nblocks - 1) // 2
    for i in range(nblocks):
        m = i - np_  # Fourier index
        # Convert extracted PETSc block to sequential dense numpy
        block_petsc = Alst[i]
        block_dense = block_petsc.copy()
        block_dense.convert(PETSc.Mat.Type.DENSE)
        block_seq = res4py.distributed_to_sequential_matrix(block_dense)
        block_arr = block_seq.getDenseArray().copy()
        block_seq.destroy()
        block_dense.destroy()

        if abs(m) <= nf:
            expected = blocks[m + nf]
        else:
            expected = np.zeros((N, N), dtype=np.complex128)

        error = np.linalg.norm(block_arr - expected)
        norm = np.linalg.norm(expected) if np.linalg.norm(expected) > 0 else 1.0
        assert error / norm < 1e-12, (
            f"Block i={i} (m={m}): relative error {error / norm:.2e}"
        )

    # Cleanup
    for mat in Alst:
        mat.destroy()
    Ahat_petsc.destroy()


def test_extract_toeplitz_blocks_single_block(comm):
    r"""Edge case: nblocks = 1, only A_0."""
    N = 4
    rng = np.random.default_rng(123)
    A0 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

    Ahat_petsc, _ = _build_block_toeplitz_petsc(comm, [A0], 1)
    Alst = _extract_toeplitz_blocks(Ahat_petsc, 1)

    block_dense = Alst[0].copy()
    block_dense.convert(PETSc.Mat.Type.DENSE)
    block_seq = res4py.distributed_to_sequential_matrix(block_dense)
    block_arr = block_seq.getDenseArray().copy()
    block_seq.destroy()
    block_dense.destroy()

    error = np.linalg.norm(block_arr - A0) / np.linalg.norm(A0)
    assert error < 1e-12

    Alst[0].destroy()
    Ahat_petsc.destroy()
