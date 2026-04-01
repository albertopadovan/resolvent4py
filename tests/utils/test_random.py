import numpy as np
import resolvent4py as res4py


def test_generate_random_sparse_matrix_sizes(comm, square_matrix_size):
    r"""Test that generated sparse matrix has correct sizes"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    A = res4py.generate_random_petsc_sparse_matrix(
        ((Nl, N), (Nl, N)), int(0.3 * N * N)
    )
    sizes = A.getSizes()
    assert sizes[0][-1] == N
    assert sizes[-1][-1] == N
    A.destroy()


def test_generate_random_sparse_matrix_rectangular(comm, rectangular_matrix_size):
    r"""Test random sparse matrix generation with rectangular sizes"""
    Nr, Nc = rectangular_matrix_size
    Nrl = res4py.compute_local_size(Nr)
    Ncl = res4py.compute_local_size(Nc)
    A = res4py.generate_random_petsc_sparse_matrix(
        ((Nrl, Nr), (Ncl, Nc)), int(0.3 * Nr * Nc)
    )
    sizes = A.getSizes()
    assert sizes[0][-1] == Nr
    assert sizes[-1][-1] == Nc
    A.destroy()


def test_generate_random_sparse_matrix_complex(comm, square_matrix_size):
    r"""Test that complex flag produces complex-valued entries"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    A = res4py.generate_random_petsc_sparse_matrix(
        ((Nl, N), (Nl, N)), int(0.3 * N * N), complex=True
    )
    # Convert to dense and check for nonzero imaginary parts
    Ad = A.copy()
    from petsc4py import PETSc

    Ad.convert(PETSc.Mat.Type.DENSE)
    Ad_seq = res4py.distributed_to_sequential_matrix(Ad)
    array = Ad_seq.getDenseArray()
    has_imag = np.any(np.abs(array.imag) > 1e-15)
    Ad_seq.destroy()
    Ad.destroy()
    A.destroy()
    assert has_imag


def test_generate_random_sparse_matrix_real(comm, square_matrix_size):
    r"""Test that real flag produces real-valued entries (with zero imag)"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    A = res4py.generate_random_petsc_sparse_matrix(
        ((Nl, N), (Nl, N)), int(0.3 * N * N), complex=False
    )
    Ad = A.copy()
    from petsc4py import PETSc

    Ad.convert(PETSc.Mat.Type.DENSE)
    Ad_seq = res4py.distributed_to_sequential_matrix(Ad)
    array = Ad_seq.getDenseArray()
    max_imag = np.max(np.abs(array.imag))
    Ad_seq.destroy()
    Ad.destroy()
    A.destroy()
    assert max_imag < 1e-15


def test_generate_random_vector_sizes(comm, square_matrix_size):
    r"""Test that generated random vector has correct global size"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    x = res4py.generate_random_petsc_vector((Nl, N))
    assert x.getSize() == N
    x.destroy()


def test_generate_random_vector_complex(comm, square_matrix_size):
    r"""Test complex random vector has nonzero imaginary part"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    x = res4py.generate_random_petsc_vector((Nl, N), complex=True)
    xs = res4py.distributed_to_sequential_vector(x)
    has_imag = np.any(np.abs(xs.getArray().imag) > 1e-15)
    xs.destroy()
    x.destroy()
    assert has_imag


def test_generate_random_vector_real(comm, square_matrix_size):
    r"""Test real random vector has zero imaginary part"""
    N = square_matrix_size[0]
    Nl = res4py.compute_local_size(N)
    x = res4py.generate_random_petsc_vector((Nl, N), complex=False)
    xs = res4py.distributed_to_sequential_vector(x)
    max_imag = np.max(np.abs(xs.getArray().imag))
    xs.destroy()
    x.destroy()
    assert max_imag < 1e-15
