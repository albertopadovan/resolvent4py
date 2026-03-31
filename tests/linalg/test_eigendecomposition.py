from petsc4py import PETSc
import scipy as sp
import numpy as np
import resolvent4py as res4py


def _build_resolvent_operator(comm, Apetsc, omega):
    """Build L = (1j*omega*I - A) and its solver."""
    Id = res4py.create_AIJ_identity(comm, Apetsc.getSizes())
    Id.scale(1j * omega)
    Id.convert(PETSc.Mat.Type.MPIAIJ)
    Id.axpy(-1.0, Apetsc)
    ksp = res4py.create_mumps_solver(Id)
    return res4py.linear_operators.MatrixLinearOperator(Id, ksp)


def test_eigendecomposition(comm, square_random_matrix):
    """Test that eigenvalues from Arnoldi match scipy's eigenvalues of A."""
    Apetsc, Apython = square_random_matrix
    omega = 20.0
    r = 10

    linop = _build_resolvent_operator(comm, Apetsc, omega)
    krylov_dim = linop.get_dimensions()[0][-1] - 1
    r = np.min([r, krylov_dim - 1])

    # Compute eigenvalues of A via shift-invert Arnoldi
    D, _ = res4py.linalg.eig(
        linop, linop.solve, krylov_dim, r,
        lambda x: 1j * omega - 1 / x, 0,
    )
    D = np.diag(D)

    # Compare against scipy reference
    ev, _ = sp.linalg.eig(Apython)
    ev_sorted = np.array([ev[np.argmin(np.abs(ev - d))] for d in D])
    error = 100 * np.max(np.abs(ev_sorted - D) / np.abs(ev_sorted))
    assert error < 5e-1


def test_match_right_and_left_eigenvectors(comm, square_random_matrix):
    """Test biorthogonalization: W^* V = I and W^* A V = diag(evals)."""
    Apetsc, _ = square_random_matrix
    Aop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    omega = 20.0
    s = 1j * omega
    r = 3

    linop = _build_resolvent_operator(comm, Apetsc, omega)
    krylov_dim = linop.get_dimensions()[0][-1] - 1
    r = np.min([r, krylov_dim])

    # Right eigendecomposition of A via shift-invert
    Dv, V = res4py.linalg.eig(
        linop, linop.solve, krylov_dim, r, lambda x: s - 1 / x, 0,
    )
    error = res4py.linalg.check_eig_convergence(Aop.apply, Dv, V)
    assert np.linalg.norm(error, ord=1) < 1e-7

    # Left eigendecomposition of A via shift-invert
    Dw, W = res4py.linalg.eig(
        linop, linop.solve_hermitian_transpose, krylov_dim, r,
        lambda x: np.conj(s) - 1 / x, 0,
    )
    error = res4py.linalg.check_eig_convergence(
        Aop.apply_hermitian_transpose, Dw, W,
    )
    assert np.linalg.norm(error, ord=1) < 1e-7

    # Biorthogonalize (conjugation of Dw handled internally)
    V, W, Dv, Dw = res4py.linalg.match_right_and_left_eigenvectors(
        V, W, Dv, Dw,
    )

    # Eigenvalues sorted by descending real part
    evals = np.diag(Dv)
    assert np.all(np.diff(evals.real) <= 0)

    # W^* V = I
    WtV = V.dot(W)
    identity_error = np.max(np.abs(WtV.getDenseArray() - np.eye(r)))
    assert identity_error < 1e-12

    # Dv = Dw (eigenvalues matched)
    eval_error = np.max(np.abs(np.diag(Dv) - np.conj(np.diag(Dw))))
    assert eval_error < 1e-5

    # W^* A V = diag(evals)
    AV = Aop.apply_mat(V)
    WtAV = AV.dot(W)
    projection_error = np.linalg.norm(WtAV.getDenseArray() - np.diag(np.diag(Dv)))
    assert projection_error < 1e-6
