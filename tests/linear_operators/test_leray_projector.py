r"""Tests for :class:`.LerayProjectorLinearOperator`.

The Leray projector is :math:`P = I - G (DG)^{-1} D`, mapping the velocity
space to itself. We build small (non-physical) divergence/gradient matrices
:math:`D` (``N_c x N_v``) and :math:`G` (``N_v x N_c``) with :math:`DG`
invertible, wrap them as resolvent4py operators, and check the action against
the dense numpy reference.
"""

import numpy as np
import scipy as sp
import resolvent4py as res4py
from .. import pytest_utils

N_V, N_C = 10, 5


def _build_leray(comm, seed=0):
    r"""Build a LerayProjectorLinearOperator and its dense numpy reference P.

    Returns ``(P, Pnp, owned)`` where ``owned`` is the list of objects the
    caller must destroy (operators no longer destroy their inputs)."""
    rng = np.random.default_rng(seed)
    Dnp = rng.standard_normal((N_C, N_V)) + 1j * rng.standard_normal((N_C, N_V))
    Gnp = rng.standard_normal((N_V, N_C)) + 1j * rng.standard_normal((N_V, N_C))
    DGnp = Dnp @ Gnp
    assert np.linalg.cond(DGnp) < 1e8, "DG must be invertible for the projector"
    Pnp = np.eye(N_V) - Gnp @ sp.linalg.inv(DGnp) @ Dnp

    Dm = pytest_utils.numpy_to_petsc(comm, Dnp)
    Gm = pytest_utils.numpy_to_petsc(comm, Gnp)
    DGm = pytest_utils.numpy_to_petsc(comm, DGnp)
    ksp = res4py.create_mumps_solver(DGm)

    D_op = res4py.linear_operators.MatrixLinearOperator(Dm)
    G_op = res4py.linear_operators.MatrixLinearOperator(Gm)
    DG_op = res4py.linear_operators.MatrixLinearOperator(DGm, ksp)
    P = res4py.linear_operators.LerayProjectorLinearOperator(D_op, DG_op, G_op)

    owned = [P, D_op, G_op, DG_op, ksp, Dm, Gm, DGm]
    return P, Pnp, owned


def test_leray_on_vectors(comm):
    r"""P x and P^* x against the dense reference."""
    P, Pnp, owned = _build_leray(comm)
    x, xnp = pytest_utils.generate_random_vector(comm, N_V)
    y = P.create_left_vector()
    error = [
        pytest_utils.compute_error_vector(comm, P.apply, x, y, Pnp.dot, xnp),
        pytest_utils.compute_error_vector(
            comm, P.apply_hermitian_transpose, x, y, Pnp.conj().T.dot, xnp
        ),
    ]
    x.destroy()
    y.destroy()
    for o in owned:
        o.destroy()
    assert np.linalg.norm(error) < 1e-10


def test_leray_on_bvs(comm):
    r"""P X and P^* X (BV variants) against the dense reference."""
    P, Pnp, owned = _build_leray(comm)
    X, Xnp = pytest_utils.generate_random_bv(comm, (N_V, 4))
    Y = P.create_left_bv(X.getSizes()[-1])
    error = [
        pytest_utils.compute_error_bv(comm, P.apply_mat, X, Y, Pnp.dot, Xnp),
        pytest_utils.compute_error_bv(
            comm, P.apply_hermitian_transpose_mat, X, Y, Pnp.conj().T.dot, Xnp
        ),
    ]
    X.destroy()
    Y.destroy()
    for o in owned:
        o.destroy()
    assert np.linalg.norm(error) < 1e-10


def test_leray_idempotency(comm):
    r"""A projector must satisfy :math:`P^2 = P`."""
    P, _, owned = _build_leray(comm)
    x, _ = pytest_utils.generate_random_vector(comm, N_V)
    Px = P.apply(x)
    PPx = P.apply(Px)
    Pxs = res4py.distributed_to_sequential_vector(Px)
    PPxs = res4py.distributed_to_sequential_vector(PPx)
    error = np.linalg.norm(
        Pxs.getArray() - PPxs.getArray()
    ) / np.linalg.norm(Pxs.getArray())
    Pxs.destroy()
    PPxs.destroy()
    x.destroy()
    Px.destroy()
    PPx.destroy()
    for o in owned:
        o.destroy()
    assert error < 1e-10
