import numpy as np
import resolvent4py as res4py
from .. import pytest_utils
from resolvent4py.linear_operators import PetscPythonLinearOperator


def test_matrix_on_vectors(comm, square_random_matrix):
    r"""Test PetscPythonLinearOperator on vectors"""
    Apetsc, Apython = square_random_matrix
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc)
    Ashell = PetscPythonLinearOperator.create_shell(linop)
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

    actions_python = [Apython.dot, Apython.conj().T.dot]
    actions_petsc = [Ashell.mult, Ashell.multHermitian]

    y = linop.create_left_vector()
    print("Size = %d" % y.getSize())
    error_vec = [
        pytest_utils.compute_error_vector_shell_operator(
            comm, actions_petsc[i], x, y, actions_python[i], xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    assert error < 1e-10
