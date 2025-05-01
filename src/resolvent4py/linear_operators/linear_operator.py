import abc
from typing import Optional, Union

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.errors_helpers import raise_not_implemented_error
from ..utils.random_helpers import generate_random_petsc_vector
from ..utils.vec_helpers import (check_complex_conjugacy,
                                 enforce_complex_conjugacy)


class LinearOperator(metaclass=abc.ABCMeta):
    r"""
    Abstract base class for linear operators :math:`L`.

    :param comm: MPI communicator :code:`MPI.COMM_WORLD`
    :type comm: MPI.Comm
    :param name: name of the linear operator
    :type name: str
    :param dimensions: local and global sizes of the range and domain
        of the linear operator, :code:`[[local rows, global rows], 
        [local columns, global columns]]`
    :type dimensions: tuple[tuple[int, int], tuple[int, int]]
    :param nblocks: number of blocks (if the linear operator has block \
        structure)
    :type nblocks: Optional[int], default is None
    """

    def __init__(
        self: "LinearOperator",
        comm: MPI.Comm,
        name: str,
        dimensions: tuple[tuple[int, int], tuple[int, int]],
        nblocks: Optional[int] = None,
    ) -> None:
        self._comm = comm
        self._name = name
        self._dimensions = dimensions
        self._nblocks = nblocks
        self._real = self.check_if_real_valued()
        self._block_cc = (
            self.check_if_complex_conjugate_structure()
            if self._nblocks != None
            else None
        )

    def get_comm(self: "LinearOperator") -> MPI.Comm:
        r"""
        The MPI communicator

        :rtype: MPI.Comm
        """
        return self._comm

    def get_name(self: "LinearOperator") -> str:
        r"""
        The name of the linear operator

        :rtype: str
        """
        return self._name

    def get_dimensions(
        self: "LinearOperator",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        r"""
        The local and global dimensions of the linear operator

        :rtype: tuple[tuple[int, int], tuple[int, int]]
        """
        return self._dimensions

    def get_nblocks(self: "LinearOperator") -> Union[int, None]:
        r"""
        The number of blocks of the linear operator

        :rtype: Union[int, None]
        """
        return self._nblocks

    def create_right_vector(self: "LinearOperator") -> PETSc.Vec:
        r"""
        :return: a PETSc vector that :math:`L` can be multiplied against
        :rtype: PETSc.Vec
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[-1])
        vec.setType("standard")
        return vec

    def create_right_bv(self: "LinearOperator", ncols: int) -> SLEPc.BV:
        r"""
        :param ncols: number of columns in the BV
        :type ncols: int

        :return: a SLEPc BV that :math:`L` can be multiplied against
        :rtype: SLEPc.BV
        """
        bv = SLEPc.BV().create(comm=self._comm)
        bv.setSizes(self._dimensions[-1], ncols)
        bv.setType("mat")
        return bv

    def create_left_vector(self: "LinearOperator") -> PETSc.Vec:
        r"""
        :return: a PETSc vector where :math:`Lx` can be stored into
        :rtype: PETSc.Vec
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[0])
        vec.setType("standard")
        return vec

    def create_left_bv(self: "LinearOperator", ncols: int) -> SLEPc.BV:
        r"""
        :param ncols: number of columns in the BV
        :param type: int

        :return: a SLEPc BV where :math:`LX` can be stored into
        :rtype: SLEPc.BV
        """
        bv = SLEPc.BV().create(comm=self._comm)
        bv.setSizes(self._dimensions[0], ncols)
        bv.setType("mat")
        return bv

    def check_if_real_valued(self: "LinearOperator") -> bool:
        r"""
        :return: :code:`True` if the linear operator is real-valued,
            :code:`False` otherwise
        :rtype: bool
        """
        sizes = self._dimensions[-1]
        x = generate_random_petsc_vector(self._comm, sizes)
        Lx = self.apply(x)
        Lxai = Lx.getArray().imag
        norm = np.sqrt(sum(self._comm.allgather(np.linalg.norm(Lxai) ** 2)))
        result = True if norm <= 1e-14 else False
        x.destroy()
        Lx.destroy()
        return result

    def check_if_complex_conjugate_structure(self: "LinearOperator") -> bool:
        r"""
        Given a vector

        .. math::
            x = \left(\ldots,x_{-1},x_{0},x_{1},\ldots\right)

        with vector-valued entries that satisfy \
        :math:`x_{-i} = \overline{x_i}`, check if the vector :math:`Lx` \
        satisfies :math:`(Lx)_{-i}=\overline{(Lx)_{i}}`. (Here, the \
        overline denote complex conjugation.)

        :return: :code:`True` if the linear operator has complex-conjugate
            structure, :code:`False` otherwise.
        :rtype: bool
        """
        x = generate_random_petsc_vector(self._comm, self._dimensions[-1])
        enforce_complex_conjugacy(self._comm, x, self._nblocks)
        cc_x = check_complex_conjugacy(self._comm, x, self._nblocks)
        if cc_x == False:
            raise ValueError(
                f"Error from {self.get_name()}.check_if_complex_conjugate"
                f"_structure(): complex conjugacy was not enforced "
                f"appropriately."
            )
        Lx = self.apply(x)
        result = check_complex_conjugacy(self._comm, Lx, self._nblocks)
        x.destroy()
        Lx.destroy()
        return result

    # Methods that must be implemented by subclasses
    @abc.abstractmethod
    def apply(
        self: "LinearOperator", x: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute :math:`y = Lx`.

        :param x: a PETSc vector
        :type x: PETSc.Vec
        :param y: a PETSc vector to store the result
        :type y: Optional[PETSc.Vec], default is None

        :rtype: PETSc.Vec
        """

    @abc.abstractmethod
    def apply_mat(
        self: "LinearOperator", X: SLEPc.BV, Y: Optional[SLEPc.BV] = None
    ) -> SLEPc.BV:
        r"""
        Compute :math:`Y = LX`

        :param X: a SLEPc BV
        :type X: SLEPc.BV
        :param Y: a SLEPc BV to store the result
        :type Y: Optional[SLEPc.BV], default is None

        :rtype: SLEPc.BV
        """

    @abc.abstractmethod
    def destroy(self: "LinearOperator") -> None:
        r"""
        Destroy the PETSc and SLEPc objects associated with :math:`L`
        """

    # Methods that don't necessarily need to be implemented by subclasses
    @raise_not_implemented_error
    def apply_hermitian_transpose(
        self: "LinearOperator", x: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute :math:`y = L^*x`.

        :param x: a PETSc vector
        :type x: PETSc.Vec
        :param y: a PETSc vector to store the result
        :type y: Optional[PETSc.Vec], default is None

        :rtype: PETSc.Vec
        """

    @raise_not_implemented_error
    def apply_hermitian_transpose_mat(
        self: "LinearOperator", X: SLEPc.BV, Y: Optional[SLEPc.BV] = None
    ) -> SLEPc.BV:
        r"""
        Compute :math:`Y = L^*X`

        :param X: a SLEPc BV
        :type X: SLEPc.BV
        :param Y: a SLEPc BV to store the result
        :type Y: Optional[SLEPc.BV], default is None

        :rtype: SLEPc.BV
        """

    @raise_not_implemented_error
    def solve(
        self: "LinearOperator", x: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute :math:`y = L^{-1}x`.

        :param x: a PETSc vector
        :type x: PETSc.Vec
        :param y: a PETSc vector to store the result
        :type y: Optional[PETSc.Vec], default is None

        :rtype: PETSc.Vec
        """

    @raise_not_implemented_error
    def solve_mat(
        self: "LinearOperator", X: SLEPc.BV, Y: Optional[SLEPc.BV] = None
    ) -> SLEPc.BV:
        r"""
        Compute :math:`Y = L^{-1}X`

        :param X: a SLEPc BV
        :type X: SLEPc.BV
        :param Y: a SLEPc BV to store the result
        :type Y: Optional[SLEPc.BV], default is None

        :rtype: SLEPc.BV
        """

    @raise_not_implemented_error
    def solve_hermitian_transpose(
        self: "LinearOperator", x: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute :math:`y = L^{-*}x`.

        :param x: a PETSc vector
        :type x: PETSc.Vec
        :param y: a PETSc vector to store the result
        :type y: Optional[PETSc.Vec], default is None

        :rtype: PETSc.Vec
        """

    @raise_not_implemented_error
    def solve_hermitian_transpose_mat(
        self: "LinearOperator", X: SLEPc.BV, Y: Optional[SLEPc.BV] = None
    ) -> SLEPc.BV:
        r"""
        Compute :math:`Y = L^{-*}X`

        :param X: a SLEPc BV
        :type X: SLEPc.BV
        :param Y: a SLEPc BV to store the result
        :type Y: Optional[SLEPc.BV], default is None

        :rtype: SLEPc.BV
        """
