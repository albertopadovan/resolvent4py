from __future__ import annotations

import abc
from typing import Optional, Tuple

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np


class DifferentialEquation(metaclass=abc.ABCMeta):
    r"""
    Abstract base class for dynamical systems of the form

    .. math::

        \dot{q} = A\,q + B(q, q),

    where :math:`A` is a linear operator and :math:`B` is a symmetric
    bilinear form representing the quadratic nonlinearity.

    Subclasses must implement :meth:`evaluate_linear_term`,
    :meth:`evaluate_quadratic_term`, and :meth:`solve_linear_system`.

    If ``periodic_diffeq`` is provided at construction, this class
    becomes a factory: it returns an instance of a dynamically
    generated subclass that mixes :class:`PeriodicDifferentialEquation`
    in front of the requested subclass in the MRO.  The user's
    :meth:`evaluate_quadratic_term` is then treated as the
    *per-time-instant* bilinear, while the periodic mixin handles the
    harmonic-balanced (IFFT → per-t → FFT) scaffolding around it.
    """

    def __new__(cls, *args, periodic_diffeq=None, **kwargs):
        # Wrap with PeriodicDifferentialEquation only when:
        #   (a) the caller is NOT already a periodic class, AND
        #   (b) periodic_diffeq is provided.
        if (
            periodic_diffeq is not None
            and not issubclass(cls, PeriodicDifferentialEquation)
        ):
            DynCls = type(
                cls.__name__,
                (PeriodicDifferentialEquation, cls),
                {},
            )
            return object.__new__(DynCls)
        return super().__new__(cls)

    def __init__(
        self, comm: PETSc.Comm, name: str,
        state_dim: Tuple[int, int], poly_deg: int,
        periodic_diffeq: Tuple[np.ndarray, np.ndarray, bool] | None = None,
    ) -> None:
        r"""
        :param comm: MPI communicator
        :type comm: PETSc.Comm
        :param name: identifier for this system
        :type name: str
        :param state_dim: state-space dimensions as
            ``(local_size, global_size)``
        :type state_dim: Tuple[int, int]
        :param poly_deg: degree of the polynomial nonlinearity
        :type poly_deg: int
        """
        self._comm = comm
        self._name = name
        self._state_dim = state_dim
        self._poly_deg = poly_deg

    def get_name(self) -> str:
        r"""Return the system name."""
        return self._name

    def get_comm(self) -> PETSc.Comm:
        r"""Return the MPI communicator."""
        return self._comm

    def get_state_dimension(self) -> Tuple[int, int]:
        r"""Return ``(local_size, global_size)`` of the state space."""
        return self._state_dim

    def get_poly_degree(self) -> int:
        r"""Return the degree of the polynomial nonlinearity."""
        return self._poly_deg
    

    @abc.abstractmethod
    def evaluate_linear_term(
        self, t: float, q: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the linear part of the RHS at time :math:`t`:
        :math:`A(t)\, q`.  For autonomous systems the ``t`` argument
        may be ignored.

        :param t: time
        :type t: float
        :param q: state vector
        :type q: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`A(t)\, q`
        :rtype: PETSc.Vec
        """
        ...

    @abc.abstractmethod
    def evaluate_quadratic_term(
        self, t: float, q1: PETSc.Vec, q2: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the quadratic bilinear term at time :math:`t`:
        :math:`B(t;\, q_1, q_2)`.  Must satisfy
        :math:`B(t;\, q_1, q_2) = B(t;\, q_2, q_1)`.  For autonomous
        systems the ``t`` argument may be ignored.

        :param t: time
        :type t: float
        :param q1: first state vector
        :type q1: PETSc.Vec
        :param q2: second state vector
        :type q2: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`B(t;\, q_1, q_2)`
        :rtype: PETSc.Vec
        """
        ...

    @abc.abstractmethod
    def solve_linear_system(
        self, s: complex, b: PETSc.Vec, x: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Solve :math:`(s I - A)\,x = b`.

        :param s: shift parameter
        :type s: complex
        :param b: right-hand side vector
        :type b: PETSc.Vec
        :param x: optional output vector (reused if provided)
        :type x: Optional[PETSc.Vec]

        :return: solution :math:`x`
        :rtype: PETSc.Vec
        """
        ...

    def evaluate_dynamics(
        self, t: float, q: PETSc.Vec, y: Optional[PETSc.Vec] = None
    ) -> PETSc.Vec:
        r"""
        Compute the full RHS at time :math:`t`:
        :math:`A(t)\, q + B(t;\, q, q)`.
        Override if the system has additional terms.

        :param t: time
        :type t: float
        :param q: state vector
        :type q: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: result of :math:`A(t)\, q + B(t;\, q, q)`
        :rtype: PETSc.Vec
        """
        y = self.evaluate_linear_term(t, q, y)
        Bqq = self.evaluate_quadratic_term(t, q, q)
        y.axpy(1.0, Bqq)
        Bqq.destroy()
        return y

from ..utils.vector import reshape_harmonic_balanced_vector_into_bv
from ..utils.bv import reshape_bv_into_harmonic_balanced_vector, bv_slice
from ..utils.time_stepping import fft, ifft

class PeriodicDifferentialEquation(DifferentialEquation, metaclass=abc.ABCMeta):
    r"""
    Cooperative mixin that lifts the time-domain linear and bilinear
    operators of a :class:`DifferentialEquation` subclass to their
    harmonic-balanced (HB) forms.

    Given a (possibly time-periodic) quadratic system

    .. math::

        \dot{q} = A(t)\,q + B(t;\, q, q),

    the perturbation about a :math:`T`-periodic orbit admits the HB
    representation

    .. math::

        \hat{q} = \big(\hat{q}_{-n_f}, \ldots, \hat{q}_0, \ldots,
        \hat{q}_{n_f}\big)^T,
        \qquad
        q(t) = \sum_{k=-n_f}^{n_f} \hat{q}_k\, e^{i k \omega t},

    in which:

    - the linear part becomes the harmonic resolvent generator
      :math:`\mathcal{L} = -\mathrm{diag}(i k \omega I) + \mathcal{A}`
      whose action on :math:`\hat{q}` satisfies

      .. math::

          (\mathcal{L}\, q)_k
          = \widehat{A(t)\, q(t)}_k - i \omega_k\, q_k;

    - the bilinear part becomes the discrete convolution

      .. math::

          \big(\widehat{B(q_1, q_2)}\big)_k
          = \frac{1}{n_t}\sum_{i=0}^{n_t-1}
              B\!\big(t_i;\, q_1(t_i),\, q_2(t_i)\big)\,
              e^{-i k \omega t_i}.

    This class implements both pipelines (IFFT → per-time-instant
    operator → FFT) on top of a user-supplied
    :meth:`DifferentialEquation.evaluate_linear_term` and
    :meth:`DifferentialEquation.evaluate_quadratic_term` that return
    the time-domain operators at a single time instant.  The per-time
    calls are made via ``super().evaluate_linear_term`` /
    ``super().evaluate_quadratic_term``, which under the MRO dispatch
    to the user's class.

    The mixin is not meant to be instantiated or subclassed directly:
    the factory in :meth:`DifferentialEquation.__new__` inserts it
    ahead of the user's subclass in the MRO when ``periodic_diffeq``
    is provided to the constructor.
    """

    def __init__(
        self,
        *args,
        periodic_diffeq: Tuple[np.ndarray, np.ndarray, bool],
        **kwargs,
    ) -> None:
        r"""
        :param periodic_diffeq: ``(omegas, time, is_period_doubling)``.

            - ``omegas``: 1-D array of *non-negative* angular
              frequencies for a real-valued system (the negative half
              is recovered by conjugate symmetry), or the full
              two-sided spectrum otherwise.  Must include the
              fundamental :math:`\omega`.
            - ``time``: 1-D array of physical-time samples, uniformly
              spaced on ``[0, T)`` with
              ``T = time[-1] + (time[1] - time[0])``.  The Nyquist
              condition ``2*pi / fund_frequency == T`` is checked
              here.
            - ``is_period_doubling``: ``bool`` flag for
              period-doubling (sub-harmonic) cases.
        :type periodic_diffeq: Tuple[np.ndarray, np.ndarray, bool]

        :raises ValueError: if the period implied by the frequency
            vector does not match ``time[-1] + dt``.
        """
        super().__init__(*args, **kwargs)

        self._omegas = periodic_diffeq[0].copy()
        self._time = periodic_diffeq[1].copy()
        self._is_period_doubling = periodic_diffeq[2]

        self._nt = len(self._time)
        self._real_bflow = False
        if np.min(self._omegas) == 0:
            self._omegas = np.concatenate(
                (np.flipud(-self._omegas[1:]), self._omegas)
            )
            self._real_bflow = True
        self._nblocks = len(self._omegas)
        fund_freq = self._omegas[int((self._nblocks - 1) / 2) + 1]

        T = self._time[-1] + (self._time[1] - self._time[0])
        if np.abs(2 * np.pi / fund_freq - T) > 1e-10:
            raise ValueError(
                f"Mismatch between time vector and frequency vector. Make sure "
                f"that 2 pi / fund_frequency = time[-1] + dt."
            )

        self._Q_freqs, self._Q_time = [], []
        for _ in range(self._poly_deg + 1):
            Qf = SLEPc.BV().create(comm=self._comm)
            Qf.setSizes(self._state_dim, self._nblocks)
            Qf.setType("mat")
            self._Q_freqs.append(Qf)

            Qt = SLEPc.BV().create(comm=self._comm)
            Qt.setSizes(self._state_dim, self._nt)
            Qt.setType("mat")
            self._Q_time.append(Qt)

        # if self._is_period_doubling:
        #     nf = int((len(self._omegas) - 1) / 2)
        #     if nf % 2 == 0:
        #         self.idces_T = np.arange(1, self._nblocks, 2)
        #         self.idces_2T = np.arange(0, self._nblocks + 1, 2)
        #     else:
        #         self.idces_T = np.arange(0, self._nblocks + 1, 2)
        #         self.idces_2T = np.arange(1, self._nblocks, 2)
            
        #     self._Q_freqs_T = SLEPc.BV().create(comm=self._comm)
        #     self._Q_freqs_T.setSizes(self._state_dim, len(self._idces_T))
        #     self._Q_freqs_T.setType("mat")

        #     self._Q_freqs_2T = SLEPc.BV().create(comm=self._comm)
        #     self._Q_freqs_2T.setSizes(self._state_dim, len(self._idces_2T))
        #     self._Q_freqs_2T.setType("mat")


    def evaluate_linear_term(
        self,
        t: float,
        q: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Apply the harmonic resolvent generator

        .. math::

            \mathcal{L} = -\mathrm{diag}(i k \omega I) + \mathcal{A}

        to an HB state vector.  Reshapes ``q`` into BV form,
        reconstructs it in the time domain at every sample
        :math:`t_i` via :func:`ifft`, calls the user's per-time-instant
        linear operator at each :math:`t_i` through
        ``super().evaluate_linear_term(t_i, qk, yk)``, projects back
        onto the harmonic basis via :func:`fft`, and subtracts the
        time-derivative term :math:`i \omega_k\, q_k` from each
        Fourier coefficient to obtain

        .. math::

            (\mathcal{L}\, q)_k
            = \widehat{A(t)\, q(t)}_k - i \omega_k\, q_k.

        The per-time call resolves through the MRO inserted by
        :meth:`DifferentialEquation.__new__`: from this mixin,
        ``super()`` points to the user's concrete subclass, whose
        :meth:`evaluate_linear_term` is the time-domain operator
        :math:`A(t_i)\, q(t_i)`.

        :param t: accepted for signature consistency with
            :meth:`DifferentialEquation.evaluate_linear_term`, but
            unused — there is no single time at the HB level; the
            internal loop supplies ``self._time[k]`` to each
            per-time-instant call instead.
        :type t: float
        :param q: HB state vector of size ``state_dim * nblocks``
        :type q: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: HB representation of :math:`\mathcal{L}\, q`
        :rtype: PETSc.Vec
        """
        # Temporal reconstruction of the harmonic-balanced vectors
        reshape_harmonic_balanced_vector_into_bv(q, self._nblocks, self._Q_freqs[0])
        for i in range(self._nt):
            q = self._Q_time[0].getColumn(i)
            ifft(self._Q_freqs[0], q, self._omegas, self._time[i])
            self._Q_time[0].restoreColumn(i, q)

        # Per-time-instant bilinear, dispatched via MRO to the user's class
        for k in range(self._nt):
            qk = self._Q_time[0].getColumn(k)
            yk = self._Q_time[-1].getColumn(k)
            yk = super().evaluate_linear_term(self._time[k], qk, yk)
            self._Q_time[-1].restoreColumn(k, yk)
            self._Q_time[0].restoreColumn(k, qk)

        # FFT back into the frequency domain and perform frequency shift (i.e., time derivative 
        # in the frequency domain). 
        self._Q_freqs[-1] = fft(self._Q_time[-1], self._Q_freqs[-1], False, True)
        for k in range (self._nblocks):
            qk = self._Q_freqs[-1].getColumn(k)
            qk_in = self._Q_freqs[0].getColumn(k)
            qk.axpy(-1j * self._omegas[k], qk_in)
            self._Q_freqs[0].restoreColumn(k, qk_in)
            self._Q_freqs[-1].restoreColumn(k, qk)
        return reshape_bv_into_harmonic_balanced_vector(self._Q_freqs[-1], y)
    
    def evaluate_quadratic_term(
        self,
        t: float,
        q1: PETSc.Vec,
        q2: PETSc.Vec,
        y: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        r"""
        Evaluate the bilinear term in the harmonic-balanced
        representation.

        Reshapes ``q1`` and ``q2`` into BV form, reconstructs them in
        the time domain at every sample :math:`t_i` via :func:`ifft`,
        calls the user's per-time-instant bilinear at each
        :math:`t_i` through
        ``super().evaluate_quadratic_term(t_i, qk1, qk2, yk)``, and
        projects the result back onto the harmonic basis via
        :func:`fft`.

        The per-time call resolves through the MRO inserted by
        :meth:`DifferentialEquation.__new__`: from this mixin,
        ``super()`` points to the user's concrete subclass, whose
        :meth:`evaluate_quadratic_term` is the *time-domain* bilinear
        :math:`B(t_i;\, q_1(t_i), q_2(t_i))`.

        :param t: accepted for signature consistency with
            :meth:`DifferentialEquation.evaluate_quadratic_term`, but
            unused — there is no single time at the HB level; the
            internal loop supplies ``self._time[k]`` to each
            per-time-instant call instead.
        :type t: float
        :param q1: first HB state vector of size
            ``state_dim * nblocks``
        :type q1: PETSc.Vec
        :param q2: second HB state vector of size
            ``state_dim * nblocks``
        :type q2: PETSc.Vec
        :param y: optional output vector (reused if provided)
        :type y: Optional[PETSc.Vec]

        :return: HB representation of :math:`B(q_1, q_2)`
        :rtype: PETSc.Vec
        """
        # Temporal reconstruction of the harmonic-balanced vectors
        qlst = [q1, q2]
        for j in range(len(qlst)):
            reshape_harmonic_balanced_vector_into_bv(qlst[j], self._nblocks, self._Q_freqs[j])
            for i in range(self._nt):
                q = self._Q_time[j].getColumn(i)
                ifft(self._Q_freqs[j], q, self._omegas, self._time[i])
                self._Q_time[j].restoreColumn(i, q)

        # Per-time-instant bilinear, dispatched via MRO to the user's class
        for k in range(self._nt):
            qk1 = self._Q_time[0].getColumn(k)
            qk2 = self._Q_time[1].getColumn(k)
            yk = self._Q_time[-1].getColumn(k)
            yk = super().evaluate_quadratic_term(self._time[k], qk1, qk2, yk)

            self._Q_time[-1].restoreColumn(k, yk)
            self._Q_time[0].restoreColumn(k, qk1)
            self._Q_time[1].restoreColumn(k, qk2)

        # FFT back into the frequency domain
        self._Q_freqs[-1] = fft(self._Q_time[-1], self._Q_freqs[-1], False, True)
        return reshape_bv_into_harmonic_balanced_vector(self._Q_freqs[-1], y)


    # def solve_linear_system(self, s, b, x = None):
    #     reshape_harmonic_balanced_vector_into_bv(b, self._nblocks, self._Q_freqs[0])
    #     bv_slice(self._Q_freqs[0], self.idces_T, self._Q_freqs_T)
    #     bv_slice(self._Q_freqs[0], self.idces_2T, self._Q_freqs_2T)
    #     bT = reshape_bv_into_harmonic_balanced_vector(self._Q_freqs_T)
    #     b2T = reshape_bv_into_harmonic_balanced_vector(self._Q_freqs_T)

    #     return super().solve_linear_system(s, b, x)

        
        