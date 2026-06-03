Spectral Submanifolds
=====================

Polynomial-expansion reduction onto a chosen spectral subspace of a
quadratic dynamical system

.. math::

    \dot{q} \;=\; A\, q \;+\; B(q, q),

where :math:`A` is a (possibly time-periodic) linear operator and
:math:`B` is a symmetric bilinear form.  The
:class:`SpectralSubmanifold` parametrises the invariant manifold
tangent to the chosen master eigenspace as
:math:`q = P(s)` with reduced dynamics
:math:`\dot{s} = \Lambda s + g(s)`, where :math:`P` and :math:`g` are
polynomials of user-specified degree.

For time-periodic :math:`A(t)`, the
:class:`PeriodicDifferentialEquation` mixin (inserted by the factory
in :meth:`DifferentialEquation.__new__` when ``periodic_diffeq`` is
provided) lifts the per-time-instant linear and bilinear operators
to their harmonic-balanced (HB) forms via the IFFT → per-instant →
FFT pipeline.

Differential Equation
---------------------
.. automodule:: resolvent4py.spectral_submanifold.differential_equation
   :members:
   :show-inheritance:

Spectral Submanifold
--------------------
.. automodule:: resolvent4py.spectral_submanifold.spectral_submanifold
   :members:
   :show-inheritance:

Spectral Submanifold ROM
------------------------
.. automodule:: resolvent4py.spectral_submanifold.spectral_submanifold_rom
   :members:
   :show-inheritance:
