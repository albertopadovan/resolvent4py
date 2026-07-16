.. Resolvent4py documentation master file, created by
   sphinx-quickstart on Mon Oct  7 23:00:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Resolvent4py
===========================

Resolvent4py is a petsc4py- and slepc4py-based toolbox to perform
analysis, model reduction and control of high-dimensional linear and
weakly-nonlinear (quadratic) dynamical systems.  The goal of this
project is to provide users with a friendly python-like experience,
while also leveraging the high-performance and parallel-computing
capabilities of the PETSc and SLEPc library.

The library is organised around a single
:class:`~resolvent4py.linear_operators.linear_operator.LinearOperator` abstraction —
matrices, low-rank factorisations, products, projections,
shift-and-scale composites, the propagator of a (possibly time-
periodic) ODE, and explicit time-periodic operators with Fourier
coefficients all share the same ``apply`` / ``apply_mat`` / ``solve``
interface, and compose freely.  Every analysis routine is written
against that interface, so anything in ``linalg`` /
``model_reduction`` works on any operator without knowing how it is
stored.

Current functionalities include:

- Right and left eigendecomposition (shift-invert Arnoldi via SLEPc,
  with biorthogonalisation of the recovered modes)
- Frequency-domain resolvent analysis (algebraic and time-stepping
  / RSVD-dt for large systems)
- Harmonic resolvent analysis of time-periodic systems, including
  end-to-end cross-checks against post-transient time integration
- Frequency-domain balanced truncation for time-invariant systems
- Time integration of linear time-invariant and time-periodic
  systems with periodic forcing (RK2 / RK3, forward + adjoint), with
  recursive ``set_evaluation_time`` propagation through composite
  operators
- Post-transient response of forced periodic ODEs, with two
  strategies: a cheap iteration when the shifted Floquet spectrum is
  stable, and a GMRES shoot-and-solve
  :math:`(I - \Phi(T,0))\,x(0) = \int_0^T \Phi(T,\tau)\,f(\tau)\,d\tau`
  that works for any shift where :math:`I - \Phi` is non-singular


If you use resolvent4py in your work, please cite the following paper
(see `here <https://www.sciencedirect.com/science/article/pii/S2352711025002523>`_ for the 
open access pdf):

   .. code-block::

      @article{PADOVAN2025102286,
      title = {Resolvent4py: A parallel Python package for analysis, model reduction and control of large-scale linear systems},
      journal = {SoftwareX},
      volume = {31},
      pages = {102286},
      year = {2025},
      issn = {2352-7110},
      doi = {https://doi.org/10.1016/j.softx.2025.102286},
      url = {https://www.sciencedirect.com/science/article/pii/S2352711025002523},
      author = {Alberto Padovan and Vishal Anantharaman and Clarence W. Rowley and Blaine Vollmer and Tim Colonius and Daniel J. Bodony},
      }


Installation Instructions
=========================

Please see `README <https://github.com/albertopadovan/resolvent4py/blob/main/README.md>`_ on GitHub.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   api-reference.rst
   refs.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   auto_examples/cgl/index
   auto_examples/toy_model/index





