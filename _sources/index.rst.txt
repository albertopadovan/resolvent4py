.. Resolvent4py documentation master file, created by
   sphinx-quickstart on Mon Oct  7 23:00:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Resolvent4py
===========================

Resolvent4py is a petsc4py- and slepc4py-based toolbox to perform 
analysis, model reduction and control of high-dimensional linear 
dynamical systems. 
The goal of this project is to provide users with a friendly python-like
experience, while also leveraging the high-performance and parallel-computing
capabilities of the PETSc and SLEPc library.
Current functionalities include:

- Right and left eigendecomposition 
- Resolvent analysis (algebraic and using time-stepping techniques)
- Harmonic resolvent analysis
- Balanced truncation (time-invariant)


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





