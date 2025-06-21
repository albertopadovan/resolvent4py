Quickstart
==========

Running the examples
--------------------

- Navigate to :code:`resolvent4py/examples/cgl/`
- Run

   .. code-block:: bash

      mpiexec -n 1 python3 -u generate_matrices.py

   to create and store the operators associated with the linearized complex 
   Ginzburg-Landau (CGL) equation. These will be saved into the :code:`data/`
   directory.

- Run:

   .. code-block:: bash

      mpiexec -n 3 python3 -u demonstrate_eigendecomposition.py

   to compute the CGL eigenvalues using 3 MPI processors. The results are stored
   in the :code:`results/` directory.


