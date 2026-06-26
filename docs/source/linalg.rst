Linalg
======

Eigendecomposition
------------------
.. automodule:: resolvent4py.linalg.eigendecomposition
   :members:

Randomized SVD
--------------
.. automodule:: resolvent4py.linalg.randomized_svd
   :members:

Resolvent Analysis via Time Stepping
------------------------------------
.. automodule:: resolvent4py.linalg.resolvent_analysis_time_stepping
   :members:

OWNS Resolvent Analysis
-----------------------
To use OWNS, provide a station-wise object with ``comm``, ``n_x``, ``dx``,
and ``get_station(j)``. Each station must return an
``OWNSStation`` containing the characteristic operators, input/output maps,
weights, and recursion parameters.

.. automodule:: resolvent4py.linalg.resolvent_analysis_owns
   :members:
