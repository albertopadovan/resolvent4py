# Resolvent4py

[![Tests](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml/badge.svg)](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Size](https://img.shields.io/github/languages/code-size/albertopadovan/resolvent4py.svg)](https://github.com/albertopadovan/resolvent4py)


`resolvent4py` is a parallel Python toolbox to perform
analysis, model reduction and control of high-dimensional linear (and
weakly-nonlinear, quadratic) dynamical systems.
It relies on `mpi4py` for multi-processing parallelism, and it leverages
the functionalities and data structures provided by `petsc4py` and `slepc4py`.
The goal of this project is to provide users with a friendly python-like
experience, while also leveraging the high-performance and parallel-computing
capabilities of the PETSc and SLEPc libraries.

The core of the package is an abstract class, called `LinearOperator`, which
serves as a blueprint for user-defined child classes that can be used to
define any linear operator.  Every analysis routine is written against
this interface, so anything under `resolvent4py/linalg` or
`resolvent4py/model_reduction` works on any operator without knowing how
it is stored.  `resolvent4py` currently ships with the following
`LinearOperator` subclasses:

- `MatrixLinearOperator` — wraps a PETSc matrix
- `LowRankLinearOperator` — `U Σ V*` low-rank factorization
- `LowRankUpdatedLinearOperator` — `A + BKC*` with automatic Woodbury solve
- `ProductLinearOperator` — composition of any number of operators
- `ProjectionLinearOperator` — oblique projection built from two BVs
- `ShiftAndScaleLinearOperator` — `α L + σ I`, composable
- `TimePeriodicMatrixLinearOperator` — `A(t) = Σₖ Aₖ eⁱωₖᵗ` for time-periodic systems
- `PropagatorLinearOperator` — one-period propagator of a (possibly time-varying) ODE
- `LerayProjectorLinearOperator` — divergence-free projector `P = I − G(DG)⁻¹D`
- `IncompressibleNavierStokesLinearOperator` — Leray-projected NS resolvent from COO matrices
- `PetscPythonLinearOperator` — wraps any `LinearOperator` as a PETSc shell matrix for use with native PETSc solvers

Once a linear operator is instantiated, `resolvent4py` currently allows for
several analyses, including:

- Right and left eigendecomposition via shift-invert Arnoldi (SLEPc), with
  biorthogonalisation of the recovered modes.
- Randomized SVD, both algebraic (`randomized_svd`) and time-stepping
  (`resolvent_analysis_rsvd_dt`, RSVD-dt) for very large systems.
- Frequency-domain resolvent analysis (algebraic or RSVD-dt).
- Harmonic resolvent analysis of time-periodic systems via algebraic
  randomized SVD, including end-to-end cross-checks against post-transient
  time integration.
- Balanced truncation for time-invariant linear systems using frequency-domain
  Gramians.
- Time integration of linear time-invariant and time-periodic systems with
  periodic forcing (RK2 / RK3, forward + adjoint), with recursive
  `set_evaluation_time` propagation through composite operators.
- Post-transient (periodic steady-state) response of forced periodic ODEs,
  with two strategies: a cheap iteration when the shifted Floquet spectrum is
  stable, and a GMRES shoot-and-solve
  `(I − Φ(T,0)) x(0) = ∫₀ᵀ Φ(T,τ) f(τ) dτ` that works for any shift where
  `I − Φ` is non-singular.

Additional functionalities (found in `resolvent4py/utils`) and available
to the user through the `resolvent4py` namespace are:

- Parallel MPI-I/O readers/writers for PETSc vectors, matrices, and SLEPc BVs
  (including harmonic-balanced block-Toeplitz assemblers).
- Communicator helpers: scatter/gather between distributed and sequential
  PETSc vectors, block-aligned local sizing, root-to-all array scatter.
- KSP factories: direct solver (`create_direct_solver`, MUMPS-based LU)
  and preconditioned GMRES (`create_gmres_solver` with `"bjacobi"` or
  `"block_banded"` preconditioner, both MUMPS-backed).
- Manipulation helpers for PETSc matrices/vectors and SLEPc BVs
  (real/imaginary parts, complex-conjugate enforcement, harmonic-balanced
  reshape, extract_block_banded, ...).

Worked examples live under `examples/`:

- `examples/cgl/` — complex Ginzburg-Landau: eigendecomposition, RSVD,
  RSVD-dt, balanced truncation.
- `examples/toy_model/` — harmonic resolvent analysis of a small
  time-periodic system.

If you use `resolvent4py` in your workflow, please cite [this](https://www.sciencedirect.com/science/article/pii/S2352711025002523) paper.
<details>
<summary>BibTeX</summary>

```bibtex
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
```
</details>

## Documentation

Click [here](https://albertopadovan.github.io/resolvent4py/).

## Dependencies

- `Python >= 3.9`
- `numpy`
- `scipy`
- `matplotlib`
- `mpi4py`
- `petsc4py >= 3.20`
- `slepc4py >= 3.20`



## Installation instructions

### Installation using `conda` and `pip` (recommended)

1. Create a new `conda` environment with, e.g., Python 3.12
    ```bash
    conda create -n resolvent4py_env python=3.12
    conda activate resolvent4py_env
    ```
    > **IMPORTANT:**
    > Please ensure that the environment variables `PETSC_DIR`, `PETSC_ARCH`
    > and `SLEPC_DIR` are unset.
    > This prevents `petsc4py` and `slepc4py` from binding to
    > unintended preexisting PETSc/SLEPc builds.
    > Likewise, ensure that any pre-installed MPI toolchains
    > (e.g., from system modules or previous installations) are not
    > interfering with the current build environment. That is,
    > ensure that the output of `which mpicc` is empty. If it is not, then
    > this indicates a potential conflict.

2. Run the command
    ```bash
    conda search 'petsc=3.25.3=complex*' --channel conda-forge --info
    ```
   to list all `PETSc` 3.25.3 builds with support for complex scalars and MUMPS.
   (Any release `>= 3.20.0` is expected to work; the most recent version we
   have tested is 3.25.3.)  This command will display metadata for each
   matching build, including its dependencies.
   Inspect the list of dependencies and look for "mumps" or "mumps-mpi".
   You will likely find two candidate builds that satisfy the MUMPS and
   complex-scalar requirements: one built with MPICH and one with OpenMPI.
   Select the one you prefer and identify the corresponding build string
   (e.g., `complex_hc93dccb_0` on a Mac osx-arm64 with OpenMPI).

3. Install `PETSc` and `petsc4py` with
    ```bash
    conda install -c conda-forge petsc=3.25.3=<build-string> petsc4py
    ```

4. Run
    ```bash
    conda search 'slepc=3.25.1=complex*' --channel conda-forge --info
    ```
   to perform the same inspection as above and identify a compatible `SLEPc`
   build (e.g., `complex_h174f6fd_0` on a Mac osx-arm64 with OpenMPI).
   If you chose the OpenMPI build in step 3, make sure you select the
   OpenMPI-compatible build of `SLEPc`.

5. Install `SLEPc` and `slepc4py` with
    ```bash
    conda install -c conda-forge slepc=3.25.1=<build-string> slepc4py
    ```

6. Install `mpi4py`
    ```bash
    conda install -c conda-forge mpi4py
    ```

7. Install `resolvent4py`
    ```bash
    pip install resolvent4py
    ```

### Installation from source

> **Note**
> If you have an existing parallel build of PETSc and SLEPc and their
> 4py counterparts configured with complex scalars
> (i.e., `--with-scalar-type=complex`) and with MUMPS (i.e.,
> `--download-mumps`), you can skip directly to step 10 (after running
> `pip install mpi4py`).


1. We recommend creating a clean Python environment using, e.g., `venv` or `conda`.
Ensure that you are using a Python version >= 3.9 by running
`python --version` in your terminal.
2. Ensure valid C, C++, and Fortran compilers are available,
along with ```make``` and ```flex```,
which can be obtained through a package management CLI.
3. Download [PETSc](https://petsc.org/release/install/download/). Any version
   `>= 3.20.0` should work.  (The latest version we tested is 3.25.3.)
4. Consult the PETSc [configuration guidelines](https://petsc.org/release/install/install/)
to configure PETSc with complex scalars (i.e., `--with-scalar-type=complex`) and
with MUMPS (i.e., `--download-mumps`).
For reference, here is a configure command (to be run inside the PETSc directory)
that has worked for us in the past,
    ```bash
    ./configure PETSC_ARCH=resolvent4py_arch --download-fblaslapack \
    --download-mumps --download-scalapack --download-parmetis \
    --download-metis --download-ptscotch --with-scalar-type=complex \
    --download-mpich --download-cmake --download-bison \
    --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
    ```
    > **Note**
    > Not all configure options shown in the multiline command above are
    > necessary for all users.  For example, if an MPI compiler is already
    > available, then `--download-mpich` may not be necessary and you can
    > pass configure flags like `--with-cc=mpicc`.  (Once again, please
    > consult the PETSc
    > [configuration guidelines](https://petsc.org/release/install/install/)
    > for your specific case.)
5. Follow the PETSc instructions (provided during the configuration step) to
   build the library.  Then make sure to export the environment variables
   `PETSC_DIR` and `PETSC_ARCH`.
6. If you downloaded MPICH during the configuration stage, then run the
   following commands to reference the correct MPI installation,
    ```bash
    export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
    export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
    ```
7. Install [SLEPc](https://slepc.upv.es/documentation/instal.htm). Any version
   `>= 3.20.0` should work.  (The latest version we tested is 3.25.1.)
8. Install `mpi4py`, `petsc4py` and `slepc4py`
    ```bash
    pip install mpi4py petsc4py==<petsc-version> slepc4py==<slepc-version>
    ```
    > **Note**
    > It is critical that you install the versions of `petsc4py` and
    > `slepc4py` corresponding to your PETSc and SLEPc installations.
    > Otherwise, the installation will likely fail.

9. Ensure that the installation was successful by running
    ```bash
    python -c "from mpi4py import MPI"
    python -c "from petsc4py import PETSc"
    python -c "from slepc4py import SLEPc"
    ```

10. Install `resolvent4py` with
    ```bash
    pip install resolvent4py
    ```

## Running the tests

The test suite is designed to run under MPI.  With `pytest` installed in your
environment, run

```bash
mpiexec -n 2 pytest tests/
```

The suite exercises 145 tests across the linear-operator, linalg, model-reduction,
and utility modules at approximately 91% line coverage.
