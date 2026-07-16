# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../src"))

project = "resolvent4py"
copyright = "2024-2026, Alberto Padovan"
author = "Alberto Padovan"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]

# Cross-project references: turn ``numpy.ndarray``, ``petsc4py.PETSc.Vec``,
# etc. in our docstrings into clickable links to the upstream API pages.
# slepc4py has no public Sphinx-inventory build, so we cannot link its
# types automatically — they are ignored below instead.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "petsc4py": ("https://petsc.org/release/petsc4py", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
}
intersphinx_disabled_reftypes = []

# The docstrings use short-form type names ("PETSc.Vec", "SLEPc.BV",
# "np.ndarray") that intersphinx cannot resolve without the fully-
# qualified module path — and slepc4py has no upstream inventory at all.
# Silence nitpicky-mode false positives on these.
nitpick_ignore = [
    # Short-form PETSc types
    ("py:class", "PETSc.Vec"),
    ("py:class", "PETSc.Mat"),
    ("py:class", "PETSc.KSP"),
    ("py:class", "PETSc.Comm"),
    ("py:class", "PETSc.PC"),
    ("py:class", "PETSc.IS"),
    ("py:class", "PETSc.Scatter"),
    ("py:class", "PETSc.Options"),
    # PETSc type constants (Mat.Type.AIJ etc. — not classes, no target)
    ("py:class", "PETSc.Mat.Type.AIJ"),
    ("py:class", "PETSc.Mat.Type.DENSE"),
    ("py:class", "PETSc.Vec.Type.STANDARD"),
    ("py:class", "PETSc.Vec.Type.SEQ"),
    # Generic Python name used as a type
    ("py:class", "any"),
    ("py:class", "Any"),
    # Docstring uses type "python" descriptively rather than as a ref
    ("py:class", 'type "python"'),
    # SLEPc types — no upstream Sphinx inventory available
    ("py:class", "SLEPc.BV"),
    ("py:class", "SLEPc.EPS"),
    ("py:class", "SLEPc.SVD"),
    ("py:class", "slepc4py.SLEPc.BV"),
    ("py:class", "slepc4py.SLEPc.EPS"),
    ("py:class", "slepc4py.SLEPc.SVD"),
    ("py:meth", "slepc4py.SLEPc.BV.multVec"),
    # Short-form numpy types
    ("py:class", "np.ndarray"),
    ("py:class", "np.array"),
    ("py:class", "np.dtype"),
    ("py:class", "np.dtypes"),
    ("py:class", "numpy.array"),
    ("py:class", "numpy.complex128"),
    # Short-form MPI types (mpi4py inventory covers the FQ ones)
    ("py:class", "MPI.Datatype"),
    ("py:class", "MPI.Comm"),
    ("py:class", "MPI.Op"),
]
# Regex ignores: :type fields that carry ", default is X" as pseudo-types
# because Sphinx-Napoleon splits the type on comma.
nitpick_ignore_regex = [
    ("py:class", r".*(default|Default) is .*"),
    ("py:class", r"defaults? to .*"),
]

sphinx_gallery_conf = {
    "examples_dirs": [
        "../../examples/toy_model",
        "../../examples/cgl",
    ],  # Paths to your scripts
    "gallery_dirs": [
        "auto_examples/toy_model",
        "auto_examples/cgl",
    ],  # Where to output the HTML
    "filename_pattern": r"^.*\.py$",  # Include all .py files
    "ignore_pattern": (
        r"generate_matrices\.py|toy_model\.py|cgl\.py"
        r"|run_post_transient_approaches\.py"
    ),
    "plot_gallery": False,
}

html_logo = "logo.png"
html_show_sphinx = False
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

rst_prolog = """
.. |pkgname| replace:: resolvent4py
.. _MatType: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.Type.html
.. _MatSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.MatSizeSpec.html#petsc4py.typing.MatSizeSpec
.. _VecSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.LayoutSizeSpec.html#petsc4py.typing.LayoutSizeSpec
.. _KSPType: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html#petsc4py.PETSc.KSP.Type
.. _KSP: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.html
.. _Vec: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Vec.html
.. _StandardVec: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Vec.Type.html#petsc4py.PETSc.Vec.Type.STANDARD
.. _LayoutSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.LayoutSizeSpec.html#petsc4py.typing.LayoutSizeSpec
.. _BV: https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc.BV-class.html
.. _MPICOMM: https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Add custom CSS
# def setup(app):
#     app.add_css_file('custom.css')
