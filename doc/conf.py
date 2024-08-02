# Configuration file for the sweights sphinx documentation builder.
from sweights import __version__ as release  # noqa
import sphinx_rtd_theme

project = "sweights"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    # 'matplotlib.sphinxext.only_directives',
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]

autoclass_content = "both"
autosummary_generate = True

# should prevent complicated types signatures in docs, but does not work?
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "Density": "Density",
    "FloatArray": "FloatArray",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_themes", "Thumbs.db", ".DS_Store"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

html_static_path = []

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
