# Configuration file for the sweights sphinx documentation builder.
from sweights import __version__ as release  # noqa
import sphinx_rtd_theme

project = "sweights"
copyright = "2023, Matt Kenzie and Hans Dembinski"

with open("../README.rst") as f:
    readme_content = f.read()

with open("index.rst.in") as f:
    index_content = f.read()

readme_content = readme_content.replace(
    "https://raw.githubusercontent.com/sweights/sweights/main/doc/", ""
)
readme_content = readme_content.replace(
    ".. version-marker-do-not-remove",
    "**These docs are for sweights version:** |release|",
)

begin = readme_content.index(".. index-replace-marker-begin-do-not-remove")
end = readme_content.index(".. index-replace-marker-end-do-not-remove")
readme_content = readme_content[:begin] + index_content + readme_content[end:]

with open("index.rst", "w") as f:
    f.write(readme_content)


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

html_static_path = ["_static"]
html_logo = "_static/sweights_logo.svg"

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
