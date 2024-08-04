.. _contribute:

Contribute
==========

Please open issues and feature requests on `Github <https://github.com/sweights/sweights>`_.

Direct contributions are also very welcome. Please submit a pull request via Github.

Installing from source
----------------------

As a normal user, you don't need to install from sources, but you may want to do this if you want to fix a bug by yourself, or if you need the latest not-yet-released development version which has a bug-fix or a new feature that you want.

Clone the `Github repository <https://github.com/sweights/sweights>`_. Then go into the new directory and run these commands:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate
    pip install -e .'[test,doc]'

This creates a local environment, separate from your standard Python environment. Then it activates this environment and installs the package with pip locally in development mode, together the extra dependencies needed to run tests and to run the notebooks in the doc folder. You can now start changing things and the effects will become visible immediately.

You can run the unit tests with this command:

.. code-block:: bash

    python -m pytest

If you want to use this version elsewhere, you can go into the respective Python virtual environment and install sweights there with

.. code-block:: bash

    pip install <path-to-sweights>

For a deeper dive into developing sweights, you may want to generate the documentation and run the coverage test. For that we use ``nox``, which sets up the environments automatically, and to speed this up, we use ``uv``.

.. code-block:: bash

    pip install nox uv
    nox -s doc  # generate docs, open build/html/index.html
    nox -s cov  # compute coverage, open build/htmlcov/index.html
