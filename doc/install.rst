.. _install:

Installation
============

pip
---

To install the latest stable version from PyPI with ``pip``:

.. code-block:: bash

    pip install sweights

Conda
-----

Not yet available, but you can safely install with pip into your existing conda environment.

Installing from source
----------------------

As a normal user, you don't need to install from sources, but it is not difficult.

You may want to do this if you want to fix a bug by yourself, or if you need the latest not-yet-released development version which has a bug-fix or a new feature that you want.

Clone the `Github repository <https://github.com/sweights/sweights>`_. Then go into the new directory and run these commands:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate
    pip install -e .'[test,doc]'

This creates a local environment, separate from your standard Python environment. Then it activates this environment and installs the package with pip locally in development mode, together the extra dependencies needed to run tests and to run the notebooks in the doc folder. You can now start changing things and the effects will become visible immediately.

You can run the unit tests with this command:

.. code-block:: bash

    python -m pytest
