"""
Noxfile for iminuit.

Pass extra arguments to pytest after --
"""

import nox
import sys

sys.path.append(".")
import python_releases

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

ENV = {
    "COVERAGE_CORE": "sysmon",  # faster coverage on Python 3.12
}

PYPROJECT = nox.project.load_toml("pyproject.toml")
MINIMUM_PYTHON = PYPROJECT["project"]["requires-python"].strip(">=")
LATEST_PYTHON = str(python_releases.latest())

nox.options.sessions = ["test", "maxtest"]


@nox.session(reuse_venv=True, venv_backend="micromamba")
def test(session: nox.Session) -> None:
    """Run all tests."""
    # running in parallel with pytest-xdist crashes ROOT
    session.conda_install(
        "root",
        "iminuit",
        "scipy",
        "pytest",
        "pytest-cov",
        "coverage",
        "matplotlib",
        channel="conda-forge",
    )
    session.install("-e.", "--no-deps")
    session.run("pytest")


# broken: this tries to compile dependencies
@nox.session(python=MINIMUM_PYTHON, venv_backend="uv")
def mintest(session: nox.Session) -> None:
    """Run tests on the minimum python version."""
    session.install("-ve.", "--only-binary=1", "--resolution=lowest-direct")
    session.install("pytest", "pytest-xdist")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args)


@nox.session(python=LATEST_PYTHON)
def maxtest(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install(
        "-e.", "numpy", "scipy", "matplotlib", "pytest", "pytest-xdist", "--pre"
    )
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args, env=ENV)


# Python-3.12 provides coverage info faster
# We need micromamba here to install ROOT
@nox.session(python="3.12", venv_backend="micromamba", reuse_venv=True)
def cov(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    session.conda_install(
        "root",
        "iminuit",
        "scipy",
        "pytest",
        "pytest-cov",
        "coverage",
        "matplotlib",
        channel="conda-forge",
    )
    session.install("-e.", "--no-deps")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "build/htmlcov")
    session.run("coverage", "report", "-m")


@nox.session(python="3.12", reuse_venv=True)
def doc(session: nox.Session) -> None:
    """Build html documentation."""
    session.install("-e.[test,doc]")

    # link check
    session.run(
        "sphinx-build",
        # "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-v",
        "-b=html",
        "-W",
        "-j=auto",
        "-d=build/doctrees",
        "doc",
        "build/html",
    )


@nox.session(python="3.12", reuse_venv=True)
def linkcheck(session: nox.Session) -> None:
    """Check all links in the documentation."""
    session.install("-e.[test,doc]")

    # link check
    session.run(
        "sphinx-build",
        "-b=linkcheck",
        "doc",
        "build/html",
    )
