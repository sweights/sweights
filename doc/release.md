# Procedure for making a new release

 * modify the version number in `src/sweights/__init__.py`
 * commit any changes and merge into main (pre-commit hooks should handle any issues)
 * build the package locally
 ```bash
 python -m build
 ```
 * this shouuld produce the distribution files in `dist/sweights-X.Y.Z-py3-none-any.whl` and `dist/sweights-X.Y.Z.tar.gz`
 * now upload the distribution archives using twine
 ```bash
 twine upload dist/sweights-X.Y.Z*
 ```
 * Create a release and new tag on github with the name `vX.Y.Z`
 * readthedocs should now build the docs once the commit and tag have been made but you should check this worked properly
