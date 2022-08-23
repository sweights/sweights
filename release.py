import os
import sys

# get version
with open("src/sweights/__init__.py") as f:
    for line in f.readlines():
        if line.startswith("__version__"):
            __version__ = line.split()[-1].strip('"')

print(f"Releasing sweights v{__version__}")

with open("doc/requirements.txt") as f:
    reqs = f.readlines()
    for req in reqs:
        if req.startswith("sweights"):
            version = req.split("sweights==")[-1]
            if version == __version__:
                sys.exit()

with open("doc/requirements.txt", "w") as f:
    for req in reqs:
        if req.startswith("sweights"):
            f.write("sweights==" + __version__ + "\n")
        else:
            f.write(req)


dist_cmd = "python setup.py sdist bdist_wheel"

a = input(f"Create distribution packages [y]/n? \n {dist_cmd}")

if a != "" and a != "y" and a != "Y":
    sys.exit()

os.system(dist_cmd)

upload_cmd = f"python -m twine upload dist/sweights-{__version__}*"

a = input(f"Upload to pypi [y]/n? \n {upload_cmd}")

if a != "" and a != "y" and a != "Y":
    sys.exit()

os.system(upload_cmd)

print(
    """Currently you will need to git add, git commit,
       git push and then on the github UI setup a new tag and
       new release.
       readthedocs will then update automatically
    """
)
