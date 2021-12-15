from sweights import __version__
import sys

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
