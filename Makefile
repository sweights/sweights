doc: build/html/index.html

build/html/index.html: doc/conf.py $(wildcard src/sweights/*.py doc/*/* doc/*)
	mkdir -p build/html
	sphinx-build -j3 -W -a -E -b html -d build/doctrees doc build/html

clean:
	rm -rf build
