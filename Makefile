doc: build/html/done

build/html/done: doc/conf.py $(wildcard src/sweights/*.py doc/*.rst doc/_static/* doc/plots/* *.rst examples/*.py)
	mkdir -p build/html
	sphinx-build -j3 -W -a -E -b html -d build/doctrees doc build/html
	touch build/html/done

clean:
	rm -rf build
