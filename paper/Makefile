.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

tectonic:
	tectonic -X compile main.tex

clean:
	latexmk -C
	rm -f main.bbl main.run.xml
