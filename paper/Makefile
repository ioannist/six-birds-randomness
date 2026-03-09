.PHONY: pdf clean distclean

pdf:
	mkdir -p build
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex; \
	else \
		echo "latexmk not found; using pdflatex/bibtex fallback"; \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
		bibtex build/main; \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
	fi

clean:
	@if [ -d build ]; then \
		find build -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o -name '*.synctex.gz' -o -name '*.toc' \) -delete; \
	fi

distclean:
	rm -rf build
	mkdir -p build
