.PHONY: pdf clean distclean flatten

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
	$(MAKE) flatten

flatten:
	mkdir -p build
	@if command -v latexpand >/dev/null 2>&1; then \
		latexpand -o build/main_flat.tex main.tex; \
	else \
		echo "latexpand not found; cannot write build/main_flat.tex" >&2; \
		exit 1; \
	fi

clean:
	@if [ -d build ]; then \
		find build -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o -name '*.synctex.gz' -o -name '*.toc' \) -delete; \
	fi

distclean:
	rm -rf build
	mkdir -p build
