# Build the paper.  `make` -> main.pdf ; `make clean` removes build artifacts.
#
# `make tectonic` is the one-command alternative and needs no TeX distribution
# beyond tectonic itself (it fetches what it needs); it is what the paper has
# actually been compiled with.
PAPER = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

$(PAPER).pdf: $(PAPER).tex abstract.tex theory.tex refs.bib
	$(LATEX) $(PAPER)
	-bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

.PHONY: clean watch tectonic
tectonic:
	tectonic -X compile $(PAPER).tex --outdir .

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc $(PAPER).pdf

# Quick syntax check without the bibliography pass.
watch:
	$(LATEX) $(PAPER)
