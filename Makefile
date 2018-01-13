LATEXCMD := lualatex
MAKE_FULL_CMD := $(LATEXCMD) -interaction=nonstopmode -synctex=1 -shell-escape "\def\makefull{1}\input{thesis.tex}"

.PHONY: partial
partial:
	latexmk -pdflatex=$(LATEXCMD) thesis.tex

.PHONY: full
full:
	-$(MAKE_FULL_CMD)
	-biber thesis
	-$(MAKE_FULL_CMD)
	-$(MAKE_FULL_CMD)

preamble: thesis.tex
	$(LATEXCMD) -interaction=nonstopmode -ini -jobname="thesis" "&pdflatex" mylatexformat.ltx """thesis.tex"""

SUBDIR_ROOTS := chapters
DIRS := . $(shell find $(SUBDIR_ROOTS) -type d)
GARBAGE_PATTERNS := *.aux *.log *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.run.xml *.tdo *.fmt *.out *.auxlock *.synctex\(busy\) *.toc tikz/*.log
GARBAGE := $(foreach DIR,$(DIRS),$(addprefix $(DIR)/,$(GARBAGE_PATTERNS)))

.PHONY: clean
clean:
	rm -f $(GARBAGE) thesis.pdf tikz/*.md5 tikz_preview.*
