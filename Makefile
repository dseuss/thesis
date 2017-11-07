MAKE_FULL_CMD := pdflatex -interaction=nonstopmode -synctex=1 "\def\makefull{1} $(MAKE_FULL_FLAG) \input{thesis.tex}"

.PHONY: partial
partial:
	latexmk thesis.tex

.PHONY: full
full:
	-$(MAKE_FULL_CMD)
	-biber thesis
	-$(MAKE_FULL_CMD)
	-$(MAKE_FULL_CMD)


SUBDIR_ROOTS := chapters
DIRS := . $(shell find $(SUBDIR_ROOTS) -type d)
GARBAGE_PATTERNS := *.aux *.log *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.run.xml *.tdo *.fmt
GARBAGE := $(foreach DIR,$(DIRS),$(addprefix $(DIR)/,$(GARBAGE_PATTERNS)))

.PHONY: clean
clean:
	rm -f $(GARBAGE) thesis.pdf
