%.tex: %.ipynb thesis.tplx
	jupyter nbconvert --to latex --template thesis --execute --ExecutePreprocessor.kernel_name=python3 $<

clean:
	rm -vf *.md *.tex
.PHONY: clean
