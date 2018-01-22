%.tex: %.md
	pandoc $< -o $@
%.md: %.ipynb
	jupyter nbconvert $< --to=markdown

clean:
	rm -vf *.md *.tex
.PHONY: clean
