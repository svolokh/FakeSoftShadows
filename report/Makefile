report.pdf: report.tex reportfiles/* acmart.cls
	pdflatex report && bibtex report && pdflatex report

.PHONY: clean
clean:
	rm report.aux report.bbl report.blg report.log report.out report.pdf 
