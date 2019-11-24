all: html pdf docx rtf

pdf: index.pdf
index.pdf: index.md
	pandoc --standalone --template style_chmduquesne.tex \
	--from markdown --to context \
	-V papersize=A4 \
	-o index.tex index.md; \
	context index.tex

html: index.html
index.html: style_chmduquesne.css index.md
	pandoc --standalone -H style_chmduquesne.css \
        --from markdown --to html \
        -o index.html index.md

docx: index.docx
index.docx: index.md
	pandoc -s -S index.md -o index.docx

rtf: index.rtf
index.rtf: index.md
	pandoc -s -S index.md -o index.rtf

clean:
	rm index.html
	rm index.tex
	rm index.tuc
	rm index.log
	rm index.pdf
	rm index.docx
	rm index.rtf
