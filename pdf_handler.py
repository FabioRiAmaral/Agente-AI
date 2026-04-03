import os
import pymupdf


class textFromPdf:
  def allText(path, docName):
    doc = pymupdf.open(os.path.join(path, docName))
    text = ""
    for page in doc:
      text += page.get_text('text')
    return text

print(textFromPdf.allText("./data/", "pdfUsuario.pdf"))