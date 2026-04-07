import os
import pymupdf

class textFromPdf:
  def __init__(self, path="../data/"):
    self.path = path
  
  @staticmethod
  def chunks(text: str, chunkSize: int = 500, overlap: int = 50) -> list[str]: # Vai quebrar o texto inteiro em chunks, o modelo de embedding da erro com o texto inteiro
    chuncks = []
    start = 0
    while start < len(text):
      end = start + chunkSize
      chunk = text[start:end]
      chuncks.append(chunk)
      start += chunkSize - overlap
    return chuncks
  
  def allText(self, docName): #Convert todo o pdf em texto
    doc = pymupdf.open(os.path.join(self.path, docName))
    text = ""
    for page in doc:
      text += page.get_text('text')
    chunks = self.chunks(text, chunkSize=500, overlap=50)
    return chunks # Modifiquei pra receber o texto tratado e pronto para o uso