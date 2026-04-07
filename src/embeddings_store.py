import os
from pdf_handler import textFromPdf
from embeddings import embeddingsFromText
import chromadb

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

class embeddingsDB():
  def __init__(self, dbPath=None, collectionName="pdf_collection"):
    dbPath = dbPath or os.path.join(DATA_DIR, "chroma_db")
    self.client = chromadb.PersistentClient(dbPath)
    self.collection = self.client.get_or_create_collection(collectionName)
    self.embedder = embeddingsFromText() #necessario por causa do self... Deu muita dor de cabeça
    
  def embedding(self, pdf_name):
    pdf_handler = textFromPdf()
    chunks = pdf_handler.allText(pdf_name)
    embeddings = self.embedder.embedText(chunks).cpu().numpy().tolist()
    ids =[f"chunk_{i}" for i in range(len(chunks))]
    self.collection.add(
      documents=chunks, 
      embeddings=embeddings, 
      ids=ids
      )
    return self.collection

  def query(self, question):
    embeddingQuestion = self.embedder.embedText([question]).cpu().numpy().tolist()
    queryResults = self.collection.query(query_embeddings=embeddingQuestion, n_results=3)
    return queryResults["documents"][0] # Consulta no pdf semantica conforme a pergunta