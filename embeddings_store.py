from pdf_handler import textFromPdf
from embeddings import embeddingsFromText
import chromadb

class embeddingsDB():
  def __init__(self, db_path="./data/chroma_db", collection_name="pdf_collection"):
    self.client = chromadb.PersistentClient(db_path)
    self.collection = self.client.get_or_create_collection(collection_name)
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

db = embeddingsDB()
db.embedding("pdfUsuario.pdf")
test = db.query("Oque é uma CPU?")
for e in test:
  print(f"\n\n-- {e}")
