from pdf_handler import textFromPdf
from embeddings import embeddingsFromText
import chromadb

pdf_handler = textFromPdf()
textToTransformPDF = pdf_handler.allText("pdfUsuario.pdf")
textToTransformUser = pdf_handler.allText("userInput.txt")

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("pdf_collection")
embedder = embeddingsFromText() #necessario por causa do self... Deu muita dor de cabeça
embeddings = embedder.embedText(textToTransformPDF).cpu().numpy().tolist()
ids =[f"chunk_{i}" for i in range(len(textToTransformPDF))]

collection.add(
  documents=textToTransformPDF, 
  embeddings=embeddings, 
  ids=ids
  )

print(f"Indexados {len(textToTransformPDF)} chunks no ChromaDB!")

query = " ".join(textToTransformPDF)
results = collection.query(query_texts=[query], n_results=2)
for doc in results["documents"][0]:
  print(f"- {doc}")