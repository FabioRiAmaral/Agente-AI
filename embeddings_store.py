from pdf_handler import textFromPdf
from embeddings import embeddingsFromText
import chromadb

textToTransformPDF = textFromPdf.allText("./data/", "pdfUsuario.pdf")
textToTransformUser = textFromPdf.allText("./data/", "userInput.txt")
chunks = textFromPdf.chunks(textToTransformPDF, chunk_size=500, overlap=50)

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("pdf_collection")
embedder = embeddingsFromText() #necessario por causa do self... Deu muita dor de cabeça
embeddings = embedder.embedText(chunks).cpu().numpy().tolist()
ids =[f"chunk_{i}" for i in range(len(chunks))]

collection.add(
  documents=chunks, 
  embeddings=embeddings, 
  ids=ids
  )

print(f"Indexados {len(chunks)} chunks no ChromaDB!")

query = " ".join(textToTransformUser)
results = collection.query(query_texts=[query], n_results=2)
for doc in results["documents"][0]:
  print(f"- {doc}")