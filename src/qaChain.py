import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from pdf_handler import textFromPdf
from embeddings import embeddingsFromText
from embeddings_store import embeddingsDB
from translator import Translator

MODELS_DIR = os.path.join('../', "models")

class LangChainPDFPipeline:
  def __init__(self, db_path="../data/chroma_db", collection="pdf_collection"):
    self.collection_name = collection
    self.db_path = db_path
    self.pdf_handler = textFromPdf()
    self.embedder = embeddingsFromText()
    self.db = embeddingsDB(dbPath=db_path, collectionName=collection)
    self.translator = Translator()

    class MyEmbeddings(Embeddings):
      def embed_documents(_, texts):
        return self.embedder.embedText(texts).cpu().numpy().tolist()

      def embed_query(_, text):
        return self.embedder.embedText([text]).cpu().numpy().tolist()[0]

    self._embedding_fn = MyEmbeddings()
    self.vectorstore = self._build_vectorstore()

    model_path = os.path.join(MODELS_DIR, "flan-t5-large")
    self.tokenizer = T5Tokenizer.from_pretrained(model_path)
    self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    self.model.config.tie_word_embeddings = False

  def _build_vectorstore(self) -> Chroma:
    return Chroma(
      client=self.db.client,
      collection_name=self.collection_name,
      embedding_function=self._embedding_fn,
      persist_directory=self.db_path,
    )

  def _reset_collection(self) -> None:
    self.db.client.delete_collection(self.collection_name)
    self.db.collection = self.db.client.get_or_create_collection(self.collection_name)
    self.vectorstore = self._build_vectorstore()

  def index_pdf(self, pdf_name: str) -> None:
    self._reset_collection()
    self.db.embedding(pdf_name)

  def semantic_query(self, question: str, k: int = 5) -> str:
    retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in results)

  def qa_with_llm(self, question: str) -> str:
    question_en = self.translator.pt_to_en(question)
    context_pt  = self.semantic_query(question, k=5)
    context_en  = self.translator.pt_to_en(context_pt)

    prompt = (
      "You are an expert assistant. Using ONLY the context below, "
      "write a detailed and complete explanation to answer the question. "
      "Include all relevant details from the context.\n\n"
      f"Context:\n{context_en}\n\n"
      f"Question: {question_en}\n\n"
      "Detailed answer:"
    )

    inputs = self.tokenizer(
      prompt,
      return_tensors="pt",
      max_length=1024,
      truncation=True,
    )

    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=400,
        num_beams=5,
        no_repeat_ngram_size=4,
        early_stopping=True,
        repetition_penalty=2.0,
        length_penalty=2.0,
      )

    answer_en = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return self.translator.en_to_pt(answer_en)