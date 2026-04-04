from pdf_handler import textFromPdf
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class embeddingsFromText:
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name) 
  
  @staticmethod
  def mean_pooling(output, mask):
    token_embeddings = output[0]
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def embedText(self, texts: list[str]):  # adaptado para receber a lista de chunks
    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
      model_output = self.model(**encoded_input) 
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1)