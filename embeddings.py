from pdf_handler import textFromPdf
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class embeddingsFromText:
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name) #já padroniza o modelo que será utilizado pra não fazer bagunça que nem o codigo anterior
    self.model = AutoModel.from_pretrained(model_name) 
  
  @staticmethod
  def mean_pooling(output, mask):  # "Processo de converter os múltiplos vetores de palavras (tokens) gerados pelo modelo 
    token_embeddings = output[0]   # em um único vetor de tamanho fixo que representa o significado de toda a sentença ou parágrafo"
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def embedText(self, texts: list[str]):  # adaptado para receber a lista de chunks
    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt') #"tokenzina" o texto recebido
    with torch.no_grad(): # segundo o documento do modelo utilizado, essa parte converte o texto nos vetores com significado semantico
      model_output = self.model(**encoded_input) 
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']) #Transforma os vetores das palvras individuais em um único vetor para toda a chunck
    return F.normalize(sentence_embeddings, p=2, dim=1) 
    # "ajusta os vetores numéricos gerados (embeddings) 
    # para que todos tenham um comprimento unitário (magnitude de 1),
    # preservando apenas a direção do vetor"