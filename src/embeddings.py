from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class embeddingsFromText:
  def __init__(self, modelName="sentence-transformers/all-MiniLM-L6-v2"):
    self.tokenizer = AutoTokenizer.from_pretrained(modelName) #já padroniza o modelo que será utilizado pra não fazer bagunça que nem o codigo anterior
    self.model = AutoModel.from_pretrained(modelName) 
  
  @staticmethod
  def mean_pooling(output, mask):  # "Processo de converter os múltiplos vetores de palavras (tokens) gerados pelo modelo 
    tokenEmbeddings = output[0]   # em um único vetor de tamanho fixo que representa o significado de toda a sentença ou parágrafo"
    inputMaskExpanded = mask.unsqueeze(-1).expand(tokenEmbeddings.size()).float()
    return torch.sum(tokenEmbeddings * inputMaskExpanded, 1) / torch.clamp(inputMaskExpanded.sum(1), min=1e-9)

  def embedText(self, texts: list[str]):  # adaptado para receber a lista de chunks
    encodedInput = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt') #"tokenzina" o texto recebido
    with torch.no_grad(): # segundo o documento do modelo utilizado, essa parte converte o texto nos vetores com significado semantico
      modelOutput = self.model(**encodedInput) 
    sentenceEmbeddings = self.mean_pooling(modelOutput, encodedInput['attention_mask'])
    return F.normalize(sentenceEmbeddings, p=2, dim=1) 
    # "ajusta os vetores numéricos gerados (embeddings) 
    # para que todos tenham um comprimento unitário (magnitude de 1),
    # preservando apenas a direção do vetor"