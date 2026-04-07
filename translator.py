import os
import torch
from transformers import MarianMTModel, MarianTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
 
def _download_if_needed(hf_name: str, local_path: str) -> None:
  if not os.path.exists(local_path):
    print(f"Baixando modelo: {hf_name}...")
    tok = MarianTokenizer.from_pretrained(hf_name)
    mdl = MarianMTModel.from_pretrained(hf_name)
    tok.save_pretrained(local_path)
    mdl.save_pretrained(local_path)
   
class Translator: # Nova classe para traduzir, o modelo de linguagem usada não consegue responder muito bem com as perguntas e o pdf em portugues
  def __init__(self):
    pt_en_path = os.path.join(MODELS_DIR, "opus-mt-pt-en")
    en_pt_path = os.path.join(MODELS_DIR, "opus-mt-en-pt")

    _download_if_needed("Helsinki-NLP/opus-mt-ROMANCE-en", pt_en_path)
    _download_if_needed("Helsinki-NLP/opus-mt-en-ROMANCE", en_pt_path)

    self.tok_pt_en = MarianTokenizer.from_pretrained(pt_en_path)
    self.mdl_pt_en = MarianMTModel.from_pretrained(pt_en_path)
    self.tok_en_pt = MarianTokenizer.from_pretrained(en_pt_path)
    self.mdl_en_pt = MarianMTModel.from_pretrained(en_pt_path)

  def _translate_chunks(self, text: str, tokenizer, model, prefix: str = "") -> str:
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < 400:
          current += sentence + ". "
        else:
          if current:
            chunks.append(current.strip())
          current = sentence + ". "
    if current:
      chunks.append(current.strip())

    translated = []
    for chunk in chunks:
      inp = tokenizer(
          prefix + chunk,
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=512,
      )
      with torch.no_grad():
        out = model.generate(**inp, num_beams=4)
      translated.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return " ".join(translated)

  def pt_to_en(self, text: str) -> str:
    return self._translate_chunks(text, self.tok_pt_en, self.mdl_pt_en)
    
  def en_to_pt(self, text: str) -> str:
    return self._translate_chunks(text, self.tok_en_pt, self.mdl_en_pt, prefix=">>pt<< ")