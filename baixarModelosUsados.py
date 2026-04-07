from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
 
# Esse codigo serve caso seja sua primeira vez usando o codigo... Para usar o codigo é necessario ter um modelo de linguagem nos seus diretorios,
# portanto criei esse codigo justamente para aqueles que vão usar, vale ressaltar que o modelo usado pesa 3Gb
 
MODEL_NAME = "google/flan-t5-large"
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "models", "flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)
 
print(f"\nModelo salvo em: {SAVE_PATH}")
print("Pronto! Você já pode usar o bot.")