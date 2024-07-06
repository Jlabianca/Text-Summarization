import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.model import initialize_model

def summarize(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = initialize_model()
    text = "Your document text here."
    summary = summarize(text, model, tokenizer)
    print(summary)
