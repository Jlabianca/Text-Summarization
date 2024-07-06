import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.model import initialize_model

def summarize(text, model, tokenizer):
    """
    Generate a summary for the given text using the model.

    Args:
    - text (str): The input text to summarize.
    - model (T5ForConditionalGeneration): The initialized model.
    - tokenizer (T5Tokenizer): The initialized tokenizer.

    Returns:
    - summary (str): The generated summary.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    # Generate the summary
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Initialize the model and tokenizer
    tokenizer, model = initialize_model()
    
    # Your custom article
    custom_article = """
    Your long article text goes here. Make sure to keep it within reasonable length
    to avoid issues with the model's maximum input length.
    """
    
    # Generate and print the summary
    summary = summarize(custom_article, model, tokenizer)
    print("Summary:")
    print(summary)
