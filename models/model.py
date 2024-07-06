from transformers import T5Tokenizer, T5ForConditionalGeneration

def initialize_model():
    """
    Initialize the T5 model and tokenizer.

    Returns:
    - tokenizer (T5Tokenizer): Tokenizer for the T5 model.
    - model (T5ForConditionalGeneration): T5 model for generating summaries.
    """
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model
