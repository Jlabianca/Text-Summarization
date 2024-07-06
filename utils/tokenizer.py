from transformers import T5Tokenizer

def initialize_tokenizer(model_name='t5-small'):
    """
    Initialize the T5 tokenizer.

    Args:
    - model_name (str): The name of the T5 model.

    Returns:
    - tokenizer (T5Tokenizer): The initialized tokenizer.
    """
    return T5Tokenizer.from_pretrained(model_name)

def decode_predictions(predictions, tokenizer):
    """
    Decode the tokenized predictions.

    Args:
    - predictions (list): List of tokenized predictions.
    - tokenizer (T5Tokenizer): The initialized tokenizer.

    Returns:
    - decoded_predictions (list): List of decoded predictions.
    """
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)
