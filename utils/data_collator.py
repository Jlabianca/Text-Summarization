from transformers import DataCollatorForSeq2Seq

def create_data_collator(tokenizer, model):
    """
    Create a data collator for Seq2Seq tasks.

    Args:
    - tokenizer (T5Tokenizer): The initialized tokenizer.
    - model (T5ForConditionalGeneration): The initialized model.

    Returns:
    - data_collator (DataCollatorForSeq2Seq): The data collator for Seq2Seq tasks.
    """
    return DataCollatorForSeq2Seq(tokenizer, model=model)
