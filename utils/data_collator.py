from transformers import DataCollatorForSeq2Seq

def create_data_collator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer, model=model)
