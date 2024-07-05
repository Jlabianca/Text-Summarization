from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.data_collator import DataCollatorForSeq2Seq

def train_model(model, tokenizer, tokenized_dataset):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
