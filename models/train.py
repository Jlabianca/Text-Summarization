from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.data_collator import create_data_collator

def train_model(model, tokenizer, datasets):
    """
    Train the model with the provided datasets.

    Args:
    - model (T5ForConditionalGeneration): The initialized model.
    - tokenizer (T5Tokenizer): The initialized tokenizer.
    - datasets (dict): A dictionary containing 'train' and 'validation' datasets.
    """
    train_dataset = datasets['train']  # Training data
    valid_dataset = datasets['validation']  # Validation data

    # Create a data collator to handle the data format
    data_collator = create_data_collator(tokenizer, model)

    # Define training settings
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",  # Where to save the model
        evaluation_strategy="epoch",  # Evaluate every epoch
        learning_rate=2e-5,  # Learning rate for the optimizer
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        weight_decay=0.01,  # Weight decay for regularization
        save_total_limit=3,  # Limit on the number of saved checkpoints
        num_train_epochs=3,  # Number of training epochs
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training the model
    trainer.train()