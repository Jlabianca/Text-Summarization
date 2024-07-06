import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.load_data import load_data
from models.model import initialize_model
from models.train import train_model

def preprocess_for_training(df, tokenizer):
    """
    Preprocess the DataFrame for training by tokenizing the inputs.

    Args:
    - df (DataFrame): The DataFrame containing the text data.
    - tokenizer (T5Tokenizer): The initialized tokenizer.

    Returns:
    - tokenized_dataset (Dataset): Tokenized dataset ready for model training.
    """
    # Function to tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['input_texts'], padding="max_length", truncation=True, max_length=512)
    
    # Apply the tokenization function to the DataFrame
    tokenized_dataset = df.map(tokenize_function, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    # Load the cleaned data from CSV files
    train_df, valid_df = load_data()

    # Initialize the model and tokenizer
    tokenizer, model = initialize_model()

    # Preprocess the training and validation data
    train_dataset = preprocess_for_training(train_df, tokenizer)
    valid_dataset = preprocess_for_training(valid_df, tokenizer)

    # Train the model
    train_model(model, tokenizer, {'train': train_dataset, 'validation': valid_dataset})
