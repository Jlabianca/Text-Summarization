from datasets import load_dataset

def load_and_preprocess_dataset():
    """
    Load and preprocess the CNN/Daily Mail dataset.

    Returns:
    - tokenized_dataset (Dataset): Tokenized dataset ready for model training.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    def preprocess_function(examples):
        inputs = examples['article']
        targets = examples['highlights']
        return {'input_texts': inputs, 'target_texts': targets}

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset
