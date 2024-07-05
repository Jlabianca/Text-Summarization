from data.preprocess import load_and_preprocess_dataset
from models.model import initialize_model
from models.train import train_model

if __name__ == "__main__":
    tokenized_dataset = load_and_preprocess_dataset()
    tokenizer, model = initialize_model()
    train_model(model, tokenizer, tokenized_dataset)
