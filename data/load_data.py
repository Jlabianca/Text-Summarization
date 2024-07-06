import pandas as pd

def load_data(train_path='data/cleaned_train.csv', valid_path='data/cleaned_valid.csv'):
    """
    Load the training and validation data from CSV files.

    Args:
    - train_path (str): Path to the training data CSV file.
    - valid_path (str): Path to the validation data CSV file.

    Returns:
    - train_df (DataFrame): Training data as a pandas DataFrame.
    - valid_df (DataFrame): Validation data as a pandas DataFrame.
    """
    # Read the CSV files into DataFrames
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    return train_df, valid_df
