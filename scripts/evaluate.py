import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_metric
from utils.tokenizer import decode_predictions

def compute_metrics(pred):
    """
    Compute the evaluation metrics (ROUGE) for the model's predictions.

    Args:
    - pred (tuple): Tuple containing the predictions and the labels.

    Returns:
    - result (dict): Dictionary containing the computed ROUGE scores.
    """
    metric = load_metric("rouge")  # Load the ROUGE metric
    predictions, labels = pred  # Get the predictions and labels
    decoded_preds = decode_predictions(predictions)  # Decode the predictions
    decoded_labels = decode_predictions(labels)  # Decode the labels

    # Compute the ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}
