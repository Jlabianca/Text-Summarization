from datasets import load_metric

def compute_rouge(predictions, references):
    """
    Compute the ROUGE scores for the given predictions and references.

    Args:
    - predictions (list): List of predicted summaries.
    - references (list): List of reference summaries.

    Returns:
    - result (dict): Dictionary containing the computed ROUGE scores.
    """
    metric = load_metric("rouge")  # Load the ROUGE metric
    # Compute and return the ROUGE scores
    result = metric.compute(predictions=predictions, references=references)
    return result
