from datasets import load_metric

def compute_rouge(predictions, references):
    metric = load_metric("rouge")
    result = metric.compute(predictions=predictions, references=references)
    return result
