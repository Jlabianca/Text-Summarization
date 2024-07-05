from datasets import load_metric
from utils.tokenizer import decode_predictions

def compute_metrics(pred):
    metric = load_metric("rouge")
    predictions, labels = pred
    decoded_preds = decode_predictions(predictions)
    decoded_labels = decode_predictions(labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}
