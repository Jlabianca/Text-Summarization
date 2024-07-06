def decode_predictions(predictions, tokenizer):
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)
