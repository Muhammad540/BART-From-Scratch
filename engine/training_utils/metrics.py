# watch this: https://www.youtube.com/watch?v=TMshhnrEXlg&ab_channel=HuggingFace

import evaluate

def calculate_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    return {k: round(v,4) for k,v in results.items()}