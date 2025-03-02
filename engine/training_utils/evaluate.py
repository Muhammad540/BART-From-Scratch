import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from tqdm import tqdm 
from engine.model_utils.engine import BART 
from engine.data_utils.dataset_loading import load_cnn_dailymail
import yaml 
from engine.training_utils.config import Config
from transformers import BartTokenizer
import evaluate

def calculate_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    return {k: round(v,4) for k,v in results.items()}

def evaluate(checkpoint_path,
             config_path):
    '''
    evaluate the bart model on the cnn/dailymail dataset
    
    Args:
        checkpoint_path: path to the checkpoint file
        config_path: path to the config file
        batch_size: batch size for evaluation
        num_samples: number of samples to evaluate on
    '''
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    batch_size = config.batch_size
    num_samples = config.num_samples
    max_seq_len = config.max_seq_len
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,_,test_dataloader = load_cnn_dailymail(batch_size=batch_size, max_length=max_seq_len)
    
    model = BART(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    model.eval()    
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Generating Summaries")):
            if i >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_hf_generation=False
            )
    
            for j in range(len(generated_ids)):
                generated_summary = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                reference_summary = tokenizer.decode(batch["labels"][j], skip_special_tokens=True)
                article = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                
                all_preds.append(generated_summary)
                all_targets.append(reference_summary)
    
    rouge_scores = calculate_rouge(all_preds, all_targets)
    
    print(f"ROUGE Scores: ")
    print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
    
    print("\n Sample Summaries:")
    for i in range(min(3, len(all_preds))):
        print(f"\nArticle {i+1}:")
        print(article[i][:500])
        print(f"\nGenerated Summary: {all_preds[i]}")
        print(f"Reference Summary: {all_targets[i]}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BART model")
    parser.add_argument("checkpoint_path", help="path to the checkpoint file")
    args = parser.parse_args()
    
    evaluate(args.checkpoint_path)
    