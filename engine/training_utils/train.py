import torch
import torch.optim as optim 
import yaml
import os
import math 
import numpy as np
from tqdm import tqdm 
from engine.model_utils.engine import BART 
from engine.data_utils.dataset_loading import load_cnn_dailymail
from engine.training_utils.metrics import calculate_rouge
from transformers import get_linear_schedule_with_warmup, BartTokenizer

class Config:
    """simple class to store the config"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def train(config_path,
          save_dir):
    """
    train bart model on the cnn/dailymail dataset
    
    Args:
        config_path: model configuration yaml file
        batch_size: batch size for training
        num_epochs: number of training epochs
        learning_rate: learning rate for the optimizer
        warmup_steps: number of warmup steps for the learning rate scheduler
        max_grad_norm: maximum gradient norm for gradient clipping
        save_dir: directory to save model checkpoints
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    config = Config(config_dict)
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    max_seq_len = config.max_seq_len
    learning_rate = config.learning_rate
    warmup_steps = config.warmup_steps
    max_grad_norm = config.max_grad_norm
    log_every = config.log_every
    eval_samples = config.eval_samples
    start_eval_gen = config.start_eval_gen
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataloader, val_dataloader, _ = load_cnn_dailymail(batch_size=batch_size, max_length=max_seq_len)
    model = BART(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer=optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader)*num_epochs
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # we will keep track of imp metrics and also during the training process 
    # we will generate summaries (for vibe checks) and to calculate rouge scores
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "learning_rate": []
    }
    
    global_step = 0
    best_val_loss = float('inf')    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_step = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            seq_len = decoder_input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)
            
            # create a cross attention mask with dim [batch_size, 1, decoder_seq_len, encoder_seq_len]
            # this will allow the decoder position to attend to all encoder positions 
            # Note: we use the encoder's attention mask, not decoder attention mask because 
            # we need to prevent attending to padding tokens in the encoder sequence
            encoder_decoder_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1,-1,seq_len,-1)
            
            # forward pass with causal mask 
            _,loss = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_padding_mask=attention_mask,
                encoder_decoder_attention_mask=encoder_decoder_attention_mask,
                decoder_causal_mask=causal_mask,
                labels=labels
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # compute perplexity = exp(loss)
            perplexity = math.exp(loss.item())
            total_loss += loss.item()
            epoch_step += 1
            global_step += 1
            
            progress_bar.set_postfix(
                {
                    "step": global_step,
                    "loss": loss.item(),
                    "avg_loss": total_loss/(epoch_step + 1),
                    "ppl": perplexity,
                    "lr": scheduler.get_last_lr()[0]
                }
            )
            
            if global_step % log_every == 0:
                metrics_history["train_loss"].append((global_step, loss.item()))
                metrics_history["perplexity"].append((global_step, perplexity))
                metrics_history["learning_rate"].append((global_step, scheduler.get_last_lr()[0]))
                
                # after waiting for start_eval_gen steps, we will generate some summaries
                if global_step > start_eval_gen:
                    print("\n---Sample Generation---")
                    sample_input = input_ids[0:1]
                    sample_mask = attention_mask[0:1]
                    
                    generated_ids = model.generate(
                        input_ids=sample_input,
                        attention_mask=sample_mask,
                        use_greedy=True
                    )
                    
                    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    original_text = tokenizer.batch_decode(sample_input, skip_special_tokens=True)
                
                    print(f"Original: {original_text[0][:100]}")
                    print(f"Generated: {generated_text[0][:100]}")

        
        model.eval()
        val_loss = 0
        val_step = 0 
        all_preds = []
        all_targets = []
        
        # lets do some eval
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                dec_seq_len = decoder_input_ids.size(1)
                causal_mask = torch.tril(torch.ones(dec_seq_len, dec_seq_len)).unsqueeze(0).unsqueeze(0).to(device)
                
                # create a cross attention mask with dim [batch_size, 1, decoder_seq_len, encoder_seq_len]
                # this will allow the decoder position to attend to all encoder positions 
                # Note: we use the encoder's attention mask, not decoder attention mask because 
                # we need to prevent attending to padding tokens in the encoder sequence
                encoder_decoder_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1,-1,dec_seq_len,-1)
                
                # forward pass
                _,loss = model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_padding_mask=attention_mask,
                    encoder_decoder_attention_mask=encoder_decoder_attention_mask,
                    decoder_causal_mask=causal_mask,
                    labels=labels
                )
                
                val_loss += loss.item()
                val_step += 1
                
                if val_step <= eval_samples:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_greedy=True
                    )
                    
                    for j in range(len(generated_ids)):
                        generated_summary = tokenizer.decode(
                            generated_ids[j],
                            skip_special_tokens=True,
                        )
                        reference_summary = tokenizer.decode(
                            labels[j],
                            skip_special_tokens=True
                        )
                        
                        all_preds.append(generated_summary)
                        all_targets.append(reference_summary)
                        
                
        avg_val_loss = val_loss / val_step
        val_perplexity = math.exp(avg_val_loss)
        
        if all_preds:
            rouge_scores = calculate_rouge(all_preds, all_targets)
            rouge1 = rouge_scores['rouge-1']['f']
            rouge2 = rouge_scores['rouge-2']['f']
            rougel = rouge_scores['rouge-l']['f']
            
            metrics_history['rouge-1'].append((global_step, rouge1))
            metrics_history['rouge-2'].append((global_step, rouge2))
            metrics_history['rouge-l'].append((global_step, rougel))
            
            print(f"\nROUGE Scores - R1: {rouge1:.4f}, R2: {rouge2:.4f}, RL: {rougel:.4f}")
        
        metrics_history['val_loss'].append((global_step, avg_val_loss))
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {total_loss/len(train_dataloader):.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.4f}")
        
        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': total_loss/len(train_dataloader),
            'val_loss': avg_val_loss,
            'metrics': metrics_history
        }, checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': total_loss/len(train_dataloader),
                'val_loss': avg_val_loss,
                'metrics': metrics_history
            }, best_model_path)
            print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
        
        print(f"Model saved to {checkpoint_path}")
        
        if all_preds:
            print("\nSample Generations:")
            for i in range(min(2, len(all_preds))):
                print(f"\nGenerated: {all_preds[i][:200]}...")
                print(f"Reference: {all_targets[i][:200]}...")
        
    print("Training complete")
    
    import json
    metrics_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f)
    print(f"Training metrics saved to {metrics_path}")
    