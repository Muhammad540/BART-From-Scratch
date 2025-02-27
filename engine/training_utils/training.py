import torch 
from tqdm import tqdm 
import numpy as np

def train_epoch(model.
                dataloader,
                optimizer,
                scheduler,
                clip_norm,
                device):
    
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs, loss = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_padding_mask=attention_mask,
            labels=labels
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)