from datasets import load_dataset 
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

class SummarizationDataset(Dataset):
    def __init__(self,
                 articles,
                 summaries,
                 tokenizer=BartTokenizer.from_pretrained("facebook/bart-base"),
                 max_length=512):
        self.articles = articles
        self.summaries = summaries 
        self.tokenizer = tokenizer 
        self.max_length = max_length 
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        
        # tokenize the article and summary 
        article_encodings = self.tokenizer(article,
                                           max_length=self.max_length,
                                           padding="max_length",
                                           truncation=True)
        summary_encodings = self.tokenizer(summary,
                                           max_length=self.max_length,
                                           padding="max_length",
                                           truncation=True)
        
        # this masking is needed to distinguish between padding and actual tokens
        article_mask = article_encodings["attention_mask"]
        summary_mask = summary_encodings["attention_mask"]
        
        # perform teacher forcing 
        # decoder input ids are what we feed into the decoder 
        # labels is what we expect the decoder output to be 
        ''' Breif summary of how teacher forcing works:
        In the decoder, we feed the input ids with the last predicted token removed
        The first token in the label is removed since it is the start token and we want to predict the next token
        So say the input sentence is "Hello, how are you?" with token ids: [<start> , 8 , 9 , 10 , 11 , 12 , <end>]
        The decoder input ids are: [<start> , 8 , 9 , 10 , 11 , 12]
        The labels are: [8 , 9 , 10 , 11 , 12 , <end>]
        So when '<start>' is fed into the decoder, the output should be 8, so the label is 8
        The next input to the decoder is 8, so the output should be 9, so the label is 9
        This continues until the last token is reached
        '''
        return {
            "input_ids": torch.tensor(article_encodings["input_ids"]),
            "attention_mask": torch.tensor(article_mask),
            "decoder_input_ids": torch.tensor(summary_encodings["input_ids"][:-1]),
            "decoder_attention_mask": torch.tensor(summary_mask[:-1]),
            "labels": torch.tensor(summary_encodings["input_ids"][1:])
        }


def load_cnn_dailymail(batch_size,
                        max_length):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    train_dataset = SummarizationDataset(
        articles=dataset["train"]["article"],
        summaries=dataset["train"]["highlights"],
        tokenizer=tokenizer,
        max_length=max_length)
    val_dataset = SummarizationDataset(
        articles=dataset["validation"]["article"],
        summaries=dataset["validation"]["highlights"],
        tokenizer=tokenizer,
        max_length=max_length)
    
    test_dataset = SummarizationDataset(
        articles=dataset["test"]["article"],
        summaries=dataset["test"]["highlights"],
        tokenizer=tokenizer,
        max_length=max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader