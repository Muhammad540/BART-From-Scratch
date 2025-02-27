from datasets import load_dataset 
import torch 
from torch.utils.data import Dataset, DataLoader

class SummarizationDataset(Dataset):
    def __init__(self,
                 articles,
                 summaries,
                 tokenizer,
                 max_length):
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
        
        article_mask = article_encodings["attention_mask"]
        
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
            "decoder_input_ids": torch.tensor(summary_encodings["input_ids"][:,:-1]),
            "labels": torch.tensor(summary_encodings["input_ids"][:,1:])
        }