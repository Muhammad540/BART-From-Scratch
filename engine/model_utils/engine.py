import torch
import torch.nn as nn
from .transformer import BartEncoder, BartDecoder

class BART(nn.Module):
    """
    BART: Denoising Seq-to-Seq Pre-training for 
    Natural Language Generation, Translation, and Comprehension
    """
    def __init__(self,
                 config):
        super().__init__()
        
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)
        
    def forward(self,
                input_ids,
                decoder_input_ids,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            decoder_input_ids: (batch_size, seq_len)
            encoder_padding_mask: (batch_size, seq_len)
            decoder_padding_mask: (batch_size, seq_len)
            decoder_causal_mask: (batch_size, seq_len, seq_len)
        """
        encoder_output = self.encoder(input_ids, 
                                      attention_mask=encoder_padding_mask)
        decoder_output = self.decoder(decoder_input_ids,
                                      encoder_output=encoder_output,
                                      self_attention_mask=decoder_causal_mask,
                                      cross_attention_mask=decoder_padding_mask)
        return decoder_output