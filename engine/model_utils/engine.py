import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Tie the decoder embedding and output projection layer weights 
        # Reason ? 
        # 1. Reduce model params
        # 2. Act as a regularizer 
        # 3. Since it is just an inverse operation (embedding: input tokens -> embedding, projection: embedding -> output tokens)
        self.decoder.output_projection.weight = self.decoder.embedding.embedding.weight
        
    def forward(self,
                input_ids,
                decoder_input_ids,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            decoder_input_ids: (batch_size, seq_len)
            encoder_padding_mask: (batch_size, seq_len)
            decoder_padding_mask: (batch_size, seq_len)
            decoder_causal_mask: (batch_size, seq_len, seq_len)
            labels: (batch_size, seq_len)
        Returns:
            (outputs, loss) if labels are provided else (output tokens)
            outputs: (batch_size, seq_len, vocab_size)
        """
        encoder_output = self.encoder(input_ids, 
                                      attention_mask=encoder_padding_mask)
        decoder_output = self.decoder(decoder_input_ids,
                                      encoder_output=encoder_output,
                                      self_attention_mask=decoder_causal_mask,
                                      cross_attention_mask=decoder_padding_mask)
        
        if labels is not None:
            loss = F.cross_entropy(
                # (batch_size * seq_len, vocab_size)
                decoder_output.view(-1, decoder_output.size(-1)),
                # (batch_size * seq_len)
                labels.view(-1),
                ignore_index=-100
            )
            return decoder_output, loss
        
        return decoder_output