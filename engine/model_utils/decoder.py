import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """
    A single decoder block for BART:
    - Masked multi-head attention (No peeking)
    - Multi-head cross attention with encoder outputs
    - Feed forward NN
    - layer Normalization 
    - Residual connections
    NOTE: Uses Pre-LN arch like the encoder also uses GELU
    """
    def __init__(self, 
                 config):
        super().__init__()
        
        self.masked_self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.decoder_attention_dropout,
        )
        
        self.cross_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.decoder_attention_dropout,
        )
        
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.decoder_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_ff_dim, config.d_model),
            nn.Dropout(config.dropout)
        )
    def forward(self,
                x,
                encoder_output,
                self_attention_mask=None,
                cross_attention_mask=None):
        """
        Args:
            encoder_output: (batch_size, seq_len, d_model)
            self_attention_mask: (batch_size, seq_len, seq_len)
            cross_attention_mask: (batch_size, seq_len, seq_len)
        Returns:
            (batch_size, seq_len, d_model)
        """
        residual = x
        x = self.layer_norm1(x)
        x = self.masked_self_attention(
            query=x,
            key=x,
            value=x,
            mask=self_attention_mask
        )
        x = x + residual
        
        # important thing to notice here is that the:
        # query is coming from the decoder itself
        # key and value are coming from the encoder
        # so basically decoder is looking at itself and attending to the encoder outputs
        # this is why it is called "cross attention"
        residual = x
        x = self.layer_norm2(x)
        x = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask = cross_attention_mask
        )
        x = x + residual 
        
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = x + residual

        return x
        
        