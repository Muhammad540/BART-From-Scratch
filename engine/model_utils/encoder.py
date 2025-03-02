import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from engine.model_utils.attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    """
    A simple/single encoder block:
    - Multi-head attention mechanism
    - Feed-forward neural network
    - Layer normalization
    NOTE: GELU instead of ReLU is used as per the BART paper
    """
    def __init__(self, 
                 config):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.encoder_ff_dim, config.d_model),
            nn.Dropout(config.dropout) 
        )
    def forward(self, 
                x, 
                attention_mask=None):
        # tidbit: residual connection is used to avoid the vanishing gradient problem
        # also the way we have implemented this is called "pre layer normalization" since BART paper uses this 
        residual = x
        
        # attention with residual connection
        x = self.layer_norm1(x)
        
        if attention_mask is not None:
            # convert from (batch size, seq len) to (batch size, 1, 1, seq len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # now expand to (batch size, 1, seq len, seq len)
            seq_len = attention_mask.size(3)
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
        
        x, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=attention_mask
        )
        
        x = x + residual
        # feedforward with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual 
        return x