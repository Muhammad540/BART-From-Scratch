import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from engine.model_utils.encoder import EncoderBlock
from engine.model_utils.decoder import DecoderBlock
from engine.model_utils.embeddings import TokenEmbedding, PositionalEncoding

class BartEncoder(nn.Module):
    """
    Fully stacked with encoder blocks 
    A single encoder block is defined in encoder.py
    """
    def __init__(self, 
                 config):
        super().__init__()
        
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model
        )
        
        self.positional_encoding = PositionalEncoding(
            embedding_dim=config.d_model,
            max_sequence_length=config.max_seq_len,
            dropout_prob=config.dropout
        )
        
        # stack the encoder blocks
        self.encoder_stacked_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.encoder_layers)
        ])
    
    def forward(self,
                x,
                attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.encoder_stacked_layers:
            x = layer(x, attention_mask)
        
        return x

class BartDecoder(nn.Module):
    """
    Fully stacked with decoder blocks 
    A single decoder block is defined in decoder.py
    """
    def __init__(self,
                 config):
        super().__init__()
        
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model
        )
        
        self.positional_encoding = PositionalEncoding(
            embedding_dim=config.d_model,
            max_sequence_length=config.max_seq_len,
            dropout_prob=config.dropout
        )
        
        self.decoder_stacked_layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.decoder_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.final_dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self,
                x,
                encoder_output,
                self_attention_mask=None,
                cross_attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len)
            encoder_output: (batch_size, seq_len, d_model)
            self_attention_mask: (batch_size, seq_len, seq_len)
            cross_attention_mask: (batch_size, seq_len, seq_len)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.decoder_stacked_layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
        
        x = self.final_layer_norm(x)
        x = self.final_dropout(x)
        x = self.output_projection(x)
        
        return x