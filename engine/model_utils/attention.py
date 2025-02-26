import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from typing import Optional 

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism similar to what is described in 'Attention is all you need'
    You linearly project each of the queries, keys and values with different learned projections
    And in each head, self attention is applied, the result are concatenated and projected to the output
    
    This basically helps the model to jointly attend to information from different positions, with different represenatational subspaces
    For example, if we have a sentence "He went to the bank to get some money, and later went for a walk along the river bank"
    The model should be able to attend to the words "bank" in different ways:
    - "bank" as a place to get money
    - "bank" as a place to walk along the river
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # dk is the dimension of each head's key, query and value 
        
        # linear layers for queries, keys and values projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # e.g 8 heads * 64 dk -> 512 d model
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def attention(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The popular attention mechanism 
        
        Args: 
            query: [batch size, num heads, seq len q, d_k]
            key: [batch size, num heads, seq len k, d_k]
            value: [batch size, num heads, seq len v, d_k]
            mask: [batch size, 1, seq len q, seq len k]
        Returns:
            Tensor of shape [batch size, num heads, seq len, d_k]
        """
        similarity = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(similarity, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_scores = torch.matmul(attention_weights, v)
        return attention_scores, attention_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch size, seq len q, d model]
            key: [batch size, seq len k, d model]
            value: [batch size, seq len v, d model]
            mask: [batch size, seq len q, seq len k]
        
        Returns:
            Two tensors output and attention weights
            output: [batch size, num heads, seq len q, d model]
            attention weights: [batch size, num heads, seq len q, seq len k]
        """
        batch_size = query.size(0)
        
        # view each of q,k,v to [batch size, seq len, num heads, dk]
        # so if (32, 10, 512) -> (32, 10, 8, 64)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # chunk it for each head
        # why transpose?
        # [batch size, seq len, num heads, dk] -> [batch size, num heads, seq len, dk]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_scores, attention_weights = self.attention(q, k, v, mask)
        
        # combine the heads (sounds kinda weird)
        # lets break down the math 
        # output -> [batch size, num heads, seq len, dk] 
        # revoke the transpose -> [batch size, seq len, num heads, dk]
        # view it back to [batch size, seq len, num heads * dk]
        combined = attention_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # project it back to d_model
        projected = self.out_proj(combined)
        
        return projected, attention_weights