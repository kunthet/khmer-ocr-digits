"""
Attention mechanisms for sequence-to-sequence modeling in OCR.

Implements Bahdanau (additive) attention for focusing on relevant
image regions during character sequence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    Computes attention weights between decoder hidden states and 
    encoder feature sequences to focus on relevant image regions.
    
    Reference: Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, 
                 encoder_hidden_size: int,
                 decoder_hidden_size: int, 
                 attention_size: int = 256):
        """
        Initialize Bahdanau attention.
        
        Args:
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states  
            attention_size: Size of attention projection layer
        """
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size
        
        # Linear projections for encoder and decoder states
        self.encoder_projection = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        
        # Attention weight computation
        self.attention_weight = nn.Linear(attention_size, 1, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.xavier_uniform_(self.attention_weight.weight)
    
    def forward(self, 
                encoder_states: torch.Tensor,
                decoder_state: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_states: Encoder hidden states [batch_size, seq_len, encoder_hidden_size]
            decoder_state: Current decoder hidden state [batch_size, decoder_hidden_size]
            encoder_mask: Mask for encoder states [batch_size, seq_len] (optional)
            
        Returns:
            Tuple of (context_vector, attention_weights)
            - context_vector: [batch_size, encoder_hidden_size]
            - attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, encoder_hidden_size = encoder_states.size()
        
        # Project encoder states
        # [batch_size, seq_len, attention_size]
        projected_encoder = self.encoder_projection(encoder_states)
        
        # Project decoder state and expand to match encoder sequence length
        # [batch_size, attention_size] -> [batch_size, 1, attention_size] -> [batch_size, seq_len, attention_size]
        projected_decoder = self.decoder_projection(decoder_state).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute attention scores
        # [batch_size, seq_len, attention_size]
        attention_input = torch.tanh(projected_encoder + projected_decoder)
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        attention_scores = self.attention_weight(attention_input).squeeze(-1)
        
        # Apply mask if provided
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(encoder_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector
        # [batch_size, seq_len, encoder_hidden_size] * [batch_size, seq_len, 1] -> [batch_size, encoder_hidden_size]
        context_vector = torch.sum(encoder_states * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    
    Alternative attention mechanism using dot-product similarity
    with scaling factor for numerical stability.
    """
    
    def __init__(self, 
                 model_dim: int,
                 dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            model_dim: Model dimension for scaling
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        self.model_dim = model_dim
        self.scale_factor = model_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, query_len, model_dim]
            key: Key tensor [batch_size, key_len, model_dim] 
            value: Value tensor [batch_size, key_len, model_dim]
            mask: Attention mask [batch_size, query_len, key_len] (optional)
            
        Returns:
            Tuple of (attended_output, attention_weights)
            - attended_output: [batch_size, query_len, model_dim]
            - attention_weights: [batch_size, query_len, key_len]
        """
        # Compute attention scores
        # [batch_size, query_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, value)
        
        return attended_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Extends single attention to multiple attention heads for
    capturing different types of relationships.
    """
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            model_dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        
        # Output projection
        self.output_projection = nn.Linear(model_dim, model_dim)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, query_len, model_dim]
            key: Key tensor [batch_size, key_len, model_dim]
            value: Value tensor [batch_size, key_len, model_dim]
            mask: Attention mask [batch_size, query_len, key_len] (optional)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.query_projection(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand mask for multiple heads if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply attention for each head
        attended_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.model_dim
        )
        
        # Final linear projection
        output = self.output_projection(attended_output)
        output = self.dropout(output)
        
        # Average attention weights across heads for visualization
        attention_weights = attention_weights.mean(dim=1)
        
        return output, attention_weights 