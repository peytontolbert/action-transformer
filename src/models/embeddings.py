import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer sequences."""
    
    def __init__(self, embed_dim: int, max_seq_len: int):
        """
        Args:
            embed_dim: Dimension of the embedding
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize positional embeddings using sine-cosine pattern."""
        position = torch.arange(self.pos_embedding.size(1)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pos_embedding.size(2), 2) * 
                           (-math.log(10000.0) / self.pos_embedding.size(2)))
        self.pos_embedding.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pos_embedding.data[0, :, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pos_embedding[:, :x.size(1), :]

class StateActionEmbedding(nn.Module):
    """Embedding layer for states and actions."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        max_seq_len: int,
        discrete_actions: bool = True
    ):
        """
        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of action space
            embed_dim: Target embedding dimension
            max_seq_len: Maximum sequence length
            discrete_actions: Whether actions are discrete
        """
        super().__init__()
        
        # State embedding
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        
        # Action embedding
        if discrete_actions:
            self.action_embedding = nn.Embedding(action_dim, embed_dim)
        else:
            self.action_embedding = nn.Linear(action_dim, embed_dim)
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            states: State tensor of shape (batch_size, seq_len, state_dim)
            actions: Action tensor of shape (batch_size, seq_len) for discrete
                    or (batch_size, seq_len, action_dim) for continuous
        Returns:
            Combined embedded tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Embed states
        state_embeddings = self.state_embedding(states)
        
        # Combine with actions if provided
        if actions is not None:
            action_embeddings = self.action_embedding(actions)
            embeddings = state_embeddings + action_embeddings
        else:
            embeddings = state_embeddings
            
        # Apply layer norm and positional encoding
        embeddings = self.layer_norm(embeddings)
        embeddings = self.pos_encoding(embeddings)
        
        return embeddings 