import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, num_heads, seq_len, seq_len)
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # Shape: (batch_size, seq_len, 3 * embed_dim)
        
        # Reshape qkv to separate heads and qkv dimensions
        # First split into 3 tensors
        qkv = qkv.chunk(3, dim=-1)  # List of 3 tensors of shape (batch_size, seq_len, embed_dim)
        q, k, v = qkv
        
        # Then reshape each tensor to include heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            # Verify mask shape matches attention scores
            assert mask.size(1) == self.num_heads, f"Mask should have {self.num_heads} heads, got {mask.size(1)}"
            # Use a large negative value instead of -inf to avoid NaN in softmax
            attn = attn.masked_fill(~mask, -1e4)
        
        # Apply softmax with better numerical stability
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # Subtract max for stability
        attn = F.softmax(attn, dim=-1)
        
        # Zero out attention to padded tokens
        if mask is not None:
            attn = attn.masked_fill(~mask, 0.0)
            
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        
        return out

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            hidden_dim: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self attention
        attn_out = self.self_attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class TransformerEncoder(nn.Module):
    """Full transformer encoder with multiple layers."""
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        if mask is not None:
            # Ensure mask has correct shape (batch_size, num_heads, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            assert mask.size(1) == self.num_heads, f"Mask should have {self.num_heads} heads, got {mask.size(1)}"
        
        for layer in self.layers:
            x = layer(x, mask)
        return x 