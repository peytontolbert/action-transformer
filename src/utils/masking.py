import torch
from typing import Optional, Tuple

def create_padding_mask(
    seq_lens: torch.Tensor,
    max_len: int
) -> torch.Tensor:
    """Creates padding mask from sequence lengths."""
    # Returns shape (batch_size, seq_len)
    return torch.arange(max_len, device=seq_lens.device)[None, :] < seq_lens[:, None]

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Creates a causal mask for self-attention."""
    # Returns shape (seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # Invert to get the causal mask

def combine_masks(padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """Combines padding and causal masks."""
    # padding_mask: (batch_size, seq_len)
    # causal_mask: (seq_len, seq_len)
    # Returns: (batch_size, seq_len, seq_len)
    return padding_mask.unsqueeze(-1) & causal_mask

def get_sequence_lengths(
    sequences: torch.Tensor,
    padding_value: int = 0
) -> torch.Tensor:
    """Get lengths of sequences that may include padding.
    
    Args:
        sequences: Tensor of shape (batch_size, seq_len, *)
        padding_value: Value used for padding
        
    Returns:
        Tensor of sequence lengths of shape (batch_size,)
    """
    # Create mask where True indicates non-padding
    mask = sequences != padding_value
    
    # Sum over sequence length dimension and any additional dimensions
    lengths = mask.sum(dim=tuple(range(1, mask.dim())))
    
    return lengths 