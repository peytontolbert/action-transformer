import torch
import torch.nn as nn
from typing import Optional, Tuple

from .embeddings import StateActionEmbedding
from .transformer import TransformerEncoder
from ..utils.masking import create_padding_mask, create_causal_mask, combine_masks
from ..losses.action_losses import ActionPredictionLoss

class ActionTransformer(nn.Module):
    """Transformer model for sequential decision making."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        discrete_actions: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            embed_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            discrete_actions: Whether actions are discrete
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.discrete_actions = discrete_actions
        
        # Create embeddings
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        if discrete_actions:
            self.action_embedding = nn.Embedding(action_dim, embed_dim)
        else:
            self.action_embedding = nn.Linear(action_dim, embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output heads
        if discrete_actions:
            self.action_head = nn.Linear(embed_dim, action_dim)
        else:
            # For continuous actions, output mean and log std
            self.action_mean = nn.Linear(embed_dim, action_dim)
            self.action_log_std = nn.Linear(embed_dim, action_dim)
        
        self.value_head = nn.Linear(embed_dim, 1)
        
        # Initialize loss function
        self.action_loss_fn = ActionPredictionLoss(discrete_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer.
        
        Args:
            states: State tensor of shape (batch_size, seq_len, state_dim)
            actions: Optional action tensor
            seq_lens: Optional sequence lengths of shape (batch_size,)
            
        Returns:
            action_preds: Predicted next actions (logits for discrete, mean/std for continuous)
            values: Predicted state values
        """
        batch_size, seq_len, _ = states.shape
        device = states.device
        
        # Create attention mask
        if seq_lens is not None:
            mask = torch.arange(seq_len, device=device)[None, :] < seq_lens[:, None]  # (batch_size, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, seq_len, seq_len)
            # Add causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask = mask & ~causal_mask.unsqueeze(0).unsqueeze(0)  # (batch_size, 1, seq_len, seq_len)
            # Expand to num_heads
            mask = mask.expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
        else:
            mask = None
        
        # Embed states and actions
        state_embeds = self.state_embedding(states)
        if actions is not None:
            action_embeds = self.action_embedding(actions)
            hidden = state_embeds + action_embeds
        else:
            hidden = state_embeds
        
        # Apply transformer
        hidden = self.transformer(hidden, mask)
        
        # Predict next actions and values
        if self.discrete_actions:
            action_preds = self.action_head(hidden)
        else:
            # For continuous actions, output mean and log std
            action_mean = self.action_mean(hidden)
            action_log_std = self.action_log_std(hidden)
            action_preds = torch.cat([action_mean, action_log_std], dim=-1)
        
        values = self.value_head(hidden).squeeze(-1)
        
        return action_preds, values
    
    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        pred_values: torch.Tensor,
        target_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute losses for actions and values.
        
        Args:
            pred_actions: Predicted actions
            target_actions: Target actions
            pred_values: Predicted values
            target_values: Target values
            mask: Optional mask
            
        Returns:
            Tuple of action loss and value loss
        """
        # Compute action loss
        if self.discrete_actions:
            action_loss = self.action_loss_fn.discrete_action_loss(
                pred_actions, target_actions, mask)
        else:
            action_loss = self.action_loss_fn.continuous_action_loss(
                pred_actions, target_actions, mask)
            
        # Compute value loss
        value_loss = self.action_loss_fn.value_loss(pred_values, target_values, mask)
        
        return action_loss, value_loss
    
    def predict_action(
        self,
        states: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Predict next action for given states.
        
        Args:
            states: State tensor of shape (batch_size, seq_len, state_dim)
            temperature: Temperature for sampling discrete actions
            
        Returns:
            Predicted actions
        """
        action_preds, _ = self.forward(states)
        
        # Get predictions for last timestep
        action_preds = action_preds[:, -1]
        
        if self.discrete_actions:
            # Apply temperature scaling and sample
            logits = action_preds / temperature
            action_probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        else:
            # For continuous actions, split into mean and log_std and return mean
            action_mean, action_log_std = torch.chunk(action_preds, 2, dim=-1)
            actions = action_mean
            
        return actions 