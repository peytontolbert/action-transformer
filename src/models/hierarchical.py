import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .embeddings import StateActionEmbedding
from .transformer import TransformerEncoder
from ..utils.masking import create_padding_mask, create_causal_mask, combine_masks
from ..losses.action_losses import ActionPredictionLoss

class TemporalAbstraction(nn.Module):
    """Temporal abstraction layer for hierarchical decision making."""
    
    def __init__(
        self,
        embed_dim: int,
        num_options: int,
        option_duration: int
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_options: Number of abstract options
            option_duration: Duration of each option in time steps
        """
        super().__init__()
        self.num_options = num_options
        self.option_duration = option_duration
        
        # Option embedding
        self.option_embedding = nn.Embedding(num_options, embed_dim)
        
        # Option termination prediction
        self.termination_head = nn.Linear(embed_dim, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        current_option: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, embed_dim)
            current_option: Optional current option indices of shape (batch_size,)
            
        Returns:
            Tuple of:
                - Option-augmented states of shape (batch_size, seq_len, embed_dim)
                - Termination probabilities of shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if current_option is not None:
            # Add option embedding to states
            option_embeds = self.option_embedding(current_option)  # (batch_size, embed_dim)
            hidden_states = hidden_states + option_embeds.unsqueeze(1)
            
        # Predict termination probabilities
        termination_logits = self.termination_head(hidden_states).squeeze(-1)
        termination_probs = torch.sigmoid(termination_logits)
        
        return hidden_states, termination_probs

class HierarchicalTransformer(nn.Module):
    """Hierarchical transformer with temporal abstraction."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        num_options: int,
        option_duration: int,
        discrete_actions: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of action space
            embed_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension in feed-forward network
            max_seq_len: Maximum sequence length
            num_options: Number of abstract options
            option_duration: Duration of each option
            discrete_actions: Whether actions are discrete
            dropout: Dropout probability
        """
        super().__init__()
        
        # Save parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.num_options = num_options
        self.option_duration = option_duration
        self.num_heads = num_heads
        
        # State-action embedding
        self.embeddings = StateActionEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            discrete_actions=discrete_actions
        )
        
        # High-level policy (option selection)
        self.high_level_transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers // 2,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.option_head = nn.Linear(embed_dim, num_options)
        
        # Temporal abstraction
        self.temporal_abstraction = TemporalAbstraction(
            embed_dim=embed_dim,
            num_options=num_options,
            option_duration=option_duration
        )
        
        # Low-level policy (action selection)
        self.low_level_transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers // 2,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Action prediction head
        if discrete_actions:
            self.action_head = nn.Linear(embed_dim, action_dim)
        else:
            # For continuous actions, output mean and log_std
            self.action_mean = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.action_log_std = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
        # Value prediction head
        self.value_head = nn.Linear(embed_dim, 1)
        
        # Loss functions
        self.action_loss_fn = ActionPredictionLoss()
        
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        options: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            states: State tensor of shape (batch_size, seq_len, state_dim)
            actions: Optional action tensor
            options: Optional current options of shape (batch_size,)
            seq_lens: Optional sequence lengths
            
        Returns:
            Dictionary containing:
                - action_preds: Action predictions
                - option_preds: Option predictions
                - termination_probs: Option termination probabilities
                - values: Value predictions
        """
        batch_size, seq_len, _ = states.shape
        device = states.device
        
        # Create attention masks
        mask = create_causal_mask(seq_len)  # (seq_len, seq_len)
        if seq_lens is not None:
            padding_mask = create_padding_mask(seq_lens, seq_len)  # (batch_size, seq_len)
            mask = combine_masks(padding_mask, mask)  # (batch_size, seq_len, seq_len)
            # Add head dimension and expand
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Embed inputs
        embeddings = self.embeddings(states, actions)
        
        # High-level policy
        high_level_hidden = self.high_level_transformer(embeddings, mask)
        option_logits = self.option_head(high_level_hidden)
        
        # Temporal abstraction
        abstracted_states, termination_probs = self.temporal_abstraction(
            high_level_hidden, options)
        
        # Low-level policy
        low_level_hidden = self.low_level_transformer(abstracted_states, mask)
        
        # Predict actions and values
        if self.discrete_actions:
            action_preds = self.action_head(low_level_hidden)
        else:
            # For continuous actions, only output mean for predictions
            # We'll use log_std only during loss computation
            action_mean = self.action_mean(low_level_hidden)
            action_preds = action_mean  # Shape: (batch_size, seq_len, action_dim)
            
        values = self.value_head(low_level_hidden).squeeze(-1)
        
        return {
            'action_preds': action_preds,
            'option_preds': option_logits,
            'termination_probs': termination_probs,
            'values': values,
            'hidden_states': low_level_hidden  # Include hidden states for loss computation
        }
    
    def compute_loss(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        target_options: Optional[torch.Tensor] = None,
        target_terminations: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # Action loss
        if self.discrete_actions:
            losses['action_loss'] = self.action_loss_fn.discrete_action_loss(
                pred_dict['action_preds'], target_actions, mask)
        else:
            # For continuous actions, compute both mean and log_std here
            action_mean = pred_dict['action_preds']
            # Recompute log_std from the same hidden states
            action_log_std = self.action_log_std(pred_dict['hidden_states'])
            action_preds = torch.cat([action_mean, action_log_std], dim=-1)
            losses['action_loss'] = self.action_loss_fn.continuous_action_loss(
                action_preds, target_actions, mask)
            
        # Option loss
        if target_options is not None:
            losses['option_loss'] = self.action_loss_fn.discrete_action_loss(
                pred_dict['option_preds'], target_options, mask)
            
        # Termination loss
        if target_terminations is not None:
            losses['termination_loss'] = nn.BCELoss()(
                pred_dict['termination_probs'], target_terminations.float())
            
        # Value loss
        if target_values is not None:
            losses['value_loss'] = self.action_loss_fn.value_loss(
                pred_dict['values'], target_values, mask)
            
        return losses 
    
    def predict_action(
        self,
        states: torch.Tensor,
        options: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Predict next action for given states.
        
        Args:
            states: State tensor of shape (batch_size, seq_len, state_dim)
            options: Optional current options of shape (batch_size,)
            temperature: Temperature for sampling discrete actions
            
        Returns:
            Predicted actions
        """
        outputs = self.forward(states, options=options)
        action_preds = outputs['action_preds']
        
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