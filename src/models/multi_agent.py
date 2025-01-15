import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import torch.nn.functional as F

from .embeddings import StateActionEmbedding
from .transformer import TransformerEncoder
from ..utils.masking import create_padding_mask, create_causal_mask, combine_masks
from ..losses.action_losses import ActionPredictionLoss

class AgentEmbedding(nn.Module):
    """Embedding layer for agent identities."""
    
    def __init__(self, num_agents: int, embed_dim: int):
        """
        Args:
            num_agents: Maximum number of agents
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.agent_embedding = nn.Embedding(num_agents, embed_dim)
        
    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_ids: Tensor of agent IDs of shape (batch_size, num_agents)
            
        Returns:
            Agent embeddings of shape (batch_size, num_agents, embed_dim)
        """
        return self.agent_embedding(agent_ids)

class MultiAgentTransformer(nn.Module):
    """Multi-agent transformer for decentralized execution with centralized training."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        max_num_agents: int,
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
            max_num_agents: Maximum number of agents
            discrete_actions: Whether actions are discrete
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.discrete_actions = discrete_actions
        self.num_heads = num_heads
        
        # Create embeddings
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        if discrete_actions:
            self.action_embedding = nn.Embedding(action_dim, embed_dim)
        else:
            self.action_embedding = nn.Linear(action_dim, embed_dim)
        
        # Agent embedding
        self.agent_embedding = AgentEmbedding(max_num_agents, embed_dim)
        
        # Temporal transformer (processes each agent's sequence)
        self.temporal_transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Agent transformer (processes interactions between agents)
        self.agent_transformer = TransformerEncoder(
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
        agent_ids: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-agent transformer.
        
        Args:
            states: State tensor of shape (batch_size, num_agents, seq_len, state_dim)
            agent_ids: Agent IDs of shape (batch_size, num_agents)
            actions: Optional action tensor
            seq_lens: Optional sequence lengths of shape (batch_size, num_agents)
            agent_mask: Optional mask of shape (batch_size, num_agents)
        """
        batch_size, num_agents, seq_len, _ = states.shape
        device = states.device
        
        # Create attention masks
        if agent_mask is None:
            agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool, device=device)
            
        # Create temporal attention mask
        if seq_lens is not None:
            # Create causal mask for each sequence
            temporal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            temporal_mask = ~temporal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            
            # Add sequence length masking
            seq_mask = torch.arange(seq_len, device=device)[None, None, :] < seq_lens[:, :, None]  # (batch_size, num_agents, seq_len)
            seq_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(3)  # (batch_size, num_agents, seq_len, seq_len)
            
            temporal_mask = temporal_mask & seq_mask
            temporal_mask = temporal_mask.unsqueeze(2)  # Add head dimension
            temporal_mask = temporal_mask.expand(-1, -1, self.num_heads, -1, -1)  # (batch_size, num_agents, num_heads, seq_len, seq_len)
        else:
            temporal_mask = None
        
        # Create agent attention mask
        agent_attn_mask = agent_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, num_agents)
        agent_attn_mask = agent_attn_mask & agent_attn_mask.transpose(-1, -2)  # (batch_size, 1, num_agents, num_agents)
        agent_attn_mask = agent_attn_mask.expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, num_agents, num_agents)
        
        # Process states and actions
        state_embeds = self.state_embedding(states)  # (batch_size, num_agents, seq_len, embed_dim)
        if actions is not None:
            if self.discrete_actions:
                action_embeds = self.action_embedding(actions)
            else:
                action_embeds = self.action_embedding(actions)
            hidden = state_embeds + action_embeds
        else:
            hidden = state_embeds
            
        # Add agent embeddings
        agent_embeds = self.agent_embedding(agent_ids)  # (batch_size, num_agents, embed_dim)
        hidden = hidden + agent_embeds.unsqueeze(2)  # Add to each timestep
        
        # Apply agent mask to embeddings
        hidden = hidden * agent_mask.unsqueeze(-1).unsqueeze(-1)
        
        # First apply temporal transformer to each agent's sequence
        temporal_hidden = hidden.view(batch_size * num_agents, seq_len, self.embed_dim)
        if temporal_mask is not None:
            temporal_mask = temporal_mask.view(batch_size * num_agents, self.num_heads, seq_len, seq_len)
        temporal_hidden = self.temporal_transformer(temporal_hidden, temporal_mask)
        temporal_hidden = temporal_hidden.view(batch_size, num_agents, seq_len, self.embed_dim)
        
        # Then apply agent transformer
        agent_hidden = temporal_hidden.transpose(1, 2)  # (batch_size, seq_len, num_agents, embed_dim)
        agent_hidden = agent_hidden.reshape(batch_size * seq_len, num_agents, self.embed_dim)
        
        # Properly reshape agent attention mask for each timestep
        agent_attn_mask = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)  # (batch_size, num_agents, num_agents)
        agent_attn_mask = agent_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, num_agents, num_agents)
        agent_attn_mask = agent_attn_mask.repeat(seq_len, 1, 1, 1)  # (batch_size * seq_len, num_heads, num_agents, num_agents)
        
        # Apply agent transformer with proper masking
        agent_hidden = self.agent_transformer(agent_hidden, agent_attn_mask)
        agent_hidden = agent_hidden.view(batch_size, seq_len, num_agents, self.embed_dim)
        agent_hidden = agent_hidden.transpose(1, 2)  # (batch_size, num_agents, seq_len, embed_dim)
        
        # Predict next actions and values
        if self.discrete_actions:
            # Get raw logits
            logits = self.action_head(agent_hidden)
            
            # Create attention mask for inactive agents
            mask_expanded = agent_mask.unsqueeze(-1).expand(-1, -1, seq_len).unsqueeze(-1).expand(-1, -1, -1, self.action_dim)
            
            # Instead of using -inf, use a large negative value that won't cause numerical issues
            logits = torch.where(mask_expanded, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))
            
            # Apply log_softmax after masking
            action_preds = F.log_softmax(logits, dim=-1)
            
            # Zero out predictions for inactive agents after softmax
            action_preds = action_preds.masked_fill(~mask_expanded, 0.0)
        else:
            action_mean = self.action_mean(agent_hidden)
            action_preds = action_mean
        
        values = self.value_head(agent_hidden).squeeze(-1)
        
        # Apply agent mask to outputs
        values = values * agent_mask.unsqueeze(-1)
        agent_hidden = agent_hidden * agent_mask.unsqueeze(-1).unsqueeze(-1)
        
        return action_preds, values, agent_hidden
    
    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        pred_values: torch.Tensor,
        target_values: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute losses for actions and values.
        
        Args:
            pred_actions: Predicted actions
            target_actions: Target actions
            pred_values: Predicted values
            target_values: Target values
            hidden_states: Hidden states for computing log_std in continuous case
            mask: Optional temporal mask
            agent_mask: Optional agent mask
            
        Returns:
            Tuple of action loss and value loss
        """
        # Create combined mask from temporal and agent masks
        if agent_mask is not None:
            # Expand agent mask to match sequence dimension
            agent_mask = agent_mask.unsqueeze(-1).expand(-1, -1, pred_actions.size(2))  # (batch_size, num_agents, seq_len)
            if mask is not None:
                # Combine with temporal mask
                mask = mask & agent_mask
            else:
                mask = agent_mask
                
            # Ensure at least one agent is active
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_actions.device), torch.tensor(0.0, device=pred_actions.device)
        
        # Compute action loss
        if self.discrete_actions:
            action_loss = self.action_loss_fn.discrete_action_loss(
                pred_actions, target_actions, mask)
        else:
            # For continuous actions, compute log_std here
            action_log_std = self.action_log_std(hidden_states)
            action_preds = torch.cat([pred_actions, action_log_std], dim=-1)
            action_loss = self.action_loss_fn.continuous_action_loss(
                action_preds, target_actions, mask)
            
        # Compute value loss
        value_loss = self.action_loss_fn.value_loss(pred_values, target_values, mask)
        
        return action_loss, value_loss 