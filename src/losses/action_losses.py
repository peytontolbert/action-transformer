import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class ActionPredictionLoss:
    """Loss functions for action prediction and value estimation."""
    
    def __init__(self, discrete_actions: bool = True):
        """
        Args:
            discrete_actions: Whether actions are discrete (True) or continuous (False)
        """
        super().__init__()
        self.discrete_actions = discrete_actions
    
    def discrete_action_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross entropy loss for discrete actions.
        
        Args:
            pred_actions: Predicted action logits of shape (batch_size, num_agents, seq_len, action_dim)
            target_actions: Target action indices of shape (batch_size, num_agents, seq_len)
            mask: Optional mask of shape (batch_size, num_agents, seq_len)
            
        Returns:
            Loss value
        """
        if mask is not None:
            # Expand mask to match prediction shape for sequence dimension
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, pred_actions.size(-1))
            
            # Zero out predictions for masked elements
            pred_actions = pred_actions * mask
            
            # Compute loss
            loss = F.nll_loss(
                pred_actions.reshape(-1, pred_actions.size(-1)),
                target_actions.reshape(-1),
                reduction='sum'
            )
            
            # Normalize by number of valid elements
            num_valid = mask[:,:,:,0].sum()
            loss = loss / (num_valid + 1e-8)
        else:
            loss = F.nll_loss(
                pred_actions.reshape(-1, pred_actions.size(-1)),
                target_actions.reshape(-1)
            )
        
        return loss
    
    def continuous_action_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute MSE loss for continuous actions.
        
        Args:
            pred_actions: Predicted action parameters (mean, log_std) of shape (batch_size, num_agents, seq_len, 2 * action_dim)
            target_actions: Target actions of shape (batch_size, num_agents, seq_len, action_dim)
            mask: Optional mask of shape (batch_size, num_agents, seq_len)
            
        Returns:
            Loss value
        """
        # Split predictions into mean and log_std
        action_dim = target_actions.size(-1)
        pred_mean, pred_log_std = torch.split(pred_actions, action_dim, dim=-1)
        pred_std = torch.exp(pred_log_std)
        
        # Compute negative log likelihood
        diff = (target_actions - pred_mean) / (pred_std + 1e-8)
        loss = 0.5 * (diff.pow(2) + 2 * pred_log_std + math.log(2 * math.pi))
        loss = loss.sum(dim=-1)  # Sum over action dimensions
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def value_loss(
        self,
        pred_values: torch.Tensor,
        target_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute MSE loss for value predictions.
        
        Args:
            pred_values: Predicted values of shape (batch_size, num_agents, seq_len)
            target_values: Target values of shape (batch_size, num_agents, seq_len)
            mask: Optional mask of shape (batch_size, num_agents, seq_len)
            
        Returns:
            Loss value
        """
        if mask is not None:
            # Flatten all tensors
            pred_flat = pred_values.reshape(-1)
            target_flat = target_values.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # Get only valid elements
            valid_pred = pred_flat[mask_flat]
            valid_target = target_flat[mask_flat]
            
            if valid_pred.size(0) > 0:
                # Compute MSE loss only on valid elements
                loss = F.mse_loss(valid_pred, valid_target)
            else:
                # Return zero loss if no valid elements
                loss = torch.tensor(0.0, device=pred_values.device)
        else:
            # If no mask, compute on all elements
            loss = F.mse_loss(pred_values, target_values)
            
        return loss 