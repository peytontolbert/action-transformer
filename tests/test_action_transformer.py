import torch
import pytest
from src.models import ActionTransformer

def test_discrete_action_transformer():
    """Test ActionTransformer with discrete actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 10
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    batch_size = 8
    seq_len = 20
    
    # Create model
    model = ActionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        discrete_actions=True
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    # Test forward pass
    action_preds, values = model(states, actions, seq_lens)
    assert action_preds.shape == (batch_size, seq_len, action_dim)
    assert values.shape == (batch_size, seq_len)
    
    # Test loss computation
    action_loss, value_loss = model.compute_loss(
        action_preds,
        actions,
        values,
        torch.randn_like(values)
    )
    assert isinstance(action_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)
    
    # Test action prediction
    pred_actions = model.predict_action(states)
    assert pred_actions.shape == (batch_size,)
    assert (pred_actions >= 0).all() and (pred_actions < action_dim).all()

def test_continuous_action_transformer():
    """Test ActionTransformer with continuous actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 6
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    batch_size = 8
    seq_len = 20
    
    # Create model
    model = ActionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        discrete_actions=False
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    # Test forward pass
    action_preds, values = model(states, actions, seq_lens)
    assert action_preds.shape == (batch_size, seq_len, 2 * action_dim)
    assert values.shape == (batch_size, seq_len)
    
    # Test loss computation
    action_loss, value_loss = model.compute_loss(
        action_preds,
        actions,
        values,
        torch.randn_like(values)
    )
    assert isinstance(action_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)
    
    # Test action prediction
    pred_actions = model.predict_action(states)
    assert pred_actions.shape == (batch_size, action_dim) 