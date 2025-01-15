import torch
import pytest
from src.models import HierarchicalTransformer

def test_hierarchical_transformer():
    """Test HierarchicalTransformer with discrete actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 10
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    num_options = 5
    option_duration = 4
    batch_size = 8
    seq_len = 20
    
    # Create model
    model = HierarchicalTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        num_options=num_options,
        option_duration=option_duration,
        discrete_actions=True
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    options = torch.randint(0, num_options, (batch_size,))
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    # Test forward pass
    outputs = model(states, actions, options, seq_lens)
    assert outputs['action_preds'].shape == (batch_size, seq_len, action_dim)
    assert outputs['option_preds'].shape == (batch_size, seq_len, num_options)
    assert outputs['termination_probs'].shape == (batch_size, seq_len)
    assert outputs['values'].shape == (batch_size, seq_len)
    
    # Test loss computation
    target_options = torch.randint(0, num_options, (batch_size, seq_len))
    target_terminations = torch.randint(0, 2, (batch_size, seq_len)).float()
    target_values = torch.randn(batch_size, seq_len)
    
    losses = model.compute_loss(
        outputs,
        actions,
        target_options,
        target_terminations,
        target_values
    )
    
    assert 'action_loss' in losses
    assert 'option_loss' in losses
    assert 'termination_loss' in losses
    assert 'value_loss' in losses
    assert all(isinstance(loss, torch.Tensor) for loss in losses.values())

def test_continuous_hierarchical_transformer():
    """Test HierarchicalTransformer with continuous actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 6
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    num_options = 5
    option_duration = 4
    batch_size = 8
    seq_len = 20
    
    # Create model
    model = HierarchicalTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        num_options=num_options,
        option_duration=option_duration,
        discrete_actions=False
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    options = torch.randint(0, num_options, (batch_size,))
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    # Test forward pass
    outputs = model(states, actions, options, seq_lens)
    assert outputs['action_preds'].shape == (batch_size, seq_len, action_dim)
    assert outputs['option_preds'].shape == (batch_size, seq_len, num_options)
    assert outputs['termination_probs'].shape == (batch_size, seq_len)
    assert outputs['values'].shape == (batch_size, seq_len)
    
    # Test loss computation
    target_options = torch.randint(0, num_options, (batch_size, seq_len))
    target_terminations = torch.randint(0, 2, (batch_size, seq_len)).float()
    target_values = torch.randn(batch_size, seq_len)
    
    losses = model.compute_loss(
        outputs,
        actions,
        target_options,
        target_terminations,
        target_values
    )
    
    assert 'action_loss' in losses
    assert 'option_loss' in losses
    assert 'termination_loss' in losses
    assert 'value_loss' in losses
    assert all(isinstance(loss, torch.Tensor) for loss in losses.values()) 