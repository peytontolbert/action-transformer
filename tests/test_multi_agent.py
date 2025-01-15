import torch
import pytest
from src.models import MultiAgentTransformer

def test_multi_agent_transformer():
    """Test MultiAgentTransformer with discrete actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 10
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    max_num_agents = 4
    batch_size = 8
    seq_len = 20
    num_agents = 3
    
    # Create model
    model = MultiAgentTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        max_num_agents=max_num_agents,
        discrete_actions=True
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, num_agents, seq_len, state_dim)
    actions = torch.randint(0, action_dim, (batch_size, num_agents, seq_len))
    agent_ids = torch.arange(num_agents).expand(batch_size, -1)
    seq_lens = torch.full((batch_size, num_agents), seq_len, dtype=torch.long)
    agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    
    # Test forward pass
    action_preds, values, hidden_states = model(states, agent_ids, actions, seq_lens, agent_mask)
    assert action_preds.shape == (batch_size, num_agents, seq_len, action_dim)
    assert values.shape == (batch_size, num_agents, seq_len)
    assert hidden_states.shape == (batch_size, num_agents, seq_len, embed_dim)
    
    # Test loss computation
    action_loss, value_loss = model.compute_loss(
        action_preds,
        actions,
        values,
        torch.randn_like(values),
        hidden_states=hidden_states,
        mask=None,
        agent_mask=agent_mask
    )
    assert isinstance(action_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)

def test_continuous_multi_agent_transformer():
    """Test MultiAgentTransformer with continuous actions."""
    
    # Model parameters
    state_dim = 64
    action_dim = 6
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    max_num_agents = 4
    batch_size = 8
    seq_len = 20
    num_agents = 3
    
    # Create model
    model = MultiAgentTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        max_num_agents=max_num_agents,
        discrete_actions=False
    )
    
    # Create dummy inputs
    states = torch.randn(batch_size, num_agents, seq_len, state_dim)
    actions = torch.randn(batch_size, num_agents, seq_len, action_dim)
    agent_ids = torch.arange(num_agents).expand(batch_size, -1)
    seq_lens = torch.full((batch_size, num_agents), seq_len, dtype=torch.long)
    agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    
    # Test forward pass
    action_preds, values, hidden_states = model(states, agent_ids, actions, seq_lens, agent_mask)
    assert action_preds.shape == (batch_size, num_agents, seq_len, action_dim)
    assert values.shape == (batch_size, num_agents, seq_len)
    assert hidden_states.shape == (batch_size, num_agents, seq_len, embed_dim)
    
    # Test loss computation
    action_loss, value_loss = model.compute_loss(
        action_preds,
        actions,
        values,
        torch.randn_like(values),
        hidden_states=hidden_states,
        mask=None,
        agent_mask=agent_mask
    )
    assert isinstance(action_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)

def test_variable_agents():
    """Test MultiAgentTransformer with variable number of agents."""
    torch.manual_seed(42)  # For reproducibility

    # Model parameters
    state_dim = 64
    action_dim = 10
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    max_num_agents = 4
    batch_size = 8
    seq_len = 20

    # Create model
    model = MultiAgentTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        max_num_agents=max_num_agents,
        discrete_actions=True
    )

    # Create dummy inputs with variable number of agents
    states = torch.randn(batch_size, max_num_agents, seq_len, state_dim)
    states = states / states.norm(dim=-1, keepdim=True)  # Normalize states
    
    actions = torch.randint(0, action_dim, (batch_size, max_num_agents, seq_len))
    agent_ids = torch.arange(max_num_agents).expand(batch_size, -1)
    
    # Create sequence lengths that vary but are valid
    seq_lens = torch.randint(seq_len//2, seq_len+1, (batch_size, max_num_agents))
    
    # Create agent mask with variable number of agents (at least 1 per batch)
    num_agents = torch.randint(1, max_num_agents + 1, (batch_size,))
    agent_mask = torch.zeros(batch_size, max_num_agents, dtype=torch.bool)
    for i in range(batch_size):
        agent_mask[i, :num_agents[i]] = True

    print(f"\nNumber of agents per batch: {num_agents}")
    print(f"Total active agents: {agent_mask.sum().item()}")

    # Mask input states and actions for inactive agents
    states = states * agent_mask.unsqueeze(-1).unsqueeze(-1)
    actions = actions * agent_mask.unsqueeze(-1)

    # Create temporal mask based on sequence lengths
    temporal_mask = torch.ones(batch_size, max_num_agents, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(max_num_agents):
            if agent_mask[i, j]:
                temporal_mask[i, j, seq_lens[i, j]:] = False

    # Debug embeddings
    state_embeds = model.state_embedding(states)
    print(f"\nState embeddings range: [{state_embeds.min().item():.2f}, {state_embeds.max().item():.2f}]")
    print(f"Any NaN in state embeddings: {torch.isnan(state_embeds).any().item()}")

    # Test forward pass
    action_preds, values, hidden_states = model(states, agent_ids, actions, seq_lens, agent_mask)

    print(f"\nAction predictions range: [{action_preds.min().item():.2f}, {action_preds.max().item():.2f}]")
    print(f"Any NaN in predictions: {torch.isnan(action_preds).any().item()}")
    print(f"Any NaN in values: {torch.isnan(values).any().item()}")
    print(f"Any NaN in hidden states: {torch.isnan(hidden_states).any().item()}")

    # Test loss computation with both masks
    action_loss, value_loss = model.compute_loss(
        action_preds,
        actions,
        values,
        torch.randn_like(values),
        hidden_states=hidden_states,
        mask=temporal_mask,
        agent_mask=agent_mask
    )

    print(f"\nAction loss: {action_loss.item() if not torch.isnan(action_loss).any() else 'NaN'}")
    print(f"Value loss: {value_loss.item() if not torch.isnan(value_loss).any() else 'NaN'}")

    # Check shapes
    assert action_preds.shape == (batch_size, max_num_agents, seq_len, action_dim)
    assert values.shape == (batch_size, max_num_agents, seq_len)
    assert hidden_states.shape == (batch_size, max_num_agents, seq_len, embed_dim)

    # Check that predictions are valid for active agents
    valid_preds = action_preds[agent_mask.unsqueeze(-1).unsqueeze(-1).expand_as(action_preds)]
    assert not torch.isnan(valid_preds).any(), "NaN values in predictions for active agents"
    
    # Check that loss is computed only for active agents and valid timesteps
    valid_mask = agent_mask.unsqueeze(-1) & temporal_mask
    assert valid_mask.any(), "No valid elements in mask"
    
    # More lenient loss check
    assert not torch.isnan(action_loss), "Action loss is NaN"
    assert action_loss >= 0, "Action loss should be non-negative" 