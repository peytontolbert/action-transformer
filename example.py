import torch
from src.models.action_transformer import ActionTransformer
from src.models.multi_agent import MultiAgentTransformer

def single_agent_example():
    """Demonstrate usage of single-agent transformer."""
    print("\n=== Single-Agent Transformer Example ===")
    
    # Model parameters
    state_dim = 64
    action_dim = 10
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    hidden_dim = 256
    max_seq_len = 50
    batch_size = 8
    seq_len = 20
    
    # Create model for discrete actions
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
    
    # Forward pass
    action_preds, values = model(states, actions, seq_lens)
    
    print(f"Input states shape: {states.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Predicted actions shape: {action_preds.shape}")
    print(f"Predicted values shape: {values.shape}")

def multi_agent_example():
    """Demonstrate usage of multi-agent transformer."""
    print("\n=== Multi-Agent Transformer Example ===")
    
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
    actions = torch.randint(0, action_dim, (batch_size, max_num_agents, seq_len))
    agent_ids = torch.arange(max_num_agents).expand(batch_size, -1)
    seq_lens = torch.full((batch_size, max_num_agents), seq_len, dtype=torch.long)
    
    # Create agent mask with variable number of agents
    num_agents = torch.randint(1, max_num_agents + 1, (batch_size,))
    agent_mask = torch.zeros(batch_size, max_num_agents, dtype=torch.bool)
    for i in range(batch_size):
        agent_mask[i, :num_agents[i]] = True
    
    # Forward pass
    action_preds, values, hidden_states = model(
        states=states,
        agent_ids=agent_ids,
        actions=actions,
        seq_lens=seq_lens,
        agent_mask=agent_mask
    )
    
    print(f"Input states shape: {states.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Agent IDs shape: {agent_ids.shape}")
    print(f"Number of agents per batch: {num_agents}")
    print(f"Predicted actions shape: {action_preds.shape}")
    print(f"Predicted values shape: {values.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")

def main():
    """Run examples."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    single_agent_example()
    multi_agent_example()

if __name__ == "__main__":
    main() 