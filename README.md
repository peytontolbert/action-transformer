# Action Transformer

A PyTorch implementation of a Transformer-based model for sequential decision-making tasks in reinforcement learning environments, supporting both single-agent and multi-agent scenarios.

## Features

- Transformer-based architecture for processing sequences of states and actions
- Support for both discrete and continuous action spaces
- Multi-agent support with agent-to-agent attention mechanisms
- Hierarchical transformer architecture for complex decision making
- Flexible embedding layers for states, actions, and agent identities
- Configurable model architecture (layers, heads, dimensions)
- Efficient handling of variable-length sequences and variable number of agents
- Integration-ready with common RL frameworks

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see both single-agent and multi-agent transformers in action:

```bash
python example.py
```

The example demonstrates:
- Model creation and configuration
- Input preparation and shape handling
- Forward pass with both transformer variants
- Variable number of agents and sequence lengths
- Proper masking and attention mechanisms

## Usage

### Single-Agent Transformer

```python
from src.models.action_transformer import ActionTransformer

# Create model for discrete actions
model = ActionTransformer(
    state_dim=64,        # Dimension of state features
    action_dim=10,       # Number of discrete actions
    embed_dim=128,       # Embedding dimension
    num_layers=6,        # Number of transformer layers
    num_heads=8,         # Number of attention heads
    hidden_dim=256,      # Hidden dimension in feed-forward network
    max_seq_len=100,     # Maximum sequence length
    discrete_actions=True # Use discrete action space
)

# For continuous actions
continuous_model = ActionTransformer(
    state_dim=64,
    action_dim=6,        # Dimension of continuous action space
    embed_dim=128,
    num_layers=6,
    num_heads=8,
    hidden_dim=256,
    max_seq_len=100,
    discrete_actions=False
)
```

### Multi-Agent Transformer

```python
from src.models.multi_agent import MultiAgentTransformer

# Create model for multi-agent scenario
model = MultiAgentTransformer(
    state_dim=64,           # Dimension of state features
    action_dim=10,          # Action dimension (discrete or continuous)
    embed_dim=128,          # Embedding dimension
    num_layers=4,           # Number of transformer layers
    num_heads=4,            # Number of attention heads
    hidden_dim=256,         # Hidden dimension
    max_seq_len=50,         # Maximum sequence length
    max_num_agents=4,       # Maximum number of agents
    discrete_actions=True   # Whether actions are discrete
)

# Forward pass with variable number of agents
action_preds, values, hidden_states = model(
    states,                 # (batch_size, num_agents, seq_len, state_dim)
    agent_ids,             # (batch_size, num_agents)
    actions,               # (batch_size, num_agents, seq_len)
    seq_lens,              # (batch_size, num_agents)
    agent_mask            # (batch_size, num_agents)
)
```

## Model Architecture

The project includes several transformer variants:

1. **Action Transformer (Single-Agent)**
   - State and action embeddings
   - Positional encodings
   - Standard transformer encoder
   - Action prediction heads

2. **Multi-Agent Transformer**
   - State, action, and agent embeddings
   - Two-level transformer architecture:
     - Temporal transformer for each agent's sequence
     - Agent transformer for inter-agent interactions
   - Support for variable number of agents
   - Masked attention for inactive agents

3. **Hierarchical Transformer**
   - Support for hierarchical decision making
   - Option prediction
   - Termination prediction
   - Low-level action prediction

## Key Components

- **Embedding Layers**
  - State embeddings
  - Action embeddings
  - Agent embeddings
  - Positional encodings

- **Transformer Modules**
  - Multi-head self-attention with masking
  - Feed-forward networks
  - Layer normalization
  - Residual connections

- **Output Heads**
  - Discrete actions: Softmax output
  - Continuous actions: Mean and log standard deviation
  - Value prediction

## Testing

```bash
pytest tests/
```

The test suite includes:
- Single-agent transformer tests
- Multi-agent transformer tests with variable agents
- Continuous and discrete action spaces
- Masking and attention mechanism tests

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
