# Action Transformer

A PyTorch implementation of a Transformer-based model for sequential decision-making tasks in reinforcement learning environments.

## Features

- Transformer-based architecture for processing sequences of states and actions
- Support for both discrete and continuous action spaces
- Flexible embedding layers for states and actions with positional encoding
- Configurable model architecture (layers, heads, dimensions)
- Efficient handling of variable-length sequences
- Integration-ready with common RL frameworks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

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

## Model Architecture

The Action Transformer consists of several key components:

1. **Embedding Layers**
   - State embeddings
   - Action embeddings
   - Learnable positional encodings

2. **Transformer Encoder**
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization and residual connections

3. **Action Prediction**
   - Discrete actions: Softmax output
   - Continuous actions: Direct output

## Testing

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License 