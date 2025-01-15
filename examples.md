# Action Transformer Examples

This document demonstrates how to use the Action Transformer models in this repository.

## Basic Examples

### 1. Single-Agent Transformer

```python
import torch
from src.models.action_transformer import ActionTransformer
from src.utils.masking import create_padding_mask

# Initialize model
model = ActionTransformer(
    state_dim=64,        # State feature dimension
    action_dim=10,       # Number of possible actions
    embed_dim=128,       # Embedding dimension
    num_layers=4,        # Number of transformer layers
    num_heads=4,         # Number of attention heads
    hidden_dim=256,      # Hidden layer dimension
    max_seq_len=100,     # Maximum sequence length
    discrete_actions=True # Using discrete action space
)

# Create sample batch
batch_size = 32
seq_len = 50

# Input states (batch_size, seq_len, state_dim)
states = torch.randn(batch_size, seq_len, 64)

# Previous actions (batch_size, seq_len)
actions = torch.randint(0, 10, (batch_size, seq_len))

# Sequence lengths for each batch
seq_lens = torch.randint(10, seq_len + 1, (batch_size,))

# Forward pass
action_preds, values = model(
    states=states,
    actions=actions,
    seq_lens=seq_lens
)

print(f"Action predictions shape: {action_preds.shape}")  # [batch_size, seq_len, action_dim]
print(f"Value predictions shape: {values.shape}")        # [batch_size, seq_len]
```

### 2. Multi-Agent Transformer

```python
from src.models.multi_agent import MultiAgentTransformer

# Initialize model
model = MultiAgentTransformer(
    state_dim=64,           # State feature dimension
    action_dim=10,          # Number of possible actions
    embed_dim=128,          # Embedding dimension
    num_layers=4,           # Number of transformer layers
    num_heads=4,            # Number of attention heads
    hidden_dim=256,         # Hidden layer dimension
    max_seq_len=50,         # Maximum sequence length
    max_num_agents=4,       # Maximum number of agents
    discrete_actions=True   # Using discrete action space
)

# Create sample batch
batch_size = 16
num_agents = 3
seq_len = 30

# Input states (batch_size, num_agents, seq_len, state_dim)
states = torch.randn(batch_size, num_agents, seq_len, 64)

# Previous actions (batch_size, num_agents, seq_len)
actions = torch.randint(0, 10, (batch_size, num_agents, seq_len))

# Agent IDs
agent_ids = torch.arange(num_agents).expand(batch_size, -1)

# Sequence lengths for each agent
seq_lens = torch.randint(10, seq_len + 1, (batch_size, num_agents))

# Agent mask (which agents are active)
agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
agent_mask[:, -1] = 0  # Last agent inactive for demonstration

# Forward pass
action_preds, values, hidden_states = model(
    states=states,
    agent_ids=agent_ids,
    actions=actions,
    seq_lens=seq_lens,
    agent_mask=agent_mask
)

print(f"Action predictions shape: {action_preds.shape}")  # [batch_size, num_agents, seq_len, action_dim]
print(f"Value predictions shape: {values.shape}")         # [batch_size, num_agents, seq_len]
print(f"Hidden states shape: {hidden_states.shape}")      # [batch_size, num_agents, seq_len, embed_dim]
```

### 3. Hierarchical Transformer

```python
from src.models.hierarchical import HierarchicalTransformer

# Initialize model
model = HierarchicalTransformer(
    state_dim=64,           # State feature dimension
    action_dim=10,          # Number of possible actions
    num_options=5,          # Number of high-level options
    embed_dim=128,          # Embedding dimension
    num_layers=4,           # Number of transformer layers
    num_heads=4,            # Number of attention heads
    hidden_dim=256,         # Hidden layer dimension
    max_seq_len=50,         # Maximum sequence length
    discrete_actions=True   # Using discrete action space
)

# Create sample batch
batch_size = 16
seq_len = 30

# Input states (batch_size, seq_len, state_dim)
states = torch.randn(batch_size, seq_len, 64)

# Previous actions (batch_size, seq_len)
actions = torch.randint(0, 10, (batch_size, seq_len))

# Current option for each sequence
options = torch.randint(0, 5, (batch_size,))

# Sequence lengths
seq_lens = torch.randint(10, seq_len + 1, (batch_size,))

# Forward pass
option_preds, action_preds, term_preds = model(
    states=states,
    actions=actions,
    options=options,
    seq_lens=seq_lens
)

print(f"Option predictions shape: {option_preds.shape}")    # [batch_size, seq_len, num_options]
print(f"Action predictions shape: {action_preds.shape}")    # [batch_size, seq_len, action_dim]
print(f"Termination predictions shape: {term_preds.shape}") # [batch_size, seq_len]
```

## Training Examples

### 1. Training Loop with Loss Computation

```python
import torch.optim as optim
from src.losses.action_losses import ActionPredictionLoss

# Initialize model and optimizer
model = ActionTransformer(
    state_dim=64,
    action_dim=10,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=100,
    discrete_actions=True
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = ActionPredictionLoss(discrete_actions=True)

# Training loop
model.train()
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass
    action_preds, values = model(
        states=batch['states'],
        actions=batch['actions'],
        seq_lens=batch['seq_lens']
    )
    
    # Compute losses
    action_loss = loss_fn.discrete_action_loss(
        action_preds,
        batch['target_actions'],
        batch['mask']
    )
    value_loss = loss_fn.value_loss(
        values,
        batch['target_values'],
        batch['mask']
    )
    
    # Combined loss
    loss = action_loss + 0.5 * value_loss
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### 2. Multi-Agent Training

```python
# Initialize model and optimizer
model = MultiAgentTransformer(
    state_dim=64,
    action_dim=10,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=50,
    max_num_agents=4,
    discrete_actions=True
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = ActionPredictionLoss(discrete_actions=True)

# Training loop
model.train()
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass
    action_preds, values, hidden_states = model(
        states=batch['states'],
        agent_ids=batch['agent_ids'],
        actions=batch['actions'],
        seq_lens=batch['seq_lens'],
        agent_mask=batch['agent_mask']
    )
    
    # Compute losses (only for active agents)
    mask = batch['mask'] & batch['agent_mask'].unsqueeze(-1)
    
    action_loss = loss_fn.discrete_action_loss(
        action_preds,
        batch['target_actions'],
        mask
    )
    value_loss = loss_fn.value_loss(
        values,
        batch['target_values'],
        mask
    )
    
    # Combined loss
    loss = action_loss + 0.5 * value_loss
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### 3. Hierarchical Training

```python
# Initialize model and optimizer
model = HierarchicalTransformer(
    state_dim=64,
    action_dim=10,
    num_options=5,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=50,
    discrete_actions=True
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = ActionPredictionLoss(discrete_actions=True)

# Training loop
model.train()
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass
    option_preds, action_preds, term_preds = model(
        states=batch['states'],
        actions=batch['actions'],
        options=batch['options'],
        seq_lens=batch['seq_lens']
    )
    
    # Compute losses
    action_loss = loss_fn.discrete_action_loss(
        action_preds,
        batch['target_actions'],
        batch['mask']
    )
    option_loss = loss_fn.discrete_action_loss(
        option_preds,
        batch['target_options'],
        batch['mask']
    )
    term_loss = F.binary_cross_entropy_with_logits(
        term_preds,
        batch['termination'],
        batch['mask'].float()
    )
    
    # Combined loss
    loss = action_loss + option_loss + 0.1 * term_loss
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

## Real-World Applications

The Action Transformer architecture is particularly well-suited for sequential decision-making tasks where context and history matter. Here are key applications:

### 1. Code Generation and Assistance

The Action Transformer can be trained to assist with coding tasks by:
- Learning from developer actions and code changes
- Understanding programming patterns and context
- Generating code completions and suggestions

Example use cases:
```python
# Code Completion Model
code_assistant = ActionTransformer(
    state_dim=512,      # Code context embedding size
    action_dim=32000,   # Vocabulary size for code tokens
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    max_seq_len=1024,   # Support longer code sequences
    discrete_actions=True
)

# State could represent:
# - Current file content
# - Open files/workspace context
# - Recent edits
# - Cursor position
# - File type and language

# Actions could be:
# - Next token prediction
# - Code block completion
# - Function/class suggestions
# - Bug fixes
# - Refactoring operations
```

### 2. Task Planning and Automation

The Hierarchical Transformer excels at breaking down complex tasks:
```python
# Task Planning Model
task_planner = HierarchicalTransformer(
    state_dim=128,          # Task context features
    action_dim=50,          # Primitive actions
    num_options=10,         # High-level tasks like:
                           # - Data preprocessing
                           # - Model training
                           # - Evaluation
                           # - Deployment
    embed_dim=256,
    num_layers=4,
    num_heads=4,
    hidden_dim=512,
    max_seq_len=100,
    discrete_actions=True
)

# Can be used for:
# - ML pipeline automation
# - CI/CD workflows
# - Data processing pipelines
# - Project management
```

### 3. Multi-Agent Systems

The Multi-Agent Transformer enables coordinated behavior:
```python
# Collaborative System
team_model = MultiAgentTransformer(
    state_dim=128,
    action_dim=20,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    hidden_dim=512,
    max_seq_len=50,
    max_num_agents=5,    # Different roles like:
                        # - Code reviewer
                        # - Implementation specialist
                        # - Testing expert
                        # - Documentation writer
                        # - Project manager
    discrete_actions=True
)

# Applications:
# - Code review automation
# - Team coordination
# - Distributed systems
# - Multi-robot control
```

### 4. Interactive Assistants

Action Transformer can power interactive AI assistants:
```python
# Interactive Assistant
assistant = ActionTransformer(
    state_dim=768,      # Large context embedding
    action_dim=1000,    # Response actions
    embed_dim=512,
    num_layers=8,
    num_heads=8,
    hidden_dim=1024,
    max_seq_len=2048,   # Long conversation history
    discrete_actions=True
)

# Features:
# - Context-aware responses
# - Memory of conversation history
# - Task tracking
# - Personalization
# - Tool use and API integration
```

### 5. Reinforcement Learning Applications

The architecture is particularly powerful for RL:
```python
# RL Agent
rl_agent = ActionTransformer(
    state_dim=128,      # Environment state
    action_dim=30,      # Available actions
    embed_dim=256,
    num_layers=4,
    num_heads=4,
    hidden_dim=512,
    max_seq_len=50,     # Recent history window
    discrete_actions=True
)

# Capabilities:
# - Learning from demonstrations
# - Policy optimization
# - Value estimation
# - Long-term planning
# - Multi-task learning
```

## Key Benefits

1. **Context Awareness**: The transformer's attention mechanism allows it to:
   - Consider full history of states and actions
   - Identify relevant patterns and dependencies
   - Make decisions based on long-term context

2. **Hierarchical Planning**: Using the hierarchical variant enables:
   - Breaking down complex tasks
   - Learning reusable sub-tasks
   - Managing long-horizon planning

3. **Multi-Agent Coordination**: The multi-agent architecture supports:
   - Team-based decision making
   - Role specialization
   - Inter-agent communication
   - Collaborative problem solving

4. **Flexibility**: The architecture can handle:
   - Variable length sequences
   - Different types of inputs/outputs
   - Multiple objectives
   - Both discrete and continuous actions

5. **Scalability**: The model scales well to:
   - Large action spaces
   - Complex state representations
   - Long sequences
   - Multiple agents

## Training Data Sources

The Action Transformer can learn from various data sources:

1. **Expert Demonstrations**
   - Human developers' coding sessions
   - Task completion recordings
   - Professional workflows

2. **Interaction Logs**
   - User-assistant conversations
   - System interaction traces
   - Tool usage patterns

3. **Automated Systems**
   - CI/CD pipelines
   - Automated tests
   - Deployment logs

4. **Multi-Agent Interactions**
   - Team collaborations
   - Code reviews
   - Project management data

These examples demonstrate how Action Transformer can be applied to real-world problems requiring sequential decision-making, planning, and coordination. 