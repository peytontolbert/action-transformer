# Action Transformer Examples

This document provides detailed examples of using the Action Transformer for various scenarios.

## Basic Usage Examples

### 1. Single-Agent Code Assistant

Train a model to assist with coding tasks:

```python
import torch
from src.models.action_transformer import ActionTransformer
from src.utils.masking import create_padding_mask

# Initialize model
code_assistant = ActionTransformer(
    state_dim=64,        # Code context features
    action_dim=10,       # Number of possible actions
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=100,
    discrete_actions=True
)

# Example: Code completion task
states = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        ",
    "class BinaryTree:\n    def __init__(self):\n        self.value = None\n        ",
]

# Convert to tensors and create batch
batch_size = len(states)
state_tensor = torch.randn(batch_size, 50, 64)  # Encoded states
action_tensor = torch.randint(0, 10, (batch_size, 50))  # Previous actions
seq_lengths = torch.tensor([45, 40])  # Actual sequence lengths

# Forward pass
action_preds, values = code_assistant(
    states=state_tensor,
    actions=action_tensor,
    seq_lens=seq_lengths
)

# Get next action
next_action = torch.argmax(action_preds[:, -1], dim=-1)
```

### 2. Multi-Agent Collaboration

Train multiple agents to collaborate on a task:

```python
from src.models.multi_agent import MultiAgentTransformer

# Initialize model for collaborative coding
team_model = MultiAgentTransformer(
    state_dim=64,
    action_dim=10,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=50,
    max_num_agents=3,  # Reviewer, Implementer, Tester
    discrete_actions=True
)

# Example: Code review scenario
batch_size = 2
num_agents = 3
seq_len = 30

# Create dummy inputs
states = torch.randn(batch_size, num_agents, seq_len, 64)  # Code state for each agent
actions = torch.randint(0, 10, (batch_size, num_agents, seq_len))  # Previous actions
agent_ids = torch.arange(num_agents).expand(batch_size, -1)  # Agent identities

# Create agent masks (all agents active)
agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)

# Forward pass
action_preds, values, hidden_states = team_model(
    states=states,
    agent_ids=agent_ids,
    actions=actions,
    seq_lens=torch.full((batch_size, num_agents), seq_len),
    agent_mask=agent_mask
)
```

## Advanced Examples

### 1. Hierarchical Task Planning

Use hierarchical transformer for complex tasks:

```python
from src.models.hierarchical import HierarchicalTransformer

# Define high-level options
options = [
    "design_architecture",
    "implement_feature",
    "write_tests",
    "review_code",
    "refactor"
]

# Initialize hierarchical model
planner = HierarchicalTransformer(
    state_dim=64,
    action_dim=10,
    num_options=len(options),
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    max_seq_len=100,
    discrete_actions=True
)

# Example: Complex development task
batch_size = 1
seq_len = 50

# Create inputs
states = torch.randn(batch_size, seq_len, 64)
actions = torch.randint(0, 10, (batch_size, seq_len))
current_option = torch.randint(0, len(options), (batch_size,))

# Forward pass
option_preds, action_preds, term_preds = planner(
    states=states,
    actions=actions,
    options=current_option,
    seq_lens=torch.tensor([seq_len])
)

# Interpret results
next_option = torch.argmax(option_preds[:, -1], dim=-1)
should_terminate = term_preds[:, -1] > 0.5
next_action = torch.argmax(action_preds[:, -1], dim=-1)
```

### 2. Real-world Training Pipeline

Complete training pipeline with data loading and evaluation:

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import AgentInteractionDataset
from src.models.action_transformer import ActionTransformer
from src.losses.action_losses import ActionPredictionLoss

class TrainingPipeline:
    def __init__(self, config):
        self.model = ActionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len,
            discrete_actions=config.discrete_actions
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = ActionPredictionLoss(discrete_actions=config.discrete_actions)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            action_preds, values = self.model(
                states=batch['states'],
                actions=batch['actions'],
                seq_lens=batch['seq_lens']
            )
            
            # Compute losses
            action_loss = self.loss_fn.discrete_action_loss(
                action_preds,
                batch['target_actions'],
                batch['mask']
            )
            value_loss = self.loss_fn.value_loss(
                values,
                batch['target_values'],
                batch['mask']
            )
            
            # Combined loss
            loss = action_loss + 0.5 * value_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

# Usage example
config = {
    'state_dim': 64,
    'action_dim': 10,
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'hidden_dim': 256,
    'max_seq_len': 100,
    'discrete_actions': True,
    'lr': 1e-4
}

# Create dataset and dataloader
dataset = AgentInteractionDataset('path/to/data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train
pipeline = TrainingPipeline(config)
num_epochs = 10

for epoch in range(num_epochs):
    loss = pipeline.train_epoch(dataloader)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

### 3. Evaluation and Inference

Example of model evaluation and inference:

```python
class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def evaluate_sequence(self, state_sequence):
        """Evaluate model on a sequence of states."""
        with torch.no_grad():
            batch_size = 1
            seq_len = len(state_sequence)
            
            # Convert states to tensor
            states = torch.stack(state_sequence).unsqueeze(0)  # [1, seq_len, state_dim]
            actions = torch.zeros(batch_size, seq_len)  # Dummy actions
            seq_lens = torch.tensor([seq_len])
            
            # Get predictions
            action_preds, values = self.model(states, actions, seq_lens)
            
            return {
                'action_probs': torch.softmax(action_preds[0], dim=-1),
                'value_estimates': values[0]
            }
    
    def get_action_distribution(self, state, temperature=1.0):
        """Get action distribution for a single state."""
        with torch.no_grad():
            state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            action_preds, _ = self.model(state, None, torch.tensor([1]))
            
            # Apply temperature
            logits = action_preds[0, 0] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            return probs

# Usage example
evaluator = ModelEvaluator(trained_model)

# Evaluate a sequence
state_sequence = [torch.randn(64) for _ in range(10)]
results = evaluator.evaluate_sequence(state_sequence)

# Get action distribution for a state
state = torch.randn(64)
action_probs = evaluator.get_action_distribution(state, temperature=0.8)
```

## Integration Examples

### 1. Web API Integration

Example of serving model predictions through a web API:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class StateInput(BaseModel):
    state: list[float]
    temperature: float = 1.0

class ActionOutput(BaseModel):
    action_probs: list[float]
    value: float

@app.post("/predict")
async def predict_action(input_data: StateInput):
    # Convert input to tensor
    state = torch.tensor(input_data.state).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        action_preds, value = model(
            states=state.unsqueeze(0),
            actions=None,
            seq_lens=torch.tensor([1])
        )
        
        # Apply temperature and get probabilities
        logits = action_preds[0, 0] / input_data.temperature
        probs = torch.softmax(logits, dim=-1)
        
        return ActionOutput(
            action_probs=probs.tolist(),
            value=value[0, 0].item()
        )
```

### 2. Multi-Agent System Integration

Example of integrating with a multi-agent system:

```python
class AgentSystem:
    def __init__(self, num_agents):
        self.model = MultiAgentTransformer(
            state_dim=64,
            action_dim=10,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            hidden_dim=256,
            max_seq_len=50,
            max_num_agents=num_agents,
            discrete_actions=True
        )
        self.state_buffer = []
        self.action_buffer = []
        
    def add_observation(self, agent_states, agent_actions):
        """Add new observation to the system."""
        self.state_buffer.append(agent_states)
        self.action_buffer.append(agent_actions)
        
        # Keep only recent history
        max_history = 50
        if len(self.state_buffer) > max_history:
            self.state_buffer = self.state_buffer[-max_history:]
            self.action_buffer = self.action_buffer[-max_history:]
    
    def get_actions(self, agent_mask=None):
        """Get next actions for all agents."""
        with torch.no_grad():
            # Prepare input tensors
            states = torch.stack(self.state_buffer).unsqueeze(0)
            actions = torch.stack(self.action_buffer).unsqueeze(0)
            seq_len = len(self.state_buffer)
            
            # Create agent IDs and mask
            num_agents = states.size(2)
            agent_ids = torch.arange(num_agents).unsqueeze(0)
            if agent_mask is None:
                agent_mask = torch.ones(1, num_agents, dtype=torch.bool)
            
            # Get predictions
            action_preds, _, _ = self.model(
                states=states,
                agent_ids=agent_ids,
                actions=actions,
                seq_lens=torch.tensor([[seq_len]]),
                agent_mask=agent_mask
            )
            
            # Return last prediction for each agent
            return torch.argmax(action_preds[0, :, -1], dim=-1)

# Usage example
system = AgentSystem(num_agents=3)

# Simulate environment steps
for step in range(100):
    # Get environment state
    agent_states = torch.randn(3, 64)  # State for each agent
    agent_actions = torch.randint(0, 10, (3,))  # Previous actions
    
    # Update system
    system.add_observation(agent_states, agent_actions)
    
    # Get next actions
    next_actions = system.get_actions()
```

These examples demonstrate the flexibility and capabilities of the Action Transformer architecture in various scenarios. They can be adapted and extended based on specific requirements. 