# How to Train Agentic Models with ChatGPT Data

This guide explains how to use the Action Transformer to train models that can learn from ChatGPT agent interactions and generalize to new tasks.

## Overview

The Action Transformer architecture is particularly well-suited for learning from language model agent data because:
1. It can process sequential interactions between agents and environments
2. It handles variable-length sequences and multiple agents
3. It can learn both high-level strategies and low-level actions
4. It supports both discrete (text-based) and continuous (embedding-based) actions

## Data Preparation

### 1. Collecting Agent Interactions

```python
# Example structure for agent interaction data
interaction_data = {
    "states": [
        "You are helping a user debug their Python code...",
        "The user has provided a stack trace showing...",
        # ... more states
    ],
    "actions": [
        "Let me analyze the error message...",
        "I'll check the function definition...",
        # ... more actions
    ],
    "rewards": [1, 0, 1],  # Optional rewards for reinforcement learning
    "agent_ids": [0, 0, 0],  # For multi-agent scenarios
    "metadata": {
        "task_type": "debugging",
        "success": True,
        # ... additional metadata
    }
}
```

### 2. Converting to Model Format

```python
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("gpt2")
embedding_model = AutoModel.from_pretrained("gpt2")

def prepare_data(interaction_data):
    # Convert states and actions to embeddings
    state_embeddings = embed_text(interaction_data["states"])
    action_embeddings = embed_text(interaction_data["actions"])
    
    # Create attention masks
    state_mask = create_padding_mask(state_embeddings)
    action_mask = create_padding_mask(action_embeddings)
    
    return {
        "states": state_embeddings,
        "actions": action_embeddings,
        "state_mask": state_mask,
        "action_mask": action_mask,
        "agent_ids": torch.tensor(interaction_data["agent_ids"])
    }
```

## Training Approaches

### 1. Behavioral Cloning

Train the model to directly imitate ChatGPT's actions:

```python
from src.models.action_transformer import ActionTransformer

# Create model for text embeddings
model = ActionTransformer(
    state_dim=768,          # GPT2 embedding dimension
    action_dim=768,         # For continuous action space
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    max_seq_len=1024,
    discrete_actions=False  # Use continuous embeddings
)

# Training loop
for batch in dataloader:
    # Forward pass
    action_preds, values = model(
        states=batch["states"],
        actions=batch["actions"],
        seq_lens=batch["seq_lens"]
    )
    
    # Compute loss
    loss = compute_embedding_loss(action_preds, batch["target_actions"])
    
    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2. Multi-Agent Learning

Train multiple specialized agents that can interact:

```python
from src.models.multi_agent import MultiAgentTransformer

# Create multi-agent model
model = MultiAgentTransformer(
    state_dim=768,
    action_dim=768,
    embed_dim=256,
    num_layers=4,
    num_heads=4,
    hidden_dim=512,
    max_seq_len=1024,
    max_num_agents=4,
    discrete_actions=False
)

# Training with multiple specialized agents
for batch in dataloader:
    # Forward pass with agent masks
    action_preds, values, hidden_states = model(
        states=batch["states"],
        agent_ids=batch["agent_ids"],
        actions=batch["actions"],
        seq_lens=batch["seq_lens"],
        agent_mask=batch["agent_mask"]
    )
```

### 3. Hierarchical Learning

Use the hierarchical transformer to learn high-level strategies:

```python
# Example of option-based training
options = ["debug_code", "explain_concept", "write_test"]
num_options = len(options)

model = HierarchicalTransformer(
    state_dim=768,
    action_dim=768,
    num_options=num_options,
    embed_dim=256,
    num_layers=4,
    num_heads=4
)

# Training loop with options
for batch in dataloader:
    # Forward pass
    option_preds, action_preds, term_preds = model(
        states=batch["states"],
        actions=batch["actions"],
        options=batch["options"],
        seq_lens=batch["seq_lens"]
    )
```

## Best Practices

1. **Data Quality**
   - Filter for high-quality interactions
   - Include diverse task types
   - Balance successful and unsuccessful interactions
   - Maintain context consistency

2. **Model Configuration**
   - Start with smaller models and scale up
   - Use appropriate embedding dimensions
   - Adjust sequence length based on interaction patterns
   - Tune attention heads for task complexity

3. **Training Tips**
   - Use gradient clipping for stability
   - Implement early stopping
   - Monitor attention patterns
   - Validate on held-out tasks

4. **Evaluation**
   - Test on novel scenarios
   - Measure coherence and consistency
   - Compare with baseline agents
   - Assess generalization ability

## Example Applications

1. **Code Assistant**
```python
# Train on code-related interactions
code_model = ActionTransformer(
    state_dim=768,
    action_dim=768,
    discrete_actions=False
)
```

2. **Multi-Expert System**
```python
# Train specialized agents for different domains
expert_model = MultiAgentTransformer(
    state_dim=768,
    action_dim=768,
    max_num_agents=4  # Different expert types
)
```

3. **Task Planner**
```python
# Train for high-level planning
planner_model = HierarchicalTransformer(
    state_dim=768,
    action_dim=768,
    num_options=10  # Different task strategies
)
```

## Integration with Existing Systems

1. **API Integration**
```python
from openai import OpenAI

def get_agent_response(state, model):
    # Get model prediction
    action_embedding = model(state_embedding)
    
    # Convert to text using GPT
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Convert this embedding to a response."},
            {"role": "user", "content": str(action_embedding)}
        ]
    )
    return response.choices[0].message.content
```

2. **Pipeline Integration**
```python
class AgentPipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.embedding_model = AutoModel.from_pretrained("gpt2")
        self.action_model = ActionTransformer(...)
    
    def process(self, user_input):
        # Convert input to state embedding
        state_embedding = self.embed_text(user_input)
        
        # Get action prediction
        action_embedding = self.action_model(state_embedding)
        
        # Convert to response
        response = self.generate_response(action_embedding)
        return response
```

## Troubleshooting

1. **Common Issues**
   - NaN values in training: Check embedding normalization
   - Poor generalization: Increase data diversity
   - Inconsistent responses: Adjust temperature in generation
   - Memory issues: Batch size and sequence length tuning

2. **Performance Optimization**
   - Use gradient checkpointing for large models
   - Implement attention caching
   - Optimize embedding computation
   - Use mixed precision training

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Attention Mechanism Paper](https://arxiv.org/abs/1706.03762) 