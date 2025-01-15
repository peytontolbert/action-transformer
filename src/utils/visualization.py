import torch
import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class AttentionVisualizer:
    """Utility class for visualizing transformer attention patterns."""
    
    @staticmethod
    def plot_attention_weights(
        attention_weights: torch.Tensor,
        save_path: Optional[Path] = None,
        title: str = "Attention Weights",
        show: bool = True
    ):
        """Plot attention weight matrix.
        
        Args:
            attention_weights: Attention weights of shape (seq_len, seq_len)
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            xticklabels=list(range(attention_weights.shape[1])),
            yticklabels=list(range(attention_weights.shape[0]))
        )
        plt.title(title)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_multi_head_attention(
        attention_weights: torch.Tensor,
        num_heads: Optional[int] = None,
        save_path: Optional[Path] = None,
        title: str = "Multi-Head Attention",
        show: bool = True
    ):
        """Plot attention weights for multiple attention heads.
        
        Args:
            attention_weights: Attention weights of shape (num_heads, seq_len, seq_len)
            num_heads: Optional number of heads to plot (default: all)
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        num_heads = num_heads or attention_weights.shape[0]
        num_heads = min(num_heads, attention_weights.shape[0])
        
        # Create subplot grid
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False
        )
        fig.suptitle(title)
        
        for i in range(num_heads):
            row = i // n_cols
            col = i % n_cols
            sns.heatmap(
                attention_weights[i],
                cmap='viridis',
                xticklabels=False,
                yticklabels=False,
                ax=axes[row, col]
            )
            axes[row, col].set_title(f'Head {i+1}')
            
        # Remove empty subplots
        for i in range(num_heads, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

class TrainingVisualizer:
    """Utility class for visualizing training progress."""
    
    @staticmethod
    def plot_metrics(
        metrics: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        title: str = "Training Metrics",
        show: bool = True
    ):
        """Plot training metrics over time.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        plt.figure(figsize=(12, 6))
        
        for name, values in metrics.items():
            plt.plot(values, label=name)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_learning_curves(
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        title: str = "Learning Curves",
        show: bool = True
    ):
        """Plot training and validation metrics.
        
        Args:
            train_metrics: Dictionary of training metric values
            val_metrics: Dictionary of validation metric values
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        assert train_metrics.keys() == val_metrics.keys(), \
            "Train and validation metrics must have same keys"
            
        n_metrics = len(train_metrics)
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2,
            figsize=(12, 4 * ((n_metrics + 1) // 2)),
            squeeze=False
        )
        fig.suptitle(title)
        
        for i, (name, train_values) in enumerate(train_metrics.items()):
            row = i // 2
            col = i % 2
            
            val_values = val_metrics[name]
            axes[row, col].plot(train_values, label=f'Train {name}')
            axes[row, col].plot(val_values, label=f'Val {name}')
            axes[row, col].set_title(name)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Value')
            axes[row, col].legend()
            axes[row, col].grid(True)
            
        # Remove empty subplots
        if n_metrics % 2 == 1:
            fig.delaxes(axes[-1, -1])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

class ActionVisualizer:
    """Utility class for visualizing model predictions and actions."""
    
    @staticmethod
    def plot_action_distribution(
        action_probs: torch.Tensor,
        action_space: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        title: str = "Action Distribution",
        show: bool = True
    ):
        """Plot discrete action probability distribution.
        
        Args:
            action_probs: Action probabilities of shape (num_actions,)
            action_space: Optional list of action names
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        if isinstance(action_probs, torch.Tensor):
            action_probs = action_probs.detach().cpu().numpy()
            
        plt.figure(figsize=(10, 6))
        x = np.arange(len(action_probs))
        plt.bar(x, action_probs)
        
        if action_space:
            plt.xticks(x, action_space, rotation=45)
        
        plt.title(title)
        plt.xlabel('Action')
        plt.ylabel('Probability')
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_action_trajectory(
        states: torch.Tensor,
        actions: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
        title: str = "Action Trajectory",
        show: bool = True
    ):
        """Plot continuous action trajectories.
        
        Args:
            states: State sequence of shape (seq_len, state_dim)
            actions: Action sequence of shape (seq_len, action_dim)
            predictions: Optional predicted actions
            save_path: Optional path to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        seq_len, action_dim = actions.shape
        
        fig, axes = plt.subplots(
            action_dim, 1,
            figsize=(12, 4 * action_dim),
            sharex=True
        )
        if action_dim == 1:
            axes = [axes]
            
        for i in range(action_dim):
            axes[i].plot(actions[:, i], label='True Action')
            if predictions is not None:
                axes[i].plot(predictions[:, i], '--', label='Predicted Action')
            axes[i].set_title(f'Action Dimension {i+1}')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)
            
        axes[-1].set_xlabel('Time Step')
        plt.suptitle(title)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close() 