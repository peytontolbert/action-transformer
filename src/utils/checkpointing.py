import os
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import time

class ModelCheckpoint:
    """Utility class for model checkpointing."""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        max_saves: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor for saving
            mode: One of {'min', 'max'} for monitoring
            save_best_only: Only save when monitored metric improves
            save_freq: Save frequency in epochs
            max_saves: Maximum number of checkpoints to keep
            verbose: Whether to print saving information
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        assert mode in ['min', 'max'], f"mode must be 'min' or 'max', got {mode}"
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_saves = max_saves
        self.verbose = verbose
        
        # Initialize tracking
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []
        
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == 'min':
            return current < best
        return current > best
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            metrics: Optional dictionary of metric values
            metadata: Optional additional metadata
            
        Returns:
            Whether checkpoint was saved
        """
        # Check if we should save
        if epoch % self.save_freq != 0:
            return False
            
        metrics = metrics or {}
        current_value = metrics.get(self.monitor)
        
        # Check if this is the best model
        is_best = False
        if current_value is not None:
            is_best = self._is_better(current_value, self.best_value)
            
        if self.save_best_only and not is_best:
            return False
            
        # Update best value
        if is_best:
            self.best_value = current_value
            
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save checkpoint
        save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        
        if self.verbose:
            print(f"Saved checkpoint to {save_path}")
            if is_best:
                print(f"New best model! {self.monitor}: {current_value:.4f}")
                
        # Save checkpoint info
        self.saved_checkpoints.append({
            'path': str(save_path),
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best
        })
        
        # Save checkpoint info to JSON
        info_path = self.save_dir / 'checkpoint_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.saved_checkpoints, f, indent=2)
            
        # Remove old checkpoints if needed
        if self.max_saves and len(self.saved_checkpoints) > self.max_saves:
            oldest = self.saved_checkpoints.pop(0)
            os.remove(oldest['path'])
            
        return True
    
    @staticmethod
    def load_checkpoint(
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Optional device to load model to
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        if device is not None:
            model = model.to(device)
            
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'metadata': checkpoint.get('metadata', {}),
            'timestamp': checkpoint.get('timestamp')
        }
    
    @staticmethod
    def get_latest_checkpoint(save_dir: Union[str, Path]) -> Optional[Path]:
        """Get path to latest checkpoint in directory.
        
        Args:
            save_dir: Directory containing checkpoints
            
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        save_dir = Path(save_dir)
        if not save_dir.exists():
            return None
            
        checkpoints = list(save_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None
            
        return max(checkpoints, key=os.path.getctime)
    
    @staticmethod
    def get_best_checkpoint(save_dir: Union[str, Path]) -> Optional[Path]:
        """Get path to best checkpoint based on checkpoint info.
        
        Args:
            save_dir: Directory containing checkpoints
            
        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        save_dir = Path(save_dir)
        info_path = save_dir / 'checkpoint_info.json'
        
        if not info_path.exists():
            return None
            
        with open(info_path, 'r') as f:
            checkpoint_info = json.load(f)
            
        best_checkpoints = [c for c in checkpoint_info if c.get('is_best', False)]
        if not best_checkpoints:
            return None
            
        return Path(best_checkpoints[-1]['path']) 