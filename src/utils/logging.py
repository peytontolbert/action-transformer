import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime

class ExperimentLogger:
    """Logger for tracking experiment metrics and metadata."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        flush_frequency: int = 1
    ):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            metadata: Optional experiment metadata
            flush_frequency: How often to flush metrics to disk
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.flush_frequency = flush_frequency
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric tracking
        self.metrics: Dict[str, List[float]] = {}
        self.step_metrics: Dict[str, List[float]] = {}
        self.current_step = 0
        self.last_flush = 0
        
        # Save metadata
        self.metadata = metadata or {}
        self.metadata.update({
            'experiment_name': experiment_name,
            'start_time': timestamp,
            'log_dir': str(self.experiment_dir)
        })
        self._save_metadata()
        
        # Initialize CSV files
        self.metric_file = self.experiment_dir / 'metrics.csv'
        self.step_metric_file = self.experiment_dir / 'step_metrics.csv'
        self._initialize_csv_files()
        
    def _save_metadata(self):
        """Save experiment metadata to JSON."""
        metadata_file = self.experiment_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _initialize_csv_files(self):
        """Initialize CSV files with headers."""
        # Epoch metrics
        with open(self.metric_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'timestamp'] + ['placeholder'])
            
        # Step metrics
        with open(self.step_metric_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'timestamp'] + ['placeholder'])
            
    def _update_csv_headers(self, metric_names: List[str], is_step_metric: bool = False):
        """Update CSV headers with new metric names."""
        file_path = self.step_metric_file if is_step_metric else self.metric_file
        temp_file = file_path.with_suffix('.tmp')
        
        # Read existing data
        with open(file_path, 'r', newline='') as f_in:
            reader = csv.reader(f_in)
            data = list(reader)
            
        # Update header
        if data:
            header = ['epoch' if not is_step_metric else 'step', 'timestamp'] + sorted(metric_names)
            data[0] = header
            
            # Write updated data
            with open(temp_file, 'w', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerows(data)
                
            # Replace original file
            temp_file.replace(file_path)
            
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (if None, epoch-based metric)
        """
        if step is not None:
            # Step-based metric
            if name not in self.step_metrics:
                self.step_metrics[name] = []
                self._update_csv_headers(list(self.step_metrics.keys()), True)
            self.step_metrics[name].append(value)
            self.current_step = step
        else:
            # Epoch-based metric
            if name not in self.metrics:
                self.metrics[name] = []
                self._update_csv_headers(list(self.metrics.keys()), False)
            self.metrics[name].append(value)
            
        # Flush if needed
        if self.current_step - self.last_flush >= self.flush_frequency:
            self.flush()
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def flush(self):
        """Flush metrics to disk."""
        # Flush epoch metrics
        if self.metrics:
            with open(self.metric_file, 'a', newline='') as f:
                writer = csv.writer(f)
                epoch = len(next(iter(self.metrics.values()))) - 1
                row = [epoch, time.time()] + [self.metrics[name][-1] for name in sorted(self.metrics.keys())]
                writer.writerow(row)
                
        # Flush step metrics
        if self.step_metrics:
            with open(self.step_metric_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [self.current_step, time.time()] + [self.step_metrics[name][-1] for name in sorted(self.step_metrics.keys())]
                writer.writerow(row)
                
        self.last_flush = self.current_step
        
    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric values
        """
        if name in self.metrics:
            return self.metrics[name]
        if name in self.step_metrics:
            return self.step_metrics[name]
        raise KeyError(f"Metric '{name}' not found")
        
    def get_latest_metric(self, name: str) -> float:
        """Get latest value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest metric value
        """
        values = self.get_metric(name)
        return values[-1] if values else float('nan')
        
    def get_metric_mean(self, name: str, window: Optional[int] = None) -> float:
        """Get mean value of a metric.
        
        Args:
            name: Metric name
            window: Optional window size for moving average
            
        Returns:
            Mean metric value
        """
        values = self.get_metric(name)
        if window:
            values = values[-window:]
        return np.mean(values) if values else float('nan')
        
    def save(self):
        """Save all metrics and flush to disk."""
        self.flush()
        
        # Save all metrics to JSON for easier loading
        metrics_file = self.experiment_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'step_metrics': self.step_metrics
            }, f, indent=2)
            
    @staticmethod
    def load(experiment_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load metrics from an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Dictionary containing metrics and metadata
        """
        experiment_dir = Path(experiment_dir)
        
        # Load metadata
        metadata_file = experiment_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Load metrics
        metrics_file = experiment_dir / 'metrics.json'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        return {
            'metadata': metadata,
            'metrics': metrics['metrics'],
            'step_metrics': metrics['step_metrics']
        } 