"""
Utility functions for training and experimentation.
"""

import torch
import random
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to {seed}")

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """Setup logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from other libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    metrics: Dict[str, float],
    save_path: str,
    config: Dict[str, Any]
):
    """Save model checkpoint with all necessary information."""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'model_class': model.__class__.__name__
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True
) -> Dict[str, Any]:
    """Load model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    return checkpoint

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in model."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    
    memory_stats = {}
    
    # CPU memory
    import psutil
    process = psutil.Process()
    memory_stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_stats['gpu_memory_max_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return memory_stats

def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

class ExperimentTracker:
    """Track multiple experiments and their results."""
    
    def __init__(self, experiments_dir: str = "results/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiment_registry.json"
        
        # Load existing experiments
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
    
    def register_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        results: Dict[str, Any]
    ):
        """Register a new experiment."""
        
        experiment_data = {
            'config': config,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed'
        }
        
        self.experiments[experiment_name] = experiment_data
        
        # Save to file
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        print(f"Experiment {experiment_name} registered")
    
    def get_best_experiments(self, metric: str = 'eval_accuracy', top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k experiments by specified metric."""
        
        experiment_scores = []
        for name, data in self.experiments.items():
            final_metrics = data['results'].get('final_eval_metrics', {})
            score = final_metrics.get(metric, 0)
            experiment_scores.append((name, score))
        
        # Sort by score (descending)
        experiment_scores.sort(key=lambda x: x[1], reverse=True)
        
        return experiment_scores[:top_k]
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        
        comparison = {
            'experiments': {},
            'summary': {}
        }
        
        for name in experiment_names:
            if name in self.experiments:
                comparison['experiments'][name] = self.experiments[name]['results']
        
        # Compute relative improvements
        if len(comparison['experiments']) >= 2:
            # Use first experiment as baseline
            baseline_name = experiment_names[0]
            baseline_metrics = comparison['experiments'][baseline_name]['final_eval_metrics']
            
            for exp_name, exp_data in comparison['experiments'].items():
                if exp_name != baseline_name:
                    exp_metrics = exp_data['final_eval_metrics']
                    improvements = {}
                    
                    for metric, value in exp_metrics.items():
                        if metric in baseline_metrics:
                            baseline_value = baseline_metrics[metric]
                            if baseline_value != 0:
                                improvement = (value - baseline_value) / baseline_value * 100
                                improvements[f'{metric}_improvement'] = improvement
                    
                    comparison['summary'][f'{exp_name}_vs_{baseline_name}'] = improvements
        
        return comparison