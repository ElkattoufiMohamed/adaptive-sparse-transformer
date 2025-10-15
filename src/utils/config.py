import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Basic model parameters
    dim: int = 512
    depth: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 2048
    vocab_size: int = 50000
    
    # Adaptive attention parameters
    local_window_size: int = 32
    global_ratio: float = 0.1
    learnable_sparsity: bool = True
    temperature: float = 1.0
    sparsity_ratio: float = 0.3
    
    # Model type
    model_type: str = "adaptive_transformer"
    
    def __post_init__(self):
        assert self.dim % self.num_heads == 0, f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
        assert 0 < self.dropout < 1, f"dropout must be between 0 and 1, got {self.dropout}"
        assert self.local_window_size > 0, f"local_window_size must be positive, got {self.local_window_size}"

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    lr_decay_steps: Optional[int] = None
    lr_decay_rate: float = 0.95
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Training settings
    accumulate_grad_batches: int = 1
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # Evaluation
    eval_every_n_steps: int = 500
    eval_batch_size: Optional[int] = None  # Uses batch_size if None
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"

@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset parameters
    dataset_name: str = "custom"
    data_dir: str = "./data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Data processing
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    min_length: int = 10
    stride: int = 256  # For sliding window tokenization
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Data augmentation
    random_masking: bool = False
    masking_prob: float = 0.15
    
    def __post_init__(self):
        assert self.max_length > self.min_length, f"max_length ({self.max_length}) must be > min_length ({self.min_length})"

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Experiment metadata
    experiment_name: str = "adaptive_attention_exp"
    project_name: str = "adaptive_transformers"
    description: str = ""
    tags: list = field(default_factory=list)
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Logging
    log_level: str = "INFO"
    log_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    
    # Wandb configuration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_gpus: int = 1
    
    def __post_init__(self):
        # Set wandb project name if not specified
        if self.use_wandb and self.wandb_project is None:
            self.wandb_project = self.project_name
        
        # Create directories
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    """Main configuration class combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file"""
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if filepath.suffix.lower() == '.yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

class ConfigManager:
    """Configuration manager for loading, saving, and merging configurations"""
    
    def __init__(self):
        self.config_cache = {}
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Config:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration file
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return Config.from_dict(config_dict)
    
    @staticmethod
    def create_default_config() -> Config:
        """Create default configuration"""
        return Config()
    
    @staticmethod
    def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
        """Merge configuration with overrides"""
        base_dict = base_config.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_config)
        return Config.from_dict(merged_dict)
    
    @staticmethod
    def from_args(args: Optional[argparse.Namespace] = None) -> Config:
        """Create configuration from command line arguments"""
        if args is None:
            # Parse command line arguments
            parser = ConfigManager.create_argument_parser()
            args = parser.parse_args()
        
        # Load base configuration
        if hasattr(args, 'config_file') and args.config_file:
            config = ConfigManager.load_config(args.config_file)
        else:
            config = ConfigManager.create_default_config()
        
        # Override with command line arguments
        overrides = {}
        
        # Model overrides
        if hasattr(args, 'dim') and args.dim:
            overrides.setdefault('model', {})['dim'] = args.dim
        if hasattr(args, 'num_layers') and args.num_layers:
            overrides.setdefault('model', {})['num_layers'] = args.num_layers
        if hasattr(args, 'num_heads') and args.num_heads:
            overrides.setdefault('model', {})['num_heads'] = args.num_heads
        
        # Training overrides
        if hasattr(args, 'batch_size') and args.batch_size:
            overrides.setdefault('training', {})['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
        if hasattr(args, 'num_epochs') and args.num_epochs:
            overrides.setdefault('training', {})['num_epochs'] = args.num_epochs
        
        # Experiment overrides
        if hasattr(args, 'experiment_name') and args.experiment_name:
            overrides.setdefault('experiment', {})['experiment_name'] = args.experiment_name
        if hasattr(args, 'output_dir') and args.output_dir:
            overrides.setdefault('experiment', {})['output_dir'] = args.output_dir
        if hasattr(args, 'use_wandb') and args.use_wandb is not None:
            overrides.setdefault('experiment', {})['use_wandb'] = args.use_wandb
        
        if overrides:
            config = ConfigManager.merge_configs(config, overrides)
        
        return config
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(description='Adaptive Transformer Training')
        
        # Configuration file
        parser.add_argument('--config-file', type=str, help='Path to configuration file')
        
        # Model arguments
        model_group = parser.add_argument_group('model')
        model_group.add_argument('--dim', type=int, help='Model dimension')
        model_group.add_argument('--num-layers', type=int, help='Number of transformer layers')
        model_group.add_argument('--num-heads', type=int, help='Number of attention heads')
        model_group.add_argument('--local-window-size', type=int, help='Local attention window size')
        
        # Training arguments
        training_group = parser.add_argument_group('training')
        training_group.add_argument('--batch-size', type=int, help='Batch size')
        training_group.add_argument('--learning-rate', type=float, help='Learning rate')
        training_group.add_argument('--num-epochs', type=int, help='Number of training epochs')
        training_group.add_argument('--weight-decay', type=float, help='Weight decay')
        
        # Data arguments
        data_group = parser.add_argument_group('data')
        data_group.add_argument('--data-dir', type=str, help='Data directory')
        data_group.add_argument('--dataset-name', type=str, help='Dataset name')
        data_group.add_argument('--max-length', type=int, help='Maximum sequence length')
        
        # Experiment arguments
        exp_group = parser.add_argument_group('experiment')
        exp_group.add_argument('--experiment-name', type=str, help='Experiment name')
        exp_group.add_argument('--output-dir', type=str, help='Output directory')
        exp_group.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
        exp_group.add_argument('--seed', type=int, help='Random seed')
        
        return parser
    
    @staticmethod
    def save_experiment_config(config: Config, experiment_dir: Union[str, Path]) -> Path:
        """Save configuration to experiment directory"""
        experiment_dir = Path(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = experiment_dir / f"config_{timestamp}.yaml"
        
        config.save(config_file)
        
        # Also save as latest config
        latest_config = experiment_dir / "config_latest.yaml"
        config.save(latest_config)
        
        return config_file
    
    @staticmethod
    def validate_config(config: Config) -> None:
        """Validate configuration values"""
        # Model validation
        assert config.model.dim > 0, "Model dimension must be positive"
        assert config.model.num_layers > 0, "Number of layers must be positive"
        assert config.model.num_heads > 0, "Number of heads must be positive"
        assert config.model.dim % config.model.num_heads == 0, "Model dim must be divisible by num_heads"
        
        # Training validation
        assert config.training.batch_size > 0, "Batch size must be positive"
        assert config.training.learning_rate > 0, "Learning rate must be positive"
        assert config.training.num_epochs > 0, "Number of epochs must be positive"
        
        # Data validation
        assert config.data.max_length > 0, "Max sequence length must be positive"
        assert config.data.num_workers >= 0, "Number of workers must be non-negative"
        
        print("✓ Configuration validation passed")

# Utility function for quick config creation
def get_config(config_path: Optional[Union[str, Path]] = None, **kwargs) -> Config:
    """
    Quick utility function to get configuration
    
    Args:
        config_path: Path to configuration file (optional)
        **kwargs: Override parameters
    
    Returns:
        Config object
    """
    if config_path:
        config = ConfigManager.load_config(config_path)
    else:
        config = ConfigManager.create_default_config()
    
    if kwargs:
        config = ConfigManager.merge_configs(config, kwargs)
    
    ConfigManager.validate_config(config)
    return config

# Example usage and configuration templates
def create_example_configs():
    """Create example configuration files"""
    configs_dir = Path("configs/experiment_configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Small model configuration
    small_config = Config(
        model=ModelConfig(
            dim=256,
            num_layers=4,
            num_heads=4,
            local_window_size=16
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=3e-4,
            num_epochs=50
        ),
        experiment=ExperimentConfig(
            experiment_name="small_adaptive_transformer",
            description="Small model for quick experimentation"
        )
    )
    small_config.save(configs_dir / "small_model.yaml")
    
    # Large model configuration
    large_config = Config(
        model=ModelConfig(
            dim=768,
            num_layers=12,
            num_heads=12,
            local_window_size=64,
            max_seq_length=4096
        ),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=100,
            mixed_precision=True
        ),
        experiment=ExperimentConfig(
            experiment_name="large_adaptive_transformer",
            description="Large model for production training",
            use_wandb=True
        )
    )
    large_config.save(configs_dir / "large_model.yaml")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create_examples":
        create_example_configs()
        print("Example configurations created in configs/experiment_configs/")
    else:
        # Load configuration from command line
        config = ConfigManager.from_args()
        print("Configuration loaded successfully!")
        print(f"Model dim: {config.model.dim}")
        print(f"Experiment name: {config.experiment.experiment_name}")