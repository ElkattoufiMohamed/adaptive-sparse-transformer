#!/usr/bin/env python3
"""
Training script for adaptive sparse transformer.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import torch.nn as nn
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import yaml
from utils.config import ConfigManager
from utils.helpers import set_seed, setup_logging
from models.transformer import AdaptiveSparseTransformer
from models.baseline_transformer import BaselineTransformer
from data.datasets import DatasetLoader
from training.trainer import create_trainer
from evaluation.metrics import analyze_attention_patterns, benchmark_models

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Adaptive Sparse Transformer')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, required=True,
                       help='Name for this experiment')
    parser.add_argument('--model-type', type=str, choices=['adaptive', 'baseline', 'both'],
                       default='adaptive', help='Which model to train')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with small dataset')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['training']['seed'])

        # Setup logging (respect debug flag)
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    set_seed(config['training']['seed'])

    # Determine device (respect config hardware.device if provided)
    hw_cfg = config.get('hardware', {})
    requested_device = hw_cfg.get('device', 'auto')
    if requested_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif requested_device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            logger.warning("Requested CUDA device but CUDA is not available — falling back to CPU.")
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device} (requested: {requested_device})")

    # Adjust pin_memory based on device
    if hw_cfg.get('pin_memory', True) and device.type != 'cuda':
        logger.debug("pin_memory was enabled in config but no CUDA found — disabling pin_memory for DataLoaders.")
        hw_cfg['pin_memory'] = False
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Setting up data loaders...")
    data_config = config['data']
    
    dataset_loader = DatasetLoader(
        dataset_name=data_config['dataset_name'],
        tokenizer_name='bert-base-uncased',
        max_length=data_config['max_length'],
        cache_dir=data_config.get('cache_dir', './data/cache'),
        seed=config['training']['seed']
    )
    
    # Update config with actual number of classes
    config['model']['num_classes'] = dataset_loader.get_num_classes()
    
    # Create dataloaders
    subset_size = 1000 if args.debug else None
    dataloaders = dataset_loader.create_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        train_subset_size=subset_size,
        eval_subset_size=subset_size // 4 if subset_size else None
    )
    
        # If debug mode, override some training params to be safe / fast
    if args.debug:
        logger.debug("Debug mode: applying safe overrides to config (LR, warmup_steps, dropout, subset sizes).")
        # Lower LR for stability
        config['training']['learning_rate'] = config['training'].get('learning_rate', 1e-3) if config['training'].get('learning_rate', 1e-3) < 5e-4 else 1e-4
        # Reduce warmup to small number so lr scheduling actually moves
        config['training']['warmup_steps'] = min(20, config['training'].get('warmup_steps', 20))
        # Add some dropout to avoid pathological overfitting/explosion
        config['model']['dropout'] = max(config['model'].get('dropout', 0.0), 0.1)
        # Ensure pin_memory is False on CPU debug
        config['hardware']['pin_memory'] = False
        logger.debug(f"Overrides applied: learning_rate={config['training']['learning_rate']}, warmup_steps={config['training']['warmup_steps']}, dropout={config['model']['dropout']}")

    # Training function
    def train_model(model_type: str, model: nn.Module) -> Dict[str, Any]:
        """Train a single model and return results."""
        
        experiment_name = f"{args.experiment_name}_{model_type}"
        logger.info(f"🚀 Starting training for {model_type} model")
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=dataloaders['train'],
            eval_loader=dataloaders['eval'],
            config=config,
            experiment_name=experiment_name,
            device=device
        )
        
        # Train the model
        training_summary = trainer.train()
        
        # Analyze attention patterns (only for adaptive model)
        if model_type == 'adaptive':
            logger.info("Analyzing attention patterns...")
            attention_analysis = analyze_attention_patterns(
                model=model,
                dataloader=dataloaders['eval'],
                device=device,
                num_samples=200,
                save_path=f"results/figures/{experiment_name}_attention_patterns.png"
            )
            training_summary['attention_analysis'] = attention_analysis
        
        return training_summary
    
    # Train models based on selection
    results = {}

        # Pre-flight checks
    md = config['model']
    if md['dim'] % md['num_heads'] != 0:
        raise ValueError(f"Model dimension {md['dim']} is not divisible by num_heads {md['num_heads']}")
    if config['training']['warmup_steps'] > (len(dataloaders['train']) * config['training']['num_epochs']):
        logger.warning("warmup_steps > total_steps. Reducing warmup_steps to 10% of total steps.")
        total_steps = len(dataloaders['train']) * config['training']['num_epochs']
        config['training']['warmup_steps'] = max(1, total_steps // 10)
    
    if args.model_type in ['adaptive', 'both']:
        # Create adaptive model
        logger.info("Creating adaptive sparse transformer...")
        adaptive_model = AdaptiveSparseTransformer(
            vocab_size=config['model']['vocab_size'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            max_seq_len=config['model']['max_seq_len'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes'],
            **config['attention']
        )

        # quick model diagnostics
        total_params = sum(p.numel() for p in adaptive_model.parameters())
        trainable_params = sum(p.numel() for p in adaptive_model.parameters() if p.requires_grad)
        logger.info(f"Adaptive model created - total_params={total_params:,}, trainable_params={trainable_params:,}")
        logger.debug(f"Model config: dim={config['model']['dim']}, num_heads={config['model']['num_heads']}, max_seq_len={config['model']['max_seq_len']}")

        
        results['adaptive'] = train_model('adaptive', adaptive_model)
    
    if args.model_type in ['baseline', 'both']:
        # Create baseline model
        logger.info("Creating baseline transformer...")
        baseline_model = BaselineTransformer(
            vocab_size=config['model']['vocab_size'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            max_seq_len=config['model']['max_seq_len'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes']
        )
        
        results['baseline'] = train_model('baseline', baseline_model)
    
    # Comparative analysis if both models were trained
    if args.model_type == 'both':
        logger.info("🔍 Running comparative analysis...")
        
        # Load best models for comparison
        adaptive_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_adaptive/best_model.pt")
        baseline_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_baseline/best_model.pt")
        
        adaptive_model.load_state_dict(adaptive_checkpoint['model_state_dict'])
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        
        # Benchmark comparison
        benchmark_results = benchmark_models(
            adaptive_model=adaptive_model,
            baseline_model=baseline_model,
            dataloader=dataloaders['eval'],
            device=device,
            num_batches=50
        )
        
        results['comparison'] = benchmark_results
        
        # Save comparison results
        comparison_path = Path("results") / "comparisons" / f"{args.experiment_name}_comparison.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(comparison_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Comparison results saved to {comparison_path}")
    
    logger.info("🎉 All experiments completed successfully!")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for model_type, summary in results.items():
        if model_type != 'comparison':
            final_metrics = summary.get('final_eval_metrics', {})
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Final Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
            print(f"  Final F1: {final_metrics.get('eval_f1', 0):.4f}")
            print(f"  Training Time: {summary.get('training_time', 0):.2f}s")
    
    if 'comparison' in results:
        improvements = results['comparison'].get('improvements', {})
        print(f"\nIMPROVEMENTS:")
        for metric, improvement in improvements.items():
            print(f"  {metric}: {improvement:+.2f}%")

if __name__ == "__main__":
    main()