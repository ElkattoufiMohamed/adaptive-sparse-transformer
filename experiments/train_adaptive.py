#!/usr/bin/env python3
"""
Train Adaptive Sparse Transformer (cleaned).
- Fixes truncated tokens and duplicate blocks
- Adds missing imports
- Adds depth <-> num_layers compatibility shim
- Keeps debug overrides and comparison run
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn

# Add src/ to path (so imports work when run from repo root)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import set_seed, setup_logging
from models.transformer import AdaptiveSparseTransformer
from models.baseline_transformer import BaselineTransformer
from data.datasets import DatasetLoader
from training.trainer import create_trainer
from evaluation.metrics import analyze_attention_patterns, benchmark_models


def _normalize_model_depth_keys(config: Dict[str, Any]) -> None:
    """
    Support both 'depth' and 'num_layers' in config['model'].
    Prefer 'depth' internally.
    """
    m = config.get('model', {})
    if 'depth' not in m and 'num_layers' in m:
        m['depth'] = m.pop('num_layers')
    # keep alias if someone accesses it elsewhere
    if 'num_layers' not in m and 'depth' in m:
        m['num_layers'] = m['depth']
    config['model'] = m


def _resolve_device(config: Dict[str, Any], logger: logging.Logger) -> torch.device:
    hw_cfg = config.get('hardware', {})
    requested = hw_cfg.get('device', 'auto')
    if requested == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif requested == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            logger.warning("Requested CUDA but CUDA is not available — falling back to CPU.")
    else:
        device = torch.device('cpu')

    # Adjust pin_memory if on CPU
    if hw_cfg.get('pin_memory', True) and device.type != 'cuda':
        logger.debug("pin_memory enabled in config but running on CPU — disabling for DataLoaders.")
        hw_cfg['pin_memory'] = False
        config['hardware'] = hw_cfg

    logger.info(f"Using device: {device} (requested: {requested})")
    return device


def main():
    # -----------------------
    # Parse arguments
    # -----------------------
    parser = argparse.ArgumentParser(description='Train Adaptive Sparse Transformer')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name for this experiment (used in result paths)')
    parser.add_argument('--model-type', type=str, choices=['adaptive', 'baseline', 'both'],
                        default='adaptive', help='Which model(s) to train')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small dataset and safer hyperparams')
    args = parser.parse_args()

    # -----------------------
    # Logging
    # -----------------------
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # -----------------------
    # Load configuration (YAML)
    # -----------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Normalize model depth keys
    _normalize_model_depth_keys(config)

    # -----------------------
    # Seed & Device
    # -----------------------
    set_seed(config['training']['seed'])
    device = _resolve_device(config, logger)

    # -----------------------
    # Data
    # -----------------------
    logger.info("Setting up data loaders...")
    data_config = config.get('data', {})
    subset_size = data_config.get('debug_subset_size', 2048) if args.debug else None

    dataset_loader = DatasetLoader(
        dataset_name=data_config.get('dataset_name', 'ag_news'),
        tokenizer_name=data_config.get('tokenizer_name', 'bert-base-uncased'),
        max_seq_len=config['model']['max_seq_len'],
        cache_dir=data_config.get('cache_dir', None)
    )

    dataloaders = dataset_loader.create_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        train_subset_size=subset_size,
        eval_subset_size=(subset_size // 4) if subset_size else None,
        pin_memory=config.get('hardware', {}).get('pin_memory', False)
    )

    # -----------------------
    # Debug overrides
    # -----------------------
    if args.debug:
        logger.debug("Debug mode: applying safe overrides (lr, warmup, dropout, pin_memory).")
        # Lower LR for stability in tiny runs
        current_lr = config['training'].get('learning_rate', 1e-3)
        config['training']['learning_rate'] = 5e-4 if current_lr < 5e-4 else 1e-4
        # Make schedulers move
        config['training']['warmup_steps'] = min(20, config['training'].get('warmup_steps', 20))
        # Add some dropout
        config['model']['dropout'] = max(config['model'].get('dropout', 0.0), 0.1)
        # Ensure pin_memory False on CPU
        config.setdefault('hardware', {})['pin_memory'] = (device.type == 'cuda')
        logger.debug(
            f"Overrides applied: learning_rate={config['training']['learning_rate']}, "
            f"warmup_steps={config['training']['warmup_steps']}, "
            f"dropout={config['model']['dropout']}, "
            f"pin_memory={config['hardware']['pin_memory']}"
        )

    # -----------------------
    # Pre-flight checks
    # -----------------------
    md = config['model']
    if md['dim'] % md['num_heads'] != 0:
        raise ValueError(f"Model dim {md['dim']} must be divisible by num_heads {md['num_heads']}")

    total_steps = len(dataloaders['train']) * config['training']['num_epochs']
    if config['training']['warmup_steps'] > total_steps:
        logger.warning("warmup_steps > total training steps; reducing warmup_steps to 10% of total steps.")
        config['training']['warmup_steps'] = max(1, total_steps // 10)

    # -----------------------
    # Training helper
    # -----------------------
    def train_model(model_type: str, model: nn.Module) -> Dict[str, Any]:
        """Train a single model and return results dict (including eval metrics)."""
        experiment_name = f"{args.experiment_name}_{model_type}"
        logger.info(f"🚀 Starting training for {model_type} model")

        trainer = create_trainer(
            model=model,
            train_loader=dataloaders['train'],
            eval_loader=dataloaders['eval'],
            config=config,
            experiment_name=experiment_name,
            device=device
        )

        summary = trainer.train()

        # Optional: analyze attention for adaptive model
        if model_type == 'adaptive':
            logger.info("Analyzing attention patterns...")
            attention_report = analyze_attention_patterns(
                model=model,
                dataloader=dataloaders['eval'],
                device=device,
                num_batches=10
            )
            summary['attention_analysis'] = attention_report

        return summary

    results: Dict[str, Any] = {}

    # -----------------------
    # Build & train model(s)
    # -----------------------
    if args.model_type in ['adaptive', 'both']:
        logger.info("Creating adaptive sparse transformer...")
        adaptive_model = AdaptiveSparseTransformer(
            vocab_size=md['vocab_size'],
            dim=md['dim'],
            depth=md['depth'],
            num_heads=md['num_heads'],
            mlp_ratio=md['mlp_ratio'],
            max_seq_len=md['max_seq_len'],
            dropout=md['dropout'],
            num_classes=md['num_classes'],
            **config.get('attention', {})
        ).to(device)
        # quick diagnostics
        total_params = sum(p.numel() for p in adaptive_model.parameters())
        trainable_params = sum(p.numel() for p in adaptive_model.parameters() if p.requires_grad)
        logger.info(f"Adaptive model created - total_params={total_params:,}, trainable_params={trainable_params:,}")
        results['adaptive'] = train_model('adaptive', adaptive_model)

    if args.model_type in ['baseline', 'both']:
        logger.info("Creating baseline transformer...")
        baseline_model = BaselineTransformer(
            vocab_size=md['vocab_size'],
            dim=md['dim'],
            depth=md['depth'],
            num_heads=md['num_heads'],
            mlp_ratio=md['mlp_ratio'],
            max_seq_len=md['max_seq_len'],
            dropout=md['dropout'],
            num_classes=md['num_classes']
        ).to(device)
        results['baseline'] = train_model('baseline', baseline_model)

    # -----------------------
    # Comparative analysis
    # -----------------------
    if args.model_type == 'both':
        logger.info("🔍 Running comparative analysis...")
        # Load best checkpoints
        adaptive_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_adaptive/best_model.pt", map_location=device)
        baseline_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_baseline/best_model.pt", map_location=device)
        adaptive_model.load_state_dict(adaptive_checkpoint['model_state_dict'])
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])

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

    # -----------------------
    # Summary printout
    # -----------------------
    logger.info("🎉 All experiments completed successfully!")
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

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
