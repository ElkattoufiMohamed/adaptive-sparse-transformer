#!/usr/bin/env python3
"""
Train Adaptive Sparse Transformer (clean, with dataloader fixes):
- Removes truncated tokens and duplicate setup
- Adds missing imports
- Adds depth <-> num_layers compatibility shim
- DEBUG overrides preserved
- Does NOT pass max_seq_len to DatasetLoader.__init__
- Optionally forwards max_seq_len to create_dataloaders if supported
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

# Ensure repo src/ is importable when run from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import set_seed, setup_logging
from models.transformer import AdaptiveSparseTransformer
from models.baseline_transformer import BaselineTransformer
from data.datasets import DatasetLoader
from training.trainer import create_trainer
from evaluation.metrics import analyze_attention_patterns, benchmark_models


def _normalize_model_depth_keys(config: Dict[str, Any]) -> None:
    m = config.get('model', {})
    if 'depth' not in m and 'num_layers' in m:
        m['depth'] = m.pop('num_layers')
    if 'num_layers' not in m and 'depth' in m:
        m['num_layers'] = m['depth']
    config['model'] = m


def _resolve_device(config: Dict[str, Any], logger: logging.Logger) -> torch.device:
    hw = config.get('hardware', {})
    requested = hw.get('device', 'auto')
    if requested == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif requested == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            logger.warning("Requested CUDA but CUDA is not available — falling back to CPU.")
    else:
        device = torch.device('cpu')

    if hw.get('pin_memory', True) and device.type != 'cuda':
        hw['pin_memory'] = False
        config['hardware'] = hw

    logger.info(f"Using device: {device} (requested: {requested})")
    return device


def main():
    # -------- args --------
    p = argparse.ArgumentParser(description='Train Adaptive Sparse Transformer')
    p.add_argument('--config', type=str, default='configs/base_config.yaml')
    p.add_argument('--experiment-name', type=str, required=True)
    p.add_argument('--model-type', type=str, choices=['adaptive', 'baseline', 'both'], default='adaptive')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    # -------- logging --------
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # -------- config --------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    _normalize_model_depth_keys(config)

    # -------- seed/device --------
    set_seed(config['training']['seed'])
    device = _resolve_device(config, logger)

    # -------- data (NO max_seq_len in DatasetLoader.__init__) --------
    logger.info("Setting up data loaders...")
    data_cfg = config.get('data', {})
    subset_size = data_cfg.get('debug_subset_size', 2048) if args.debug else None

    dataset_loader = DatasetLoader(
        dataset_name=data_cfg.get('dataset_name', 'ag_news'),
        tokenizer_name=data_cfg.get('tokenizer_name', 'bert-base-uncased'),
        cache_dir=data_cfg.get('cache_dir', None)
    )

    # If your DatasetLoader.create_dataloaders supports max_seq_len, keep the kw line; otherwise delete it.
    create_dl_kwargs = dict(
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        train_subset_size=subset_size,
        eval_subset_size=(subset_size // 4) if subset_size else None,
        pin_memory=config.get('hardware', {}).get('pin_memory', False)
    )
    # Optional pass-through:
    if 'max_seq_len' in config.get('model', {}):
        create_dl_kwargs['max_seq_len'] = config['model']['max_seq_len']

    dataloaders = dataset_loader.create_dataloaders(**create_dl_kwargs)

    # -------- debug overrides --------
    if args.debug:
        logger.debug("Debug mode: applying safe overrides.")
        lr_now = config['training'].get('learning_rate', 1e-3)
        config['training']['learning_rate'] = 5e-4 if lr_now < 5e-4 else 1e-4
        config['training']['warmup_steps'] = min(20, config['training'].get('warmup_steps', 20))
        config['model']['dropout'] = max(config['model'].get('dropout', 0.0), 0.1)
        config.setdefault('hardware', {})['pin_memory'] = (device.type == 'cuda')

    # -------- preflight --------
    md = config['model']
    if md['dim'] % md['num_heads'] != 0:
        raise ValueError(f"Model dim {md['dim']} must be divisible by num_heads {md['num_heads']}")

    total_steps = len(dataloaders['train']) * config['training']['num_epochs']
    if config['training']['warmup_steps'] > total_steps:
        logging.warning("warmup_steps > total steps; reducing to 10% of total.")
        config['training']['warmup_steps'] = max(1, total_steps // 10)

    # -------- train helper --------
    def train_model(model_type: str, model: nn.Module) -> Dict[str, Any]:
        exp_name = f"{args.experiment_name}_{model_type}"
        logger.info(f"🚀 Training {model_type}")
        trainer = create_trainer(
            model=model,
            train_loader=dataloaders['train'],
            eval_loader=dataloaders['eval'],
            config=config,
            experiment_name=exp_name,
            device=device
        )
        summary = trainer.train()
        if model_type == 'adaptive':
            logger.info("Analyzing attention patterns...")
            summary['attention_analysis'] = analyze_attention_patterns(
                model=model, dataloader=dataloaders['eval'], device=device, num_batches=10
            )
        return summary

    results: Dict[str, Any] = {}

    # -------- build/train --------
    if args.model_type in ['adaptive', 'both']:
        logger.info("Creating adaptive model...")
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
        logger.info(f"Adaptive params: {sum(p.numel() for p in adaptive_model.parameters()):,}")
        results['adaptive'] = train_model('adaptive', adaptive_model)

    if args.model_type in ['baseline', 'both']:
        logger.info("Creating baseline model...")
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

    # -------- compare --------
    if args.model_type == 'both':
        logger.info("🔍 Comparative benchmark...")
        adaptive_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_adaptive/best_model.pt", map_location=device)
        baseline_checkpoint = torch.load(f"results/experiments/{args.experiment_name}_baseline/best_model.pt", map_location=device)
        adaptive_model.load_state_dict(adaptive_checkpoint['model_state_dict'])
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])

        bench = benchmark_models(
            adaptive_model=adaptive_model,
            baseline_model=baseline_model,
            dataloader=dataloaders['eval'],
            device=device,
            num_batches=50
        )
        results['comparison'] = bench

        outp = Path("results") / "comparisons" / f"{args.experiment_name}_comparison.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w') as f:
            json.dump(bench, f, indent=2)
        logger.info(f"Comparison saved: {outp}")

    # -------- summary --------
    logger.info("🎉 Done.")
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for mtype, summary in results.items():
        if mtype == 'comparison':
            continue
        fm = summary.get('final_eval_metrics', {})
        print(f"\n{mtype.upper()} MODEL:")
        print(f"  Final Accuracy: {fm.get('eval_accuracy', 0):.4f}")
        print(f"  Final F1:       {fm.get('eval_f1', 0):.4f}")
        print(f"  Training Time:  {summary.get('training_time', 0):.2f}s")

    if 'comparison' in results:
        print("\nIMPROVEMENTS:")
        for k, v in results['comparison']['improvements'].items():
            print(f"  {k}: {v:+.2f}%")


if __name__ == "__main__":
    main()
