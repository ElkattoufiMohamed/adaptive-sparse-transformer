#!/usr/bin/env python3
"""
Train Adaptive Sparse Transformer (multi-dataset ready):
- Iterates over one or many datasets from config.data.dataset_name
- Suffixes experiment-name per dataset (e.g., bench_imdb, bench_ag_news, ...)
- Keeps earlier fixes: cache_dir default, depth<->num_layers shim, no ellipses/dups
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

    # -------- data (cache_dir + dataset list) --------
    logger.info("Setting up data loaders...")
    data_cfg = config.get('data', {})
    subset_size = data_cfg.get('debug_subset_size', 2048) if args.debug else None

    # Resolve a safe default cache dir if none specified in YAML
    repo_root = Path(__file__).resolve().parents[1]
    default_cache = repo_root / "data" / ".cache"
    cache_dir = data_cfg.get('cache_dir') or str(default_cache)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Accept single dataset name or a list in YAML
    dataset_names = data_cfg.get('dataset_name', 'imdb')
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Store per-dataset summaries for a clean printout
    all_results: Dict[str, Any] = {}

    # Base experiment name (we suffix with dataset)
    base_exp_name = args.experiment_name

    for ds_name in dataset_names:
        logger.info(f"=== Dataset: {ds_name} ===")
        # Build dataloaders (no max_seq_len kw in constructor)
        dataset_loader = DatasetLoader(
            dataset_name=ds_name,
            tokenizer_name=data_cfg.get('tokenizer_name', 'bert-base-uncased'),
            cache_dir=cache_dir
        )
        dataloaders = dataset_loader.create_dataloaders(
            batch_size=config['training']['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            train_subset_size=subset_size,
            eval_subset_size=(subset_size // 4) if subset_size else None,
            pin_memory=config.get('hardware', {}).get('pin_memory', False),
            max_seq_len=config['model'].get('max_seq_len', 256),
        )

        # Effective num_classes per dataset (if provided by loader)
        effective_num_classes = None
        if hasattr(dataset_loader, "num_classes") and callable(dataset_loader.num_classes):
            try:
                effective_num_classes = int(dataset_loader.num_classes())
            except Exception:
                effective_num_classes = None

        md = dict(config['model'])  # copy to avoid mutating config across datasets
        if effective_num_classes is not None:
            md['num_classes'] = effective_num_classes

        # Debug overrides
        if args.debug:
            logger.debug("Debug mode: applying safe overrides.")
            lr_now = config['training'].get('learning_rate', 1e-3)
            config['training']['learning_rate'] = 5e-4 if lr_now < 5e-4 else 1e-4
            config['training']['warmup_steps'] = min(20, config['training'].get('warmup_steps', 20))
            md['dropout'] = max(md.get('dropout', 0.0), 0.1)
            config.setdefault('hardware', {})['pin_memory'] = (device.type == 'cuda')

        # Preflight
        if md['dim'] % md['num_heads'] != 0:
            raise ValueError(f"Model dim {md['dim']} must be divisible by num_heads {md['num_heads']}")

        total_steps = len(dataloaders['train']) * config['training']['num_epochs']
        if config['training']['warmup_steps'] > total_steps:
            logging.warning("warmup_steps > total steps; reducing to 10% of total.")
            config['training']['warmup_steps'] = max(1, total_steps // 10)

        # Helper: train a model
        def train_model(model_type: str, model: nn.Module) -> Dict[str, Any]:
            exp_name = f"{base_exp_name}_{ds_name}_{model_type}"
            logger.info(f"🚀 Training {model_type} on {ds_name}")
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

        # Build/train model(s)
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

        # Optional comparative benchmark per dataset
        if args.model_type == 'both':
            logger.info("🔍 Comparative benchmark...")
            # Re-load best checkpoints
            adaptive_checkpoint = torch.load(f"results/experiments/{base_exp_name}_{ds_name}_adaptive/best_model.pt", map_location=device)
            baseline_checkpoint = torch.load(f"results/experiments/{base_exp_name}_{ds_name}_baseline/best_model.pt", map_location=device)
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

            outp = Path("results") / "comparisons" / f"{base_exp_name}_{ds_name}_comparison.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, 'w') as f:
                json.dump(bench, f, indent=2)
            logger.info(f"Comparison saved: {outp}")

        # Save & print a concise per-dataset summary
        all_results[ds_name] = results
        logger.info(f"🎉 Finished dataset {ds_name}")
        print("\n" + "=" * 80)
        print(f"EXPERIMENT SUMMARY — DATASET: {ds_name}")
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

    # Optional: write a manifest of datasets processed
    manifest = Path("results") / "experiments" / f"{base_exp_name}_datasets.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump({"datasets": dataset_names}, f, indent=2)


if __name__ == "__main__":
    main()
