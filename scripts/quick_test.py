#!/usr/bin/env python3
"""
Quick test script to verify everything works before full training.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import yaml

from data.datasets import DatasetLoader
from models.baseline_transformer import BaselineTransformer
from models.transformer import AdaptiveSparseTransformer
from utils.helpers import set_seed, setup_logging


def quick_test():
    """Run a quick test of the entire pipeline."""

    print("🧪 Running quick pipeline test...")

    # Setup
    setup_logging()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load minimal config
    config = {
        "model": {
            "vocab_size": 30522,
            "dim": 128,  # Small for testing
            "depth": 2,  # Shallow for speed
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "max_seq_len": 64,
            "dropout": 0.1,
            "num_classes": 2,
        },
        "attention": {
            "local_window_size": 16,
            "global_ratio": 0.1,
            "learnable_sparsity": True,
            "temperature": 1.0,
        },
        "training": {"batch_size": 4, "learning_rate": 1e-4, "num_epochs": 1},
        "data": {"dataset_name": "imdb", "max_length": 64},
    }

    # Test data loading
    print("📊 Testing data loading...")
    dataset_loader = DatasetLoader(
        dataset_name="imdb", max_length=64, cache_dir="./test_cache"
    )

    # Load tiny subset for testing
    train_dataset = dataset_loader.load_dataset("train", subset_size=20)
    eval_dataset = dataset_loader.load_dataset("test", subset_size=10)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(
        f"✅ Data loading successful - Train: {len(train_loader)} batches, Eval: {len(eval_loader)} batches"
    )

    # Test models
    print("🤖 Testing models...")

    # Adaptive model
    # Merge configs without duplicates
    model_config = {**config["model"], **config["attention"]}
    adaptive_model = AdaptiveSparseTransformer(**model_config)
    adaptive_model.to(device)

    # Baseline model
    baseline_model = BaselineTransformer(**config["model"])
    baseline_model.to(device)

    print(
        f"✅ Models created - Adaptive params: {sum(p.numel() for p in adaptive_model.parameters()):,}"
    )
    print(
        f"                   Baseline params: {sum(p.numel() for p in baseline_model.parameters()):,}"
    )

    # Test forward passes
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Adaptive model forward pass
    adaptive_model.eval()
    with torch.no_grad():
        adaptive_output = adaptive_model(
            batch["input_ids"], batch["attention_mask"], return_attention_info=True
        )

    # Baseline model forward pass
    baseline_model.eval()
    with torch.no_grad():
        baseline_output = baseline_model(batch["input_ids"], batch["attention_mask"])

    print("✅ Forward passes successful")
    print(f"   Adaptive output shape: {adaptive_output['logits'].shape}")
    print(f"   Baseline output shape: {baseline_output['logits'].shape}")

    # Test attention analysis
    if "attention_info" in adaptive_output:
        attention_info = adaptive_output["attention_info"]
        print(f"   Attention layers analyzed: {len(attention_info)}")

        if attention_info and "pattern_weights" in attention_info[0]:
            pattern_weights = attention_info[0]["pattern_weights"]
            print(f"   Pattern weights shape: {pattern_weights.shape}")
            print(
                f"   Average pattern usage - Local: {pattern_weights[:, 0].mean():.3f}, "
                f"Global: {pattern_weights[:, 1].mean():.3f}, "
                f"Sparse: {pattern_weights[:, 2].mean():.3f}"
            )

    # Test training step
    print("🏃 Testing training step...")

    adaptive_model.train()
    optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    outputs = adaptive_model(batch["input_ids"], batch["attention_mask"])
    loss = criterion(outputs["logits"], batch["labels"])

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"✅ Training step successful - Loss: {loss.item():.4f}")

    print("\n🎉 All tests passed! Ready for full training.")


if __name__ == "__main__":
    quick_test()
