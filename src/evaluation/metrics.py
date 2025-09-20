"""
Comprehensive evaluation metrics for transformer models.
Includes standard ML metrics plus efficiency analysis.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(
    predictions: List[int],
    labels: List[int],
    num_classes: int = 2
) -> Dict[str, float]:
    """Compute standard and per-class classification metrics."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Add per-class metrics for small class counts
    if num_classes <= 10:
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        for i in range(len(precision_per_class)):
            metrics[f"precision_class_{i}"] = precision_per_class[i]
            metrics[f"recall_class_{i}"] = recall_per_class[i]
            metrics[f"f1_class_{i}"] = f1_per_class[i]

    return metrics


def compute_efficiency_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Compute efficiency metrics: inference time, throughput, memory usage.
    """

    model.eval()
    batch_times = []
    memory_usage = []
    total_samples = 0
    total_time = 0.0

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch["input_ids"], batch["attention_mask"])

    # Measurement
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch["input_ids"].size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()

            start = time.time()
            _ = model(batch["input_ids"], batch["attention_mask"], return_attention_info=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            elapsed = end - start
            batch_times.append(elapsed)
            total_time += elapsed
            total_samples += batch_size

            if device.type == "cuda":
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append((memory_after - memory_before) / 1024**2)  # MB

    avg_inference_time = np.mean(batch_times)
    std_inference_time = np.std(batch_times)

    metrics = {
        "avg_inference_time_ms": avg_inference_time * 1000,
        "std_inference_time_ms": std_inference_time * 1000,
        "throughput_samples_per_sec": total_samples / total_time if total_time > 0 else 0.0,
    }

    if device.type == "cuda" and memory_usage:
        metrics["avg_memory_usage_mb"] = float(np.mean(memory_usage))
        metrics["peak_memory_usage_mb"] = float(torch.cuda.max_memory_reserved() / 1024**2)

    return metrics


def analyze_attention_patterns(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 100,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Analyze attention patterns collected from adaptive attention layers.
    """

    model.eval()
    all_pattern_weights = []
    sequence_lengths = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(all_pattern_weights) >= num_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], return_attention_info=True)

            attention_info = outputs.get("attention_info", [])
            for layer_info in attention_info:
                if "pattern_weights" in layer_info:
                    weights = layer_info["pattern_weights"].detach().cpu().numpy()
                    # Flatten to (batch, 3)
                    if weights.ndim == 3:  # (batch, heads, 3)
                        weights = weights.mean(axis=1)
                    all_pattern_weights.extend(weights)

                    seq_lens = batch["attention_mask"].sum(dim=1).cpu().numpy()
                    sequence_lengths.extend(seq_lens)

    if not all_pattern_weights:
        return {"error": "No attention pattern data found"}

    pattern_weights = np.array(all_pattern_weights)
    sequence_lengths = np.array(sequence_lengths)

    analysis = {
        "avg_local_usage": float(np.mean(pattern_weights[:, 0])),
        "avg_global_usage": float(np.mean(pattern_weights[:, 1])),
        "avg_sparse_usage": float(np.mean(pattern_weights[:, 2])),
        "pattern_variance": np.var(pattern_weights, axis=0).tolist(),
        "pattern_correlations": np.corrcoef(pattern_weights.T).tolist(),
        "sequence_length_stats": {
            "mean": float(np.mean(sequence_lengths)),
            "std": float(np.std(sequence_lengths)),
            "min": int(np.min(sequence_lengths)),
            "max": int(np.max(sequence_lengths)),
        },
    }

    # Stratify by sequence length if enough data
    if len(sequence_lengths) > 50:
        percentiles = np.percentile(sequence_lengths, [33, 67])
        short_mask = sequence_lengths < percentiles[0]
        medium_mask = (sequence_lengths >= percentiles[0]) & (sequence_lengths < percentiles[1])
        long_mask = sequence_lengths >= percentiles[1]

        analysis["pattern_by_length"] = {
            "short_sequences": pattern_weights[short_mask].mean(axis=0).tolist() if short_mask.any() else [0, 0, 0],
            "medium_sequences": pattern_weights[medium_mask].mean(axis=0).tolist() if medium_mask.any() else [0, 0, 0],
            "long_sequences": pattern_weights[long_mask].mean(axis=0).tolist() if long_mask.any() else [0, 0, 0],
        }

    if save_path:
        create_attention_visualizations(pattern_weights, sequence_lengths, save_path)

    return analysis


def create_attention_visualizations(
    pattern_weights: np.ndarray,
    sequence_lengths: np.ndarray,
    save_path: str
):
    """Generate plots for attention pattern analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Adaptive Attention Pattern Analysis", fontsize=16, fontweight="bold")

    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(pattern_weights[:, 0], bins=30, alpha=0.6, label="Local", color="blue")
    ax1.hist(pattern_weights[:, 1], bins=30, alpha=0.6, label="Global", color="red")
    ax1.hist(pattern_weights[:, 2], bins=30, alpha=0.6, label="Sparse", color="green")
    ax1.set_title("Distribution of Attention Pattern Weights")
    ax1.legend()

    # Scatter vs sequence length
    ax2 = axes[0, 1]
    alpha = min(0.6, 1000 / len(sequence_lengths))
    ax2.scatter(sequence_lengths, pattern_weights[:, 0], alpha=alpha, label="Local", color="blue", s=10)
    ax2.scatter(sequence_lengths, pattern_weights[:, 1], alpha=alpha, label="Global", color="red", s=10)
    ax2.scatter(sequence_lengths, pattern_weights[:, 2], alpha=alpha, label="Sparse", color="green", s=10)
    ax2.set_title("Pattern Usage vs Sequence Length")
    ax2.legend()

    # Correlation heatmap
    ax3 = axes[1, 0]
    corr = np.corrcoef(pattern_weights.T)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                xticklabels=["Local", "Global", "Sparse"],
                yticklabels=["Local", "Global", "Sparse"], ax=ax3)
    ax3.set_title("Pattern Correlations")

    # Usage by bins
    ax4 = axes[1, 1]
    num_bins = 5
    seq_bins = np.linspace(sequence_lengths.min(), sequence_lengths.max(), num_bins + 1)
    bin_centers = (seq_bins[:-1] + seq_bins[1:]) / 2
    local, global_, sparse = [], [], []

    for i in range(num_bins):
        mask = (sequence_lengths >= seq_bins[i]) & (sequence_lengths < seq_bins[i + 1])
        if mask.any():
            local.append(pattern_weights[mask, 0].mean())
            global_.append(pattern_weights[mask, 1].mean())
            sparse.append(pattern_weights[mask, 2].mean())
        else:
            local.append(0)
            global_.append(0)
            sparse.append(0)

    x = np.arange(len(bin_centers))
    width = 0.25
    ax4.bar(x - width, local, width, label="Local", alpha=0.7)
    ax4.bar(x, global_, width, label="Global", alpha=0.7)
    ax4.bar(x + width, sparse, width, label="Sparse", alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{int(c)}" for c in bin_centers])
    ax4.set_title("Pattern Usage by Sequence Length")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Attention visualizations saved to {save_path}")


def benchmark_models(
    adaptive_model: nn.Module,
    baseline_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 20
) -> Dict[str, Dict[str, float]]:
    """Benchmark adaptive vs baseline models on speed, memory, and size."""

    results = {}
    for name, model in {"adaptive": adaptive_model, "baseline": baseline_model}.items():
        metrics = compute_efficiency_metrics(model, dataloader, device, num_batches)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[name] = {
            **metrics,
            "total_parameters": params,
            "trainable_parameters": trainable,
            "model_size_mb": params * 4 / (1024**2),  # float32
        }

    # Relative improvements
    if "adaptive" in results and "baseline" in results:
        improvements = {}
        base, adap = results["baseline"], results["adaptive"]

        if base.get("avg_inference_time_ms", 0) > 0:
            improvements["inference_time_improvement_percent"] = (
                (base["avg_inference_time_ms"] - adap["avg_inference_time_ms"]) / base["avg_inference_time_ms"] * 100
            )
        if base.get("throughput_samples_per_sec", 0) > 0:
            improvements["throughput_improvement_percent"] = (
                (adap["throughput_samples_per_sec"] - base["throughput_samples_per_sec"]) / base["throughput_samples_per_sec"] * 100
            )

        results["improvements"] = improvements

    return results
