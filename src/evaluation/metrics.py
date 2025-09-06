"""
Comprehensive evaluation metrics for transformer models.
Includes standard ML metrics plus efficiency analysis.
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_classification_metrics(
    predictions: List[int], 
    labels: List[int], 
    num_classes: int = 2
) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Add per-class metrics for detailed analysis
    if num_classes <= 10:  # Only for small number of classes
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        for i in range(len(precision_per_class)):
            metrics[f'precision_class_{i}'] = precision_per_class[i]
            metrics[f'recall_class_{i}'] = recall_per_class[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    return metrics

def compute_efficiency_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Compute efficiency metrics: FLOPs, memory usage, inference time.
    
    This is crucial for comparing our adaptive attention with baseline!
    """
    
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Warmup
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch['input_ids'], batch['attention_mask'])
    
    # Measure inference time
    inference_times = []
    memory_usage = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Measure memory before
            if device.type == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # Measure inference time
            start_time = time.time()
            
            outputs = model(
                batch['input_ids'], 
                batch['attention_mask'],
                return_attention_info=True
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Measure memory after
            if device.type == 'cuda':
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append((memory_after - memory_before) / 1024**2)  # MB
    
    # Compute statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    metrics = {
        'avg_inference_time_ms': avg_inference_time * 1000,
        'std_inference_time_ms': std_inference_time * 1000,
        'throughput_samples_per_sec': len(batch['input_ids']) / avg_inference_time,
    }
    
    if device.type == 'cuda' and memory_usage:
        metrics['avg_memory_usage_mb'] = np.mean(memory_usage)
        metrics['peak_memory_usage_mb'] = np.max(memory_usage)
    
    return metrics

def analyze_attention_patterns(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 100,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Analyze attention patterns used by adaptive attention mechanism.
    
    This creates the visualizations that will wow interviewers!
    """
    
    model.eval()
    
    all_pattern_weights = []
    sequence_lengths = []
    sample_texts = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(all_pattern_weights) >= num_samples:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                batch['input_ids'],
                batch['attention_mask'],
                return_attention_info=True
            )
            
            attention_info = outputs.get('attention_info', [])
            
            # Extract pattern weights from each layer
            for layer_idx, layer_info in enumerate(attention_info):
                if 'pattern_weights' in layer_info:
                    pattern_weights = layer_info['pattern_weights'].cpu().numpy()
                    all_pattern_weights.extend(pattern_weights)
                    
                    # Track sequence lengths for analysis
                    seq_lens = batch['attention_mask'].sum(dim=1).cpu().numpy()
                    sequence_lengths.extend(seq_lens)
    
    if not all_pattern_weights:
        return {"error": "No attention pattern data found"}
    
    # Convert to numpy arrays
    pattern_weights = np.array(all_pattern_weights)  # Shape: (num_samples, 3)
    sequence_lengths = np.array(sequence_lengths)
    
    # Compute statistics
    analysis = {
        'avg_local_usage': np.mean(pattern_weights[:, 0]),
        'avg_global_usage': np.mean(pattern_weights[:, 1]),
        'avg_sparse_usage': np.mean(pattern_weights[:, 2]),
        'pattern_variance': np.var(pattern_weights, axis=0),
        'pattern_correlations': np.corrcoef(pattern_weights.T),
        'sequence_length_stats': {
            'mean': np.mean(sequence_lengths),
            'std': np.std(sequence_lengths),
            'min': np.min(sequence_lengths),
            'max': np.max(sequence_lengths)
        }
    }
    
    # Analyze pattern usage vs sequence length
    if len(sequence_lengths) > 50:
        # Bin by sequence length
        short_mask = sequence_lengths < np.percentile(sequence_lengths, 33)
        medium_mask = (sequence_lengths >= np.percentile(sequence_lengths, 33)) & \
                     (sequence_lengths < np.percentile(sequence_lengths, 67))
        long_mask = sequence_lengths >= np.percentile(sequence_lengths, 67)
        
        analysis['pattern_by_length'] = {
            'short_sequences': {
                'avg_local': np.mean(pattern_weights[short_mask, 0]),
                'avg_global': np.mean(pattern_weights[short_mask, 1]),
                'avg_sparse': np.mean(pattern_weights[short_mask, 2])
            },
            'medium_sequences': {
                'avg_local': np.mean(pattern_weights[medium_mask, 0]),
                'avg_global': np.mean(pattern_weights[medium_mask, 1]),
                'avg_sparse': np.mean(pattern_weights[medium_mask, 2])
            },
            'long_sequences': {
                'avg_local': np.mean(pattern_weights[long_mask, 0]),
                'avg_global': np.mean(pattern_weights[long_mask, 1]),
                'avg_sparse': np.mean(pattern_weights[long_mask, 2])
            }
        }
    
    # Create visualizations if save_path provided
    if save_path:
        create_attention_visualizations(pattern_weights, sequence_lengths, save_path)
    
    return analysis

def create_attention_visualizations(
    pattern_weights: np.ndarray,
    sequence_lengths: np.ndarray,
    save_path: str
):
    """Create comprehensive attention pattern visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adaptive Attention Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pattern distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(pattern_weights[:, 0], alpha=0.7, label='Local', bins=30, color='blue')
    ax1.hist(pattern_weights[:, 1], alpha=0.7, label='Global', bins=30, color='red')
    ax1.hist(pattern_weights[:, 2], alpha=0.7, label='Sparse', bins=30, color='green')
    ax1.set_xlabel('Pattern Weight')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Attention Pattern Weights')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pattern usage vs sequence length
    ax2 = axes[0, 1]
    scatter_alpha = min(0.6, 1000 / len(sequence_lengths))
    ax2.scatter(sequence_lengths, pattern_weights[:, 0], alpha=scatter_alpha, 
                label='Local', color='blue', s=10)
    ax2.scatter(sequence_lengths, pattern_weights[:, 1], alpha=scatter_alpha, 
                label='Global', color='red', s=10)
    ax2.scatter(sequence_lengths, pattern_weights[:, 2], alpha=scatter_alpha, 
                label='Sparse', color='green', s=10)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Pattern Weight')
    ax2.set_title('Pattern Usage vs Sequence Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Pattern correlation heatmap
    ax3 = axes[1, 0]
    correlation_matrix = np.corrcoef(pattern_weights.T)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=['Local', 'Global', 'Sparse'],
                yticklabels=['Local', 'Global', 'Sparse'], ax=ax3)
    ax3.set_title('Pattern Weight Correlations')
    
    # 4. Average pattern usage by sequence length bins
    ax4 = axes[1, 1]
    
    # Create bins
    num_bins = 5
    seq_len_bins = np.linspace(sequence_lengths.min(), sequence_lengths.max(), num_bins + 1)
    bin_centers = (seq_len_bins[:-1] + seq_len_bins[1:]) / 2
    
    local_avgs = []
    global_avgs = []
    sparse_avgs = []
    
    for i in range(num_bins):
        mask = (sequence_lengths >= seq_len_bins[i]) & (sequence_lengths < seq_len_bins[i + 1])
        if np.sum(mask) > 0:
            local_avgs.append(np.mean(pattern_weights[mask, 0]))
            global_avgs.append(np.mean(pattern_weights[mask, 1]))
            sparse_avgs.append(np.mean(pattern_weights[mask, 2]))
        else:
            local_avgs.append(0)
            global_avgs.append(0)
            sparse_avgs.append(0)
    
    x = np.arange(len(bin_centers))
    width = 0.25
    
    ax4.bar(x - width, local_avgs, width, label='Local', color='blue', alpha=0.7)
    ax4.bar(x, global_avgs, width, label='Global', color='red', alpha=0.7)
    ax4.bar(x + width, sparse_avgs, width, label='Sparse', color='green', alpha=0.7)
    
    ax4.set_xlabel('Sequence Length Bins')
    ax4.set_ylabel('Average Pattern Weight')
    ax4.set_title('Pattern Usage by Sequence Length')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(c)}' for c in bin_centers])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention pattern visualizations saved to {save_path}")

def benchmark_models(
    adaptive_model: nn.Module,
    baseline_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive benchmarking of adaptive vs baseline models.
    
    This creates the comparison data that proves our approach works!
    """
    
    print("🔬 Benchmarking models...")
    
    models = {
        'adaptive': adaptive_model,
        'baseline': baseline_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Benchmarking {model_name} model...")
        
        # Standard evaluation metrics
        eval_metrics = compute_efficiency_metrics(model, dataloader, device, num_batches)
        
        # Model complexity metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        complexity_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        }
        
        results[model_name] = {
            **eval_metrics,
            **complexity_metrics
        }
    
    # Compute relative improvements
    if 'adaptive' in results and 'baseline' in results:
        adaptive_results = results['adaptive']
        baseline_results = results['baseline']
        
        improvements = {}
        for metric in ['avg_inference_time_ms', 'throughput_samples_per_sec']:
            if metric in results:
                if metric == 'avg_inference_time_ms':
                    # Lower is better for inference time
                    improvement = (baseline_results[metric] - adaptive_results[metric]) / baseline_results[metric] * 100
                    improvements[f'{metric}_improvement_percent'] = improvement
                else:
                    # Higher is better for throughput
                    improvement = (adaptive_results[metric] - baseline_results[metric]) / baseline_results[metric] * 100
                    improvements[f'{metric}_improvement_percent'] = improvement
        
        results['improvements'] = improvements
    
    return results