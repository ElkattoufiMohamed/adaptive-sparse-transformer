# src/evaluation/metrics.py
# Clean, dependency-light metrics & benchmarking utilities.
# - No ellipses/truncations
# - Stable imports only (torch, numpy)
# - Exact keys used by train_adaptive.py ("improvements", "throughput_improvement_percent", etc.)

from typing import Dict, Any, Optional, List, Tuple
import time
import math

import torch
import numpy as np


@torch.no_grad()
def _model_step(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs a forward step and returns (logits, labels).
    Supports typical HF-style batches: input_ids, attention_mask, labels.
    Falls back to passing the whole batch as **kwargs (except 'labels').
    """
    labels = None
    if "labels" in batch:
        labels = batch["labels"].to(device)

    # Move tensors to device
    kw = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and k != "labels":
            kw[k] = v.to(device)

    logits = model(**kw)
    # If model returns a tuple or dict, try common keys
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    elif isinstance(logits, dict):
        logits = logits.get("logits", next(iter(logits.values())))

    return logits, labels


def _preds_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        # e.g., sequence classification with pooled logits in last dim; take last position
        logits = logits[:, -1, :]
    return logits.argmax(dim=-1)


def _compute_confusion(preds: np.ndarray, labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    if num_classes is None:
        num_classes = int(max(preds.max(initial=0), labels.max(initial=0))) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y, p] += 1
    return cm


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _macro_f1(cm: np.ndarray) -> float:
    # per-class precision/recall/f1, then unweighted mean over classes that appear
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        # include class even if absent; macro-F1 commonly includes all
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate_classification(model: torch.nn.Module,
                            dataloader,
                            device: torch.device,
                            num_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Basic classification metrics: accuracy and macro-F1 on up to num_batches of dataloader.
    """
    model.eval()
    preds_all: List[int] = []
    labels_all: List[int] = []

    for i, batch in enumerate(dataloader):
        logits, labels = _model_step(model, batch, device)
        if labels is None:
            # If labels are missing, we cannot compute accuracy/F1.
            break
        preds = _preds_from_logits(logits)
        preds_all.extend(preds.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

        if num_batches is not None and (i + 1) >= num_batches:
            break

    if not preds_all or not labels_all:
        return {"eval_accuracy": 0.0, "eval_f1": 0.0}

    preds_np = np.asarray(preds_all, dtype=np.int64)
    labels_np = np.asarray(labels_all, dtype=np.int64)
    accuracy = float((preds_np == labels_np).mean())
    cm = _compute_confusion(preds_np, labels_np)
    f1 = _macro_f1(cm)

    return {"eval_accuracy": accuracy, "eval_f1": f1}


@torch.no_grad()
def measure_inference_time(model: torch.nn.Module,
                           dataloader,
                           device: torch.device,
                           num_batches: int = 50,
                           warmup: int = 5) -> Dict[str, float]:
    """
    Measures average per-batch inference time (seconds) with a small warmup.
    """
    model.eval()
    # Warmup
    it = iter(dataloader)
    for _ in range(min(warmup, num_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        _model_step(model, batch, device)

    # Timed
    start = time.perf_counter()
    total_batches = 0
    it = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        _model_step(model, batch, device)
        total_batches += 1
    elapsed = time.perf_counter() - start
    avg_batch_time = elapsed / total_batches if total_batches else float("inf")
    return {"avg_batch_inference_time_sec": float(avg_batch_time)}


@torch.no_grad()
def measure_throughput(model: torch.nn.Module,
                       dataloader,
                       device: torch.device,
                       num_batches: int = 50,
                       warmup: int = 5) -> Dict[str, float]:
    """
    Measures samples/sec over num_batches (after warmup).
    """
    model.eval()
    # Warmup
    it = iter(dataloader)
    for _ in range(min(warmup, num_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        _model_step(model, batch, device)

    # Timed throughput
    start = time.perf_counter()
    seen = 0
    total_batches = 0
    it = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        bsz = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                bsz = v.size(0)
                break
        if bsz is None:
            # fallback if batch is weird
            bsz = 1

        _model_step(model, batch, device)
        seen += int(bsz)
        total_batches += 1

    elapsed = time.perf_counter() - start
    tps = seen / elapsed if elapsed > 0 else 0.0
    return {
        "throughput_samples_per_sec": float(tps),
        "measured_batches": int(total_batches),
        "measured_samples": int(seen),
        "elapsed_sec": float(elapsed),
    }


@torch.no_grad()
def analyze_attention_patterns(model: torch.nn.Module,
                               dataloader,
                               device: torch.device,
                               num_batches: int = 10) -> Dict[str, Any]:
    """
    Best-effort attention diagnostics. If the model exposes attention weights or
    a debug dict (e.g., model.last_attention_info), summarize them. Otherwise,
    return an empty-but-valid report.
    """
    model.eval()
    report: Dict[str, Any] = {
        "available": False,
        "num_batches": 0,
        "summary": {},
    }

    # Heuristic: check typical attributes/hooks some adaptive models expose
    # e.g., model.enable_attention_debug(), model.clear_attention_debug(), model.attention_debug
    enable_debug = getattr(model, "enable_attention_debug", None)
    clear_debug = getattr(model, "clear_attention_debug", None)
    fetch_debug = lambda m: getattr(m, "attention_debug", None)

    if callable(enable_debug):
        enable_debug(True)
    if callable(clear_debug):
        clear_debug()

    batches_seen = 0
    for i, batch in enumerate(dataloader):
        _model_step(model, batch, device)
        batches_seen += 1
        if (i + 1) >= num_batches:
            break

    dbg = fetch_debug(model)
    if dbg is None:
        # Try a second convention: model.last_attention_info
        dbg = getattr(model, "last_attention_info", None)

    if dbg is not None:
        report["available"] = True
        # Summarize a few common fields if present
        summary: Dict[str, Any] = {}
        if isinstance(dbg, dict):
            # Example optional fields:
            #   "pattern_weights": (B, P)
            #   "mask_density": float or list
            #   "avg_local_span": float
            # Collect safe stats:
            pw = dbg.get("pattern_weights", None)
            if isinstance(pw, torch.Tensor):
                pw = pw.detach().float().mean(dim=0).cpu().tolist()
                summary["pattern_weights_mean"] = pw
            md = dbg.get("mask_density", None)
            if isinstance(md, (list, tuple)):
                try:
                    summary["mask_density_mean"] = float(np.mean(md))
                except Exception:
                    pass
            elif isinstance(md, (int, float)):
                summary["mask_density_mean"] = float(md)
            als = dbg.get("avg_local_span", None)
            if isinstance(als, (int, float)):
                summary["avg_local_span"] = float(als)
        report["summary"] = summary

    report["num_batches"] = batches_seen
    # Cleanup
    if callable(enable_debug):
        enable_debug(False)
    return report


@torch.no_grad()
def benchmark_models(adaptive_model: torch.nn.Module,
                     baseline_model: torch.nn.Module,
                     dataloader,
                     device: torch.device,
                     num_batches: int = 50) -> Dict[str, Any]:
    """
    Benchmarks adaptive vs baseline: accuracy/F1, avg batch inference time, throughput.
    Returns a dict with per-model metrics and an 'improvements' section (percent deltas).
    Positive improvement means adaptive is better (higher), negative for time (lower is better).
    """
    # Accuracy / F1 (use same slice for fairness)
    adaptive_cls = evaluate_classification(adaptive_model, dataloader, device, num_batches=num_batches)
    baseline_cls = evaluate_classification(baseline_model, dataloader, device, num_batches=num_batches)

    # Inference time
    adaptive_time = measure_inference_time(adaptive_model, dataloader, device, num_batches=num_batches)
    baseline_time = measure_inference_time(baseline_model, dataloader, device, num_batches=num_batches)

    # Throughput
    adaptive_tp = measure_throughput(adaptive_model, dataloader, device, num_batches=num_batches)
    baseline_tp = measure_throughput(baseline_model, dataloader, device, num_batches=num_batches)

    def pct_impr(adapt: float, base: float, higher_is_better: bool) -> float:
        # percent change relative to baseline
        if base == 0:
            return 0.0
        raw = (adapt - base) / base * 100.0
        return float(raw if higher_is_better else -raw)  # invert sign for "lower is better"

    improvements = {
        "accuracy_improvement_percent": pct_impr(adaptive_cls["eval_accuracy"],  baseline_cls["eval_accuracy"],  True),
        "f1_improvement_percent":       pct_impr(adaptive_cls["eval_f1"],        baseline_cls["eval_f1"],        True),
        "inference_time_improvement_percent": pct_impr(adaptive_time["avg_batch_inference_time_sec"],
                                                       baseline_time["avg_batch_inference_time_sec"], False),
        "throughput_improvement_percent":     pct_impr(adaptive_tp["throughput_samples_per_sec"],
                                                       baseline_tp["throughput_samples_per_sec"], True),
    }

    return {
        "adaptive": {
            **adaptive_cls,
            **adaptive_time,
            **adaptive_tp,
        },
        "baseline": {
            **baseline_cls,
            **baseline_time,
            **baseline_tp,
        },
        "improvements": improvements,
        "settings": {
            "num_batches": int(num_batches),
        }
    }
