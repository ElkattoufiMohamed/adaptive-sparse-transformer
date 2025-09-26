"""
Professional training infrastructure for transformer experiments.
Rewritten with safer AMP handling, cleaner logging, and robust scheduler/optimizer interplay.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: efficiency metrics helper (may be unused)
try:
    from evaluation.metrics import compute_efficiency_metrics
except Exception:
    compute_efficiency_metrics = None

from utils.helpers import load_model, save_model, set_seed

logger = logging.getLogger(__name__)


class AdaptiveTransformerTrainer:
    """
    Robust trainer for adaptive sparse transformer experiments.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: Path,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.use_wandb = use_wandb

        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        training_config = config["training"]
        self.num_epochs = int(training_config.get("num_epochs", 1))
        self.gradient_clip_norm = float(training_config.get("gradient_clip_norm", 0.0))
        self.eval_steps = int(config.get("evaluation", {}).get("eval_steps", 500))
        self.save_steps = int(config.get("evaluation", {}).get("save_steps", 1000))
        self.logging_steps = int(config.get("evaluation", {}).get("logging_steps", 100))
        self.loss_name = training_config.get("loss", "cross_entropy").lower()

        # Initialize optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Mixed precision training - follow config explicitly
        hw_cfg = config.get("hardware", {})
        self.use_amp = bool(hw_cfg.get("mixed_precision", False))
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = -np.inf
        self.train_metrics: List[Dict[str, float]] = []
        self.eval_metrics: List[Dict[str, float]] = []

        # Initialize wandb if requested
        if self.use_wandb:
            self._setup_wandb()

        logger.info(f"Trainer initialized - Device: {device}, AMP: {self.use_amp}")

    # ----------------------
    # Setup helpers
    # ----------------------
    def _setup_optimizer(self) -> None:
        """Initialize optimizer with separate learning rates for pattern selector."""
        training_config = self.config["training"]
        
        # Separate pattern selector parameters from others
        pattern_params = []
        pattern_param_names = []
        other_params = []
        other_param_names = []
        
        for name, param in self.model.named_parameters():
            if 'pattern_selector' in name or 'pattern_bias' in name:
                pattern_params.append(param)
                pattern_param_names.append(name)
            else:
                other_params.append(param)
                other_param_names.append(name)
        
        # Log parameter groups
        logger.info(f"Pattern selector parameters ({len(pattern_params)}): {pattern_param_names[:3]}...")
        logger.info(f"Other parameters: {len(other_params)}")
        
        # Create parameter groups with different learning rates
        base_lr = float(training_config.get("learning_rate", 1e-4))
        pattern_lr_multiplier = float(training_config.get("pattern_lr_multiplier", 10.0))
        
        optimizer_grouped_parameters = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': float(training_config.get("weight_decay", 0.01))
            },
            {
                'params': pattern_params,
                'lr': base_lr * pattern_lr_multiplier,  # 10x higher LR by default
                'weight_decay': 0.0,  # No weight decay for pattern selector
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        logger.info(f"Optimizer: AdamW with base LR={base_lr}, pattern selector LR={base_lr * pattern_lr_multiplier}")

    def _setup_scheduler(self) -> None:
        """Initialize the learning rate scheduler (transformers schedulers)."""
        scheduler_config = self.config.get("scheduler", {})
        schedule_type = scheduler_config.get("type", "cosine")

        total_steps = max(1, len(self.train_loader) * self.num_epochs)
        warmup_steps = int(self.config["training"].get("warmup_steps", 0))
        # If warmup > total steps, clamp it to a small fraction
        if warmup_steps >= total_steps:
            logger.warning("warmup_steps >= total_steps; clamping warmup_steps to 10% of total_steps")
            warmup_steps = max(1, total_steps // 10)

        try:
            if schedule_type == "cosine":
                from transformers import get_cosine_schedule_with_warmup

                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
            elif schedule_type == "linear":
                from transformers import get_linear_schedule_with_warmup

                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
            else:
                self.scheduler = None
        except Exception as e:
            logger.exception("Failed to create scheduler from transformers; scheduler disabled. Error: %s", e)
            self.scheduler = None

        logger.info(f"Scheduler: {schedule_type}, Warmup: {warmup_steps} steps, Total steps: {total_steps}")

    def _setup_wandb(self) -> None:
        """Initialize weights & biases (if configured)."""
        wandb_config = self.config.get("wandb", {})
        # Provide a stable name
        run_name = wandb_config.get("name", None)
        wandb.init(
            project=wandb_config.get("project", "adaptive-sparse-transformer"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=self.config,
        )
        wandb.watch(self.model, log="all", log_freq=1000)
        logger.info("Wandb initialized")

    # ----------------------
    # Main training step
    # ----------------------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with pattern loss integration."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Debug attention masks to detect all-zero masks
        if "attention_mask" in batch:
            mask_sums = batch["attention_mask"].sum(dim=1)
            logger.debug(
                "attention_mask token counts - min:%d mean:%.2f max:%d",
                int(mask_sums.min().item()),
                float(mask_sums.float().mean().item()),
                int(mask_sums.max().item()),
            )

        # Forward (with AMP if enabled)
        with autocast(enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"], 
                attention_mask=batch.get("attention_mask", None), 
                return_attention_info=True
            )
            logits = outputs["logits"]

            # Task loss
            if self.loss_name == "cross_entropy":
                loss_f = nn.CrossEntropyLoss()
                task_loss = loss_f(logits, batch["labels"])
            else:
                loss_f = nn.CrossEntropyLoss()
                task_loss = loss_f(logits, batch["labels"])
            
            # Aggregate pattern losses from all attention layers
            pattern_loss = torch.tensor(0.0, device=self.device, dtype=task_loss.dtype)
            pattern_metrics = {}
            
            if 'attention_info' in outputs and outputs['attention_info']:
                total_pattern_loss = 0.0
                num_valid_layers = 0
                
                for i, layer_info in enumerate(outputs['attention_info']):
                    if layer_info and 'total_pattern_loss' in layer_info:
                        # Convert to tensor if needed
                        layer_pattern_loss = layer_info['total_pattern_loss']
                        if not isinstance(layer_pattern_loss, torch.Tensor):
                            layer_pattern_loss = torch.tensor(layer_pattern_loss, device=self.device)
                        
                        total_pattern_loss += layer_pattern_loss
                        num_valid_layers += 1
                        
                        # Track individual loss components from first layer for logging
                        if i == 0:
                            for key in ['diversity_loss', 'variance_loss', 'consistency_loss', 
                                    'underuse_penalty', 'decisiveness_loss']:
                                if key in layer_info:
                                    val = layer_info[key]
                                    if isinstance(val, torch.Tensor):
                                        pattern_metrics[f'pattern_{key}'] = float(val.item())
                                    else:
                                        pattern_metrics[f'pattern_{key}'] = float(val)
                
                # Average pattern loss across layers
                if num_valid_layers > 0:
                    pattern_loss = total_pattern_loss / num_valid_layers
            
            # Ramp up pattern loss weight over training
            # Start small and gradually increase to full weight
            ramp_up_steps = 5000
            max_pattern_weight = 0.2  # Maximum weight for pattern loss
            ramp_progress = min(1.0, self.global_step / ramp_up_steps)
            pattern_weight = max_pattern_weight * (1 - np.exp(-3 * ramp_progress))
            
            # Combined loss
            loss = task_loss + pattern_weight * pattern_loss

        # Log numeric checks
        logger.debug("Task loss = %.6f, Pattern loss = %.6f, Total loss = %.6f", 
                    float(task_loss.item()), float(pattern_loss.item()), float(loss.item()))
        logger.debug("Pattern weight = %.4f", pattern_weight)
        logger.debug("Loss finite: %s", torch.isfinite(loss).item())

        if not torch.isfinite(loss):
            logger.critical("NaN/Inf loss detected - skipping this batch (global_step=%d)", self.global_step)
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": float("nan"), "accuracy": 0.0, "learning_rate": self.optimizer.param_groups[0]["lr"]}

        # Backprop
        if self.use_amp:
            assert self.scaler is not None
            self.scaler.scale(loss).backward()

            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)

            if self.gradient_clip_norm and self.gradient_clip_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                logger.debug("Gradient norm after clipping (AMP): %.6f", float(total_norm))
            
            # Check pattern selector gradients specifically
            for name, param in self.model.named_parameters():
                if 'pattern_selector' in name and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if self.global_step % 100 == 0:  # Log every 100 steps
                        logger.debug(f"Pattern selector gradient norm for {name}: {grad_norm:.8f}")
                    break
            
            # step optimizer w/ scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip_norm and self.gradient_clip_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                logger.debug("Gradient norm after clipping: %.6f", float(total_norm))
            
            # Check pattern selector gradients
            for name, param in self.model.named_parameters():
                if 'pattern_selector' in name and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if self.global_step % 100 == 0:
                        logger.debug(f"Pattern selector gradient norm for {name}: {grad_norm:.8f}")
                    break
            
            self.optimizer.step()

        # scheduler step after optimizer step
        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except Exception as e:
                logger.exception("Scheduler step failed: %s", e)

        # zero grads
        self.optimizer.zero_grad(set_to_none=True)

        # Compute accuracy (no grad)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = float((preds == batch["labels"]).float().mean().item())

        # gather attention stats
        attention_info = outputs.get("attention_info", [])
        avg_attention_stats = self._compute_attention_stats(attention_info)

        step_metrics: Dict[str, float] = {
            "loss": float(loss.item()),
            "task_loss": float(task_loss.item()),
            "pattern_loss": float(pattern_loss.item()),
            "pattern_weight": float(pattern_weight),
            "accuracy": accuracy,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "pattern_lr": float(self.optimizer.param_groups[1]["lr"]) if len(self.optimizer.param_groups) > 1 else 0.0,
            **avg_attention_stats,
            **pattern_metrics
        }

        return step_metrics

    # ----------------------
    # Attention stats
    # ----------------------
    def _compute_attention_stats(self, attention_info: List[Dict]) -> Dict[str, float]:
        """Compute average attention pattern statistics across all layers in the batch."""
        if not attention_info:
            return {}

        local_ratios, global_ratios, sparse_ratios = [], [], []
        for layer_info in attention_info:
            if not layer_info:
                continue
            if "local_ratio" in layer_info:
                local_ratios.append(layer_info["local_ratio"])
                global_ratios.append(layer_info["global_ratio"])
                sparse_ratios.append(layer_info.get("sparse_ratio", 0.0))

        if local_ratios:
            return {
                "avg_local_ratio": float(np.mean(local_ratios)),
                "avg_global_ratio": float(np.mean(global_ratios)),
                "avg_sparse_ratio": float(np.mean(sparse_ratios)),
            }
        return {}

    # ----------------------
    # Evaluation
    # ----------------------
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set and return metrics."""
        self.model.eval()

        total_loss = 0.0
        all_predictions: List[int] = []
        all_labels: List[int] = []
        all_attention_stats: List[Dict[str, float]] = []

        logger.info("Running evaluation...")

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None), return_attention_info=True)
                logits = outputs["logits"]

                # compute loss
                loss_f = nn.CrossEntropyLoss()
                loss = loss_f(logits, batch["labels"])
                total_loss += float(loss.item())

                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())

                attention_info = outputs.get("attention_info", [])
                batch_attention_stats = self._compute_attention_stats(attention_info)
                if batch_attention_stats:
                    all_attention_stats.append(batch_attention_stats)

        avg_loss = total_loss / max(1, len(self.eval_loader))
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="weighted")

        # average attention statistics
        avg_attention_stats: Dict[str, float] = {}
        if all_attention_stats:
            for key in all_attention_stats[0].keys():
                values = [s.get(key, 0.0) for s in all_attention_stats if key in s]
                avg_attention_stats[f"eval_{key}"] = float(np.mean(values))

        eval_metrics: Dict[str, float] = {
            "eval_loss": float(avg_loss),
            "eval_accuracy": float(accuracy),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "eval_f1": float(f1),
            **avg_attention_stats,
        }

        # Optional: compute efficiency metrics if helper available
        if compute_efficiency_metrics is not None and self.config.get("evaluation", {}).get("compute_efficiency", False):
            try:
                eff = compute_efficiency_metrics(self.model, self.eval_loader, device=self.device, num_batches=10)
                eval_metrics.update(eff)
            except Exception:
                logger.exception("compute_efficiency_metrics failed; skipping.")

        logger.info(
            "Evaluation - Loss: %.4f, Accuracy: %.4f, F1: %.4f",
            eval_metrics["eval_loss"],
            eval_metrics["eval_accuracy"],
            eval_metrics["eval_f1"],
        )

        self.model.train()
        return eval_metrics

    # ----------------------
    # Checkpointing
    # ----------------------
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint with metrics and optimizer state."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
            "best_eval_metric": self.best_eval_metric,
        }

        checkpoint_path = self.experiment_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("New best model saved with metric: %.4f", metrics.get("eval_accuracy", 0.0))

        logger.info("Checkpoint saved to %s", checkpoint_path)

    # ----------------------
    # Main training loop
    # ----------------------
    def train(self) -> Dict[str, Any]:
        logger.info("Starting training...")
        logger.info("Total epochs: %d", self.num_epochs)
        logger.info("Steps per epoch: %d", len(self.train_loader))
        logger.info("Total steps: %d", len(self.train_loader) * self.num_epochs)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info("\n📚 Epoch %d/%d", epoch + 1, self.num_epochs)

            epoch_metrics: List[Dict[str, float]] = []
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            logger.debug("Starting epoch %d, about to iterate through %d batches", epoch + 1, len(self.train_loader))

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    logger.debug("Processing batch %d/%d", batch_idx + 1, len(self.train_loader))
                    step_metrics = self.train_step(batch)
                    logger.debug("train_step completed for batch %d", batch_idx + 1)

                    epoch_metrics.append(step_metrics)
                    self.global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": f"{step_metrics.get('loss', float('nan')):.4f}",
                            "acc": f"{step_metrics.get('accuracy', 0.0):.3f}",
                            "lr": f"{step_metrics.get('learning_rate', 0.0):.2e}",
                        }
                    )

                    # Logging (wandb + internal)
                    if self.global_step % self.logging_steps == 0:
                        logger.debug("Logging at step %d", self.global_step)
                        recent = epoch_metrics[-self.logging_steps :]
                        avg_metrics = self._average_metrics(recent)
                        self.train_metrics.append(avg_metrics)
                        if self.use_wandb:
                            wandb.log({f"train_{k}": v for k, v in avg_metrics.items()}, step=self.global_step)

                    # Evaluation
                    if self.global_step % self.eval_steps == 0:
                        logger.debug("Starting evaluation at step %d", self.global_step)
                        eval_metrics = self.evaluate()
                        self.eval_metrics.append(eval_metrics)

                        current_metric = eval_metrics.get("eval_accuracy", 0.0)
                        is_best = current_metric > self.best_eval_metric
                        if is_best:
                            self.best_eval_metric = current_metric

                        if self.use_wandb:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Save checkpoint if requested
                        if self.global_step % self.save_steps == 0:
                            logger.debug("Saving checkpoint at step %d", self.global_step)
                            self.save_checkpoint(eval_metrics, is_best=is_best)

                except Exception as e:
                    logger.exception("Exception at batch %d: %s", batch_idx + 1, e)
                    # Save a small snapshot for debugging
                    try:
                        err_path = self.experiment_dir / f"error_snapshot_step_{self.global_step}.pt"
                        torch.save(
                            {
                                "exception": str(e),
                                "batch_idx": batch_idx,
                                "config": self.config,
                                "model_state_dict": self.model.state_dict(),
                            },
                            err_path,
                        )
                        logger.info("Saved error snapshot to %s", err_path)
                    except Exception:
                        logger.exception("Failed to save error snapshot.")
                    raise

        # Final evaluation & checkpoint
        logger.info("🏁 Training completed! Running final evaluation...")
        final_eval_metrics = self.evaluate()
        self.save_checkpoint(final_eval_metrics, is_best=False)

        training_time = time.time() - start_time
        logger.info("Total training time: %.2f seconds", training_time)

        summary = {
            "final_eval_metrics": final_eval_metrics,
            "best_eval_metric": self.best_eval_metric,
            "total_steps": self.global_step,
            "training_time": training_time,
            "train_metrics_history": self.train_metrics,
            "eval_metrics_history": self.eval_metrics,
        }

        summary_path = self.experiment_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb:
            wandb.log({"final_eval_accuracy": final_eval_metrics.get("eval_accuracy", 0.0)})
            wandb.finish()

        return summary

    # ----------------------
    # Utilities
    # ----------------------
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        avg: Dict[str, float] = {}
        for key in metrics_list[0].keys():
            vals = [m.get(key, 0.0) for m in metrics_list if key in m]
            if vals:
                avg[key] = float(np.mean(vals))
        return avg


class BaselineTransformerTrainer(AdaptiveTransformerTrainer):
    """Trainer for baseline transformer (no adaptive attention stats)."""

    def _compute_attention_stats(self, attention_info: List[Dict]) -> Dict[str, float]:
        return {}


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: Dict[str, Any],
    experiment_name: str,
    device: Optional[torch.device] = None,
) -> AdaptiveTransformerTrainer:
    """Factory function to create the appropriate trainer class."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path("results") / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.__class__.__name__
    trainer_class = AdaptiveTransformerTrainer if "Adaptive" in model_name else BaselineTransformerTrainer

    trainer = trainer_class(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        use_wandb=config.get("wandb", {}).get("project") is not None,
    )

    logger.info("Created %s for %s", trainer_class.__name__, experiment_name)
    return trainer
